"""RAG chatbot over the player-clustering outputs.

    pip install openai python-dotenv
    # put OPENAI_API_KEY=sk-... in chatbot/.env (or in the repo root .env)
    python chatbot/build_db.py     # once
    python chatbot/chat.py         # then chat

Architecture -- two retrieval paths, the model routes between them:

    "top 10 strikers by xG"        -> query_players  (SQL, exact aggregation)
    "why wasn't silhouette used?"  -> search_docs    (BM25 over summary.md)
    "is Messi elite?"              -> both

Embedding the 19,729 player rows and doing vector search would answer none of
the first kind: ORDER BY and COUNT are not similarity operations. So the numbers
live in SQLite and only the prose gets retrieved.
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from retriever import Retriever

HERE = Path(__file__).resolve().parent
DB = HERE / "players.db"

# Load OPENAI_API_KEY from .env. We look in chatbot/.env first, then fall back
# to the repo root .env, so the key can sit alongside the script or at the
# project root -- whichever the user prefers. load_dotenv is a no-op if the
# file doesn't exist, and existing shell exports win over the file (override=False).
load_dotenv(HERE / ".env")
load_dotenv(HERE.parent / ".env")

# Verbose is opt-in: by default the chat loop only prints the bot's final
# answer. Flip on with --verbose / -v, or set VERBOSE=1 in the environment
# (handy for shell sessions where you always want the trace). CLI wins over env.
VERBOSE = os.environ.get("VERBOSE") == "1"


def vprint(*args, **kwargs) -> None:
    """Print only when verbose mode is on."""
    if VERBOSE:
        print(*args, **kwargs)

MODEL = "gpt-4o-mini"
MAX_ROWS = 50          # cap on rows returned to the model, to bound context
MAX_DOC_CHARS = 2000   # cap on each retrieved chunk

_retriever: Retriever | None = None


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

# Only a single read-only statement is ever allowed through. The model is not a
# trusted SQL author: it can be steered by anything in the data it reads, so the
# guard is structural (read-only connection + statement allowlist), not a
# prompt instruction.
_READONLY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(attach|pragma|insert|update|delete|drop|alter|create|replace|vacuum)\b",
    re.IGNORECASE,
)


def query_players(sql: str) -> str:
    """Run a read-only SQL query against the player database and return the rows.

    Use this for anything numeric: counts, rankings, filters, aggregates,
    per-cluster or per-position statistics, or looking up a specific player.

    Tables:
      players                  one row per player (19,729 rows)
      kmeans_summary           per position: n, pca_dims, chosen k, silhouette
      kmeans_evaluation        stability sweep: bootstrap_ari, min_jaccard, mv_lift per (position, k)
      kmeans_cluster_stability per-cluster bootstrap Jaccard
      kmeans_k_sweep           silhouette / Davies-Bouldin / CH / inertia for k in 2..10

    Notes that change the right query:
      - kmeans_cluster ids are only meaningful within a position_group. Always
        group or filter by position_group alongside kmeans_cluster.
      - latest_market_value and highest_market_value are unreliable for star
        players -- Messi, Haaland and the rest of the elite cohort read 0 even
        though the cluster has them as Elite. About 7,300 of the 19,729 rows
        have non-zero values and those look plausible; do not rank the
        top-of-league stars by these columns, but it is fine to use them for
        aggregate stats over the populated rows.
      - Two different quality columns, do not conflate them. kmeans_tier ranks
        the clusters within a position, 1 = best. category is a quantile split
        of those tiers into Elite / Good / Average / Poor, so several tiers can
        share one category. Rank by kmeans_tier; filter by category.

    Args:
        sql: A single SELECT (or WITH ... SELECT) statement. No writes, no PRAGMA,
            no ATTACH. Add your own LIMIT; results are truncated at 50 rows.
    """
    statement = sql.strip().rstrip(";")
    if not _READONLY.match(statement) or _FORBIDDEN.search(statement):
        return "ERROR: only a single read-only SELECT/WITH statement is allowed."
    if ";" in statement:
        return "ERROR: only one statement per call."

    try:
        # mode=ro makes the guard above a belt-and-braces check rather than the
        # only thing standing between a generated string and the data.
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        rows = con.execute(statement).fetchmany(MAX_ROWS + 1)
        con.close()
    except sqlite3.Error as exc:
        return f"SQL error: {exc}"

    if not rows:
        return "0 rows."

    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]
    header = " | ".join(rows[0].keys())
    body = "\n".join(
        " | ".join("" if v is None else str(v) for v in row) for row in rows
    )
    note = f"\n({MAX_ROWS} rows shown; more matched — add LIMIT or aggregate)" if truncated else ""
    return f"{header}\n{body}{note}"


def search_docs(query: str) -> str:
    """Search the project's written documentation and return the best passages.

    Use this for how and why questions rather than numeric ones: how k was
    chosen, why market value is excluded, what the preprocessing pipeline does,
    what a particular position's cluster looks like in plain terms.

    The corpus is summary.md (methodology, stability results, caveats),
    cluster_profiles.md (mean stats and example players for every cluster), and
    pipeline_internals.md (the scripts' own docstrings -- the only place the
    Elite/Good/Average/Poor rule and the market-value evidence are written out).

    Args:
        query: Search terms. Prefer the vocabulary the docs themselves use --
            e.g. "bootstrap ARI Jaccard", "RobustScaler PCA", "Striker cluster 3".
    """
    global _retriever
    if _retriever is None:
        _retriever = Retriever()

    hits = _retriever.search(query, top_k=4)
    if not hits:
        return "No matching passages."

    parts = []
    for score, chunk in hits:
        parts.append(
            f"--- {chunk.source} :: {chunk.heading} (score {score:.1f}) ---\n"
            f"{chunk.text[:MAX_DOC_CHARS]}"
        )
    return "\n\n".join(parts)


# OpenAI tool schema. The function names and docstrings are sent to the model
# verbatim, so the prose here is what the model uses to decide when to call
# what -- same role as Anthropic's @beta_tool played.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_players",
            "description": query_players.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": (
                            "A single SELECT (or WITH ... SELECT) statement. "
                            "No writes, no PRAGMA, no ATTACH. Add your own LIMIT; "
                            "results are truncated at 50 rows."
                        ),
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": search_docs.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search terms. Prefer the vocabulary the docs "
                            "themselves use -- e.g. 'bootstrap ARI Jaccard', "
                            "'RobustScaler PCA', 'Striker cluster 3'."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
]

# Map tool name -> callable, for dispatching tool_calls from the model.
TOOL_FUNCS = {
    "query_players": query_players,
    "search_docs": search_docs,
}


# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------

def player_columns() -> str:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cols = [r[1] for r in con.execute("PRAGMA table_info(players)")]
    con.close()
    return ", ".join(cols)


def system_prompt() -> str:
    return f"""You answer questions about a football player clustering project.

19,729 players were clustered into performance tiers separately within each of
8 position groups, using KMeans on a median-impute -> RobustScaler -> PCA(90%)
pipeline. k was chosen per position by bootstrap stability (ARI >= 0.90 and
every cluster's Jaccard >= 0.75), not by the elbow and not by silhouette.

You have two tools. Route by the kind of question:
- Numbers, rankings, counts, filters, specific players -> query_players.
- Method, rationale, caveats, what a cluster means -> search_docs.
- Many questions need both: get the number, then the interpretation.

Rules:
- Never state a statistic you did not get from a tool call in this conversation.
  If a query returns nothing, say so rather than filling the gap from memory.
- kmeans_cluster ids repeat across position groups and are not comparable
  across them. Always qualify a cluster by its position group. The same goes
  for kmeans_tier and category: both are assigned within a position, so a
  "Poor" striker and a "Poor" centre back were judged on different scales.
- latest_market_value and highest_market_value read 0 for star players
  (Messi, Haaland, etc.) even when their cluster tier is Elite -- an
  upstream join bug for the elite cohort. About 7,300 of 19,729 rows have
  plausible non-zero values, so the columns are usable in aggregate, but do
  not quote a specific star's value or rank players by these columns.
- Silhouette scores here are low (~0.2) because the feature space is one
  continuous cloud. That is expected and does not mean the clusters are wrong.
- Answer in prose, briefly. Give the number and what it means; skip the SQL
  unless asked.

players table columns: {player_columns()}"""


# --------------------------------------------------------------------------
# Chat loop
# --------------------------------------------------------------------------

def main() -> None:
    global VERBOSE

    parser = argparse.ArgumentParser(
        description="RAG chatbot over the player-clustering outputs."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print tool arguments and raw tool results as they happen.",
    )
    args = parser.parse_args()
    if args.verbose:
        VERBOSE = True

    if not DB.exists():
        sys.exit(f"{DB} not found. Run: python {HERE / 'build_db.py'}")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set.")

    client = OpenAI()
    messages: list[dict] = [{"role": "system", "content": system_prompt()}]

    if VERBOSE:
        print(f"[verbose] model={MODEL}, db={DB}")
    print("Player clustering assistant. Ctrl-C or 'quit' to exit.\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if user.lower() in {"quit", "exit"}:
            return
        if not user:
            continue

        messages.append({"role": "user", "content": user})

        # Manual tool loop. The model can chain tool calls -- e.g. search_docs
        # then query_players -- so we keep going until it returns a message
        # with no tool_calls.
        while True:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=16000,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            # No tool calls -> the model is done. Print its answer and break
            # back to the input prompt.
            if not msg.tool_calls:
                if msg.content:
                    print(f"\nbot> {msg.content}\n")
                    messages.append({"role": "assistant", "content": msg.content})
                break

            # Persist the assistant turn (with its tool_calls) before the tool
            # results, otherwise OpenAI rejects the next request.
            messages.append(msg)

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                print(f"  [{name}]")
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as exc:
                    result = f"ERROR: bad arguments from model: {exc}"
                else:
                    vprint(f"    args: {json.dumps(args, indent=2)}")
                    try:
                        result = TOOL_FUNCS[name](**args)
                    except Exception as exc:  # noqa: BLE001
                        result = f"ERROR: {exc}"
                vprint(f"    result: {result[:500]}{'...' if len(result) > 500 else ''}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )


if __name__ == "__main__":
    main()