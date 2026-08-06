"""Reusable chat engine for the player-clustering chatbot.

This module wraps the tool-calling loop from chat.py so it can be driven by
both the CLI (chat.py) and the HTTP API (api.py). The CLI used to own the loop;
hoisting it here keeps a single source of truth for system prompt, tool routing,
and the read-only SQL guard.

Public surface:
    ChatEngine()      -- one per process; holds the OpenAI client and the
                         conversation history sent to the model.
    .send(user_msg)   -- returns the assistant's final text plus the
                         tool-calling trace that produced it (for the UI
                         sidebar / verbose mode).
    .reset()          -- clears the history, useful for the UI's "new chat".
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from chatbot.retriever import Retriever

HERE = Path(__file__).resolve().parent
DB = HERE / "players.db"

# Load OPENAI_API_KEY from chatbot/.env first, then repo root .env. Shell
# exports win over the file (override=False), matching the CLI behaviour.
load_dotenv(HERE / ".env")
load_dotenv(HERE.parent / ".env")

MODEL = "gpt-4o-mini"
MAX_ROWS = 50
MAX_DOC_CHARS = 2000

# Read-only SQL guard. Same regex set as the CLI: structural check, not a
# prompt instruction (a model can be steered by anything it reads in data).
_READONLY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(attach|pragma|insert|update|delete|drop|alter|create|replace|vacuum)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

def query_players(sql: str) -> str:
    """Run a read-only SQL query against the player database and return the rows."""
    statement = sql.strip().rstrip(";")
    if not _READONLY.match(statement) or _FORBIDDEN.search(statement):
        return "ERROR: only a single read-only SELECT/WITH statement is allowed."
    if ";" in statement:
        return "ERROR: only one statement per call."

    try:
        # mode=ro is the belt-and-braces check; the regex above is the brace.
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
    """Search the project's written documentation and return the best passages."""
    hits = _RETRIEVER.search(query, top_k=4)
    if not hits:
        return "No matching passages."

    parts = []
    for score, chunk in hits:
        parts.append(
            f"--- {chunk.source} :: {chunk.heading} (score {score:.1f}) ---\n"
            f"{chunk.text[:MAX_DOC_CHARS]}"
        )
    return "\n\n".join(parts)


# One retriever per process. The body is small (a few hundred KB of markdown)
# and rebuild is cheap, but tkinter-style per-request construction is wasteful.
_RETRIEVER = Retriever()
TOOL_FUNCS = {
    "query_players": query_players,
    "search_docs": search_docs,
}


# --------------------------------------------------------------------------
# OpenAI tool schema
# --------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
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
                            "themselves use — e.g. 'bootstrap ARI Jaccard', "
                            "'RobustScaler PCA', 'Striker cluster 3'."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
]


def _player_columns() -> str:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cols = [r[1] for r in con.execute("PRAGMA table_info(players)")]
    con.close()
    return ", ".join(cols)


def system_prompt() -> str:
    """Build the system prompt. Pulls the live column list from players.db so the
    model sees the actual schema, not a hand-maintained copy that can drift."""
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
  (Messi, Haaland, etc.) even when their cluster tier is Elite — an
  upstream join bug for the elite cohort. About 7,300 of 19,729 rows have
  plausible non-zero values, so the columns are usable in aggregate, but do
  not quote a specific star's value or rank players by these columns.
- Silhouette scores here are low (~0.2) because the feature space is one
  continuous cloud. That is expected and does not mean the clusters are wrong.
- Answer in prose, briefly. Give the number and what it means; skip the SQL
  unless asked.

players table columns: {_player_columns()}"""


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

@dataclass
class TraceStep:
    """One tool call the model made on the way to the answer, for the UI."""
    tool: str
    args: dict[str, Any]
    result_preview: str


@dataclass
class EngineReply:
    text: str
    steps: list[TraceStep] = field(default_factory=list)


class ChatEngine:
    """Stateful chat engine. One instance per HTTP session, keyed by session_id."""

    def __init__(self, model: str = MODEL, max_tokens: int = 16000):
        if not os.environ.get("OPENAI_API_KEY"):
            print("OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))
            raise RuntimeError("OPENAI_API_KEY is not set.")
        if not DB.exists():
            raise RuntimeError(f"{DB} not found. Run: python chatbot/build_db.py")
        self.client = OpenAI()
        self.model = model
        self.max_tokens = max_tokens
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt()}
        ]

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": system_prompt()}]

    def send(self, user_msg: str) -> EngineReply:
        """Send a user message, run the tool-calling loop until the model returns
        a final answer, and return the answer plus the trace of tools it used."""
        self.messages.append({"role": "user", "content": user_msg})
        steps: list[TraceStep] = []

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                text = msg.content or ""
                if text:
                    self.messages.append({"role": "assistant", "content": text})
                return EngineReply(text=text, steps=steps)

            # Persist the assistant turn (with its tool_calls) before the tool
            # results, otherwise OpenAI rejects the next request.
            self.messages.append(msg)

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as exc:
                    result = f"ERROR: bad arguments from model: {exc}"
                    args = {}
                else:
                    try:
                        result = TOOL_FUNCS[name](**args)
                    except Exception as exc:  # noqa: BLE001
                        result = f"ERROR: {exc}"

                preview = result if len(result) <= 500 else result[:500] + "..."
                steps.append(TraceStep(tool=name, args=args, result_preview=preview))

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
