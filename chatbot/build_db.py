"""Build the chatbot's two knowledge stores from the clustering outputs.

Run this once (and again whenever run_kmeans.py / classify_players.py rerun):

    python chatbot/build_db.py

Produces:
  chatbot/players.db          -- SQLite. The *numeric* store: 19,729 players +
                                 the evaluation tables. Queried with real SQL,
                                 because "top 10 strikers by xG" is an ORDER BY,
                                 not a similarity search.
  chatbot/docs/*.md           -- The *prose* store, chunked and keyword-searched
                                 at query time. summary.md, a generated profile
                                 for every (position, cluster) pair (a bare id
                                 like "Winger 2" means nothing without its
                                 centroid stats), and the pipeline docstrings,
                                 which hold rules summary.md never states -- the
                                 quantile split behind Elite/Good/Average/Poor
                                 among them.
"""

import ast
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
DB = HERE / "players.db"
DOCS = HERE / "docs"

PLAYERS_CSV = ROOT / "clean_player_features_categorised.csv"
OUTPUTS = ROOT / "outputs"
LABELS_CSV = OUTPUTS / "kmeans_labels.csv"

# The pipeline's reasoning lives in these docstrings and nowhere else --
# summary.md documents the clustering but never states, for instance, how a
# player ends up labelled "Elite".
SOURCE_FILES = [
    "common.py",
    "check_market_value.py",
    "evaluate.py",
    "run_kmeans.py",
    "classify_players.py",
]

# Small analysis tables that answer "how was k chosen" style questions.

SIDE_TABLES = {
    "kmeans_summary": OUTPUTS / "kmeans_summary.csv",
    "kmeans_evaluation": OUTPUTS / "kmeans_evaluation.csv",
    "kmeans_cluster_stability": OUTPUTS / "kmeans_cluster_stability.csv",
    "kmeans_k_sweep": OUTPUTS / "kmeans_k_sweep.csv",
}

# Columns worth indexing -- every question filters on at least one of these.
INDEXED = ["player_name", "position_group", "category", "kmeans_cluster", "country"]

PROFILE_COLS = [
    "goals_per_90",
    "assists_per_90",
    "goal_contributions_per_90",
    "estimated_xg",
    "estimated_xa",
    "minutes_per_match",
    "matches_played",
    "age",
    "height_in_cm",
    "talent_score",
]


def _demojibake_once(s: str) -> str:
    """Undo one round of "UTF-8 bytes decoded as cp1252/latin-1"."""
    for encoding in ("cp1252", "latin-1"):
        try:
            return s.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return s


def demojibake(s):
    """Repair mangled accents, e.g. 'Kylian MbappÃ©' -> 'Kylian Mbappé'.

    18% of player names arrive mangled from an upstream encoding bug. Left
    alone, a user asking about Mbappé or Ibrahimović gets zero rows -- the
    chatbot would look broken while faithfully reporting broken data. Repaired
    here rather than in the CSV: the CSV is the pipeline's output, and fixing
    the real join is upstream's job.

    Applied repeatedly because some values were mangled twice over.
    """
    if not isinstance(s, str):
        return s
    for _ in range(3):
        repaired = _demojibake_once(s)
        if repaired == s:
            break
        s = repaired
    return s


def build_sqlite() -> pd.DataFrame:
    df = pd.read_csv(PLAYERS_CSV)

    for col in ("player_name", "first_name", "last_name", "country", "current_club_name"):
        if col in df.columns:
            df[col] = df[col].map(demojibake)

    # kmeans_tier exists only in kmeans_labels.csv -- the categorised CSV carries
    # kmeans_cluster and category but not the tier ranking (1 = best cluster in
    # that position). It is not redundant with category: Goalkeeper tiers 7-8 are
    # both labelled Elite, so a tier question cannot be answered from category.
    if LABELS_CSV.exists():
        tiers = pd.read_csv(LABELS_CSV, usecols=["player_key", "kmeans_tier"])
        before = len(df)
        df = df.merge(tiers, on="player_key", how="left")
        missing = int(df["kmeans_tier"].isna().sum())
        assert len(df) == before, "kmeans_labels.csv join duplicated player rows"
        if missing:
            print(f"  ! {missing:,} players have no kmeans_tier")
    else:
        print(f"  ! {LABELS_CSV.name} not found -- kmeans_tier unavailable")

    if DB.exists():
        DB.unlink()

    with sqlite3.connect(DB) as con:
        df.to_sql("players", con, index=False)
        for name, path in SIDE_TABLES.items():
            if path.exists():
                pd.read_csv(path).to_sql(name, con, index=False)
            else:
                print(f"  ! skipped {name} ({path.name} not found)")
        for col in INDEXED:
            if col in df.columns:
                con.execute(f'CREATE INDEX "idx_{col}" ON players("{col}")')
        con.commit()

    print(f"  players.db  <- {len(df):,} players, {len(df.columns)} columns")
    return df


def write_cluster_profiles(df: pd.DataFrame) -> None:
    """One markdown section per (position_group, kmeans_cluster).

    This is the piece that makes cluster ids answerable in natural language:
    the model can retrieve "Striker cluster 3 averages 0.61 goals/90, n=448,
    typical members X, Y, Z" instead of guessing what cluster 3 is.
    """
    lines = [
        "# Cluster profiles (per position group)",
        "",
        "Each row is one KMeans cluster. Means are over the players in that",
        "cluster. `category` is the modal quality label assigned by",
        "classify_players.py. Example players are the most-capped members.",
        "",
    ]

    for pos, sub in df.groupby("position_group", sort=True):
        lines.append(f"## {pos} (n={len(sub):,})")
        lines.append("")
        for cid, grp in sub.groupby("kmeans_cluster", sort=True):
            means = grp[[c for c in PROFILE_COLS if c in grp.columns]].mean().round(3)
            examples = ", ".join(
                grp.nlargest(3, "matches_played")["player_name"].astype(str)
            )
            modal = grp["category"].mode()
            category = modal.iloc[0] if len(modal) else "n/a"
            lines.append(f"### {pos} — cluster {cid}")
            lines.append("")
            lines.append(f"- players: {len(grp):,}")
            lines.append(f"- dominant category: {category}")
            if "kmeans_tier" in grp.columns:
                tier = grp["kmeans_tier"].mode()
                if len(tier):
                    lines.append(f"- tier (1 = best in position): {int(tier.iloc[0])}")
            for k, v in means.items():
                lines.append(f"- mean {k}: {v}")
            lines.append(f"- example players: {examples}")
            lines.append("")

    path = DOCS / "cluster_profiles.md"
    # path.write_text("\n".join(lines))
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  docs/cluster_profiles.md  <- {df['position_group'].nunique()} position groups")


def write_pipeline_docs() -> None:
    """Lift module and function docstrings out of the pipeline into the corpus.

    Docstrings are the only place several answerable rules are written down --
    the Elite/Good/Average/Poor quantile split, why RobustScaler over
    StandardScaler, what the market-value evidence actually was. Extracting them
    keeps one source of truth: edit the script, rerun this, the bot updates.
    """
    lines = [
        "# Pipeline internals (extracted from source docstrings)",
        "",
        "How each script works, in its authors' own words. Regenerated from the",
        "`.py` files, so this never drifts from the code.",
        "",
    ]

    found = 0
    for name in SOURCE_FILES:
        path = ROOT / name
        if not path.exists():
            print(f"  ! {name} not found")
            continue
        tree = ast.parse(path.read_text())
        lines.append(f"## {name}")
        lines.append("")
        module_doc = ast.get_docstring(tree)
        if module_doc:
            lines.extend([module_doc.strip(), ""])
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node)
            if not doc:
                continue
            lines.append(f"### {name} — {node.name}()")
            lines.append("")
            lines.extend([doc.strip(), ""])
        found += 1

    # (DOCS / "pipeline_internals.md").write_text("\n".join(lines))
    (DOCS / "pipeline_internals.md").write_text("\n".join(lines),encoding="utf-8")
    print(f"  docs/pipeline_internals.md  <- {found} source files")


def copy_summary() -> None:
    src = ROOT / "summary.md"
    if src.exists():
        # (DOCS / "summary.md").write_text(src.read_text())
        (DOCS / "summary.md").write_text(src.read_text(encoding="utf-8"),encoding="utf-8")
        print("  docs/summary.md  <- copied from project root")
    else:
        print("  ! summary.md not found at project root")


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    print("Building chatbot stores...")
    frame = build_sqlite()
    write_cluster_profiles(frame)
    write_pipeline_docs()
    copy_summary()
    print("Done.")
