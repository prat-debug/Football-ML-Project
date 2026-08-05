# Player Clustering — Notebook

A KMeans clustering pipeline over 19,729 football players, plus a chat-style
notebook UI that lets you ask questions of the dataset in natural language.

The pipeline groups players into performance tiers **per position** (never
across all players — a goalkeeper and a striker share no useful feature
space), validates the clusters by bootstrap stability, and exposes the
results through two surfaces:

- A **CLI chatbot** (`chatbot/chat.py`) for terminal questions.
- A **Tailwind web UI** (`chatbot/static/index.html`) that serves the same
  engine over HTTP, with a second tab for browsing the raw `outputs/*.csv`
  tables.

This README covers everything: setup, the pipeline, the chat engine, the
web UI, the data, and the outputs.

---

## 1. Quick start

```bash
# One-time setup
cd /Users/lenditt/Documents/import-porject/P1
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt

# Put your OpenAI key in the chatbot's .env (the chatbot loads it from
# chatbot/.env first, then the repo root .env).
echo "OPENAI_API_KEY=sk-..." > chatbot/.env

# Build the chatbot's knowledge store (run once, and again after any of the
# run_*.py scripts re-write outputs/).
python chatbot/build_db.py

# Launch the web UI (chat + tables)
python chatbot/api.py
# → open http://127.0.0.1:8000

# Or use the CLI instead
python chatbot/chat.py
```

The web UI is the recommended surface. The CLI is the same engine without
the table browser.

---

## 2. What's in the repo

```
P1/
├── README.md                              ← this file
├── requirements.txt                       ← pinned dependencies
├── summary.md                             ← methodology writeup (read this)
│
├── clean_player_features.csv              ← raw input (19,729 player rows)
├── clean_player_features_categorised.csv  ← enriched input the runtime uses
│
├── common.py                              ← shared pipeline + FEATURES list
├── check_market_value.py                  ← diagnoses the broken market-value column
├── evaluate.py                            ← bootstrap stability + market-value test
├── run_kmeans.py                          ← picks k per position, writes labels
├── classify_players.py                    ← maps clusters → Elite / Good / Average / Poor
│
├── outputs/                               ← everything the pipeline writes
│   ├── kmeans_summary.csv                 ← 8 rows: chosen k per position
│   ├── kmeans_evaluation.csv              ← stability sweep at each k
│   ├── kmeans_k_sweep.csv                 ← silhouette/DB/CH/inertia for k=2..10
│   ├── kmeans_cluster_stability.csv       ← per-cluster bootstrap Jaccard
│   └── kmeans_labels.csv                  ← 19,729 final player → cluster assignments
│
└── chatbot/
    ├── chat.py                            ← CLI chat loop (thin, drives the engine)
    ├── engine.py                          ← reusable engine: tools, prompt, loop
    ├── api.py                             ← FastAPI server + static UI mount
    ├── build_db.py                        ← builds chatbot/players.db from outputs/
    ├── retriever.py                       ← BM25 over chatbot/docs/*.md
    ├── players.db                         ← SQLite: the chatbot's numeric store
    ├── .env                               ← OPENAI_API_KEY (gitignored in real life)
    ├── docs/
    │   ├── summary.md                     ← (overlap) what the chatbot reads
    │   ├── cluster_profiles.md            ← per-cluster stats for every (position, cluster)
    │   └── pipeline_internals.md          ← docstrings of the pipeline scripts
    └── static/
        ├── index.html                     ← UI shell (Notebook + Tables tabs)
        ├── app.js                         ← client logic
        └── styles.css                     ← paper theme + table styles
```

---

## 3. The data

19,729 player rows in `clean_player_features_categorised.csv`. Each
row carries per-90 attacking output, raw minutes/goals/assists, age,
height, and — **broken** — market value.

**Features used for clustering** (defined in [`common.py`](common.py)):

```
goals_per_90, assists_per_90, goal_contributions_per_90,
estimated_xg, estimated_xa, minutes_per_match, matches_played,
age, height_in_cm
```

**Explicitly excluded**:
- `*_score` columns — they are a weighted formula built from the same raw
  columns. Including them would just carve the formula into slices.
- `*_market_value` — see [§4.4](#44-broken-column-latest_market_value).

**Pipeline** (per position group):

```
median-impute  →  RobustScaler  →  PCA(90% variance)  →  KMeans
```

---

## 4. The pipeline

### 4.1 Scripts

| Script | Purpose |
|---|---|
| [`common.py`](common.py) | CSV loader, `FEATURES` list, the median-impute → RobustScaler → PCA → KMeans pipeline factory, per-cluster profiling. |
| [`check_market_value.py`](check_market_value.py) | Diagnoses the broken upstream join (see below). |
| [`evaluate.py`](evaluate.py) | Measures **stability** (bootstrap ARI + per-cluster Jaccard) and **external validity** (market-value tercile prediction, market value never seen by the model). |
| [`run_kmeans.py`](run_kmeans.py) | Picks k per position via the stability table, then writes the final labels and cluster profiles. |
| [`classify_players.py`](classify_players.py) | Maps clusters to Elite / Good / Average / Poor via the quantile split documented in `chatbot/docs/pipeline_internals.md`. |

### 4.2 How k is chosen

**Not by the elbow**, **not by silhouette** (it picks k=2 for every
position because it structurally rewards coarse splits).

k is chosen by **bootstrap stability** from `evaluate.py`:

```
bootstrap ARI  >= 0.90
  AND
every cluster has Jaccard >= 0.75
```

Take the **finest** k that still survives resampling. **Per position**,
because positions disagree — Full Back collapses at k=4 (ARI 0.58) but is
solid at k=6, while Centre Back is best at k=3.

### 4.3 Output files

All written to `outputs/`:

| File | Rows | What it is |
|---|---|---|
| `kmeans_summary.csv` | 8 | One row per position: chosen k, PCA dims, silhouette, smallest cluster. |
| `kmeans_evaluation.csv` | 56 | Stability sweep at each k: ARI, Jaccard, market-value lift. |
| `kmeans_k_sweep.csv` | 72 | Silhouette / Davies-Bouldin / Calinski-Harabasz / inertia for k=2..10. |
| `kmeans_cluster_stability.csv` | 280 | Per-cluster bootstrap Jaccard at the chosen k. |
| `kmeans_labels.csv` | 19,729 | Final player → (cluster, tier) assignment. |

### 4.4 Broken column: `latest_market_value`

The market-value column is internally consistent (0.85 Pearson with
`highest_market_value`) but **orthogonal to performance** (≈0.01 Pearson
with every perf metric). Messi, Mbappé, Haaland and Bellingham are all
recorded at **0**, while the dataset max (140 M) belongs to a centre back
with 26 matches and 1 goal.

**Verdict**: the value block is joined to the wrong player rows. It is
therefore excluded from clustering and **not trusted** as a validation
signal. See [`check_market_value.py`](check_market_value.py) for the
evidence and [`summary.md`](summary.md) for the full diagnosis.

---

## 5. The chatbot

### 5.1 The engine

[`chatbot/engine.py`](chatbot/engine.py) owns the OpenAI client, the
system prompt, and the tool-calling loop. Two tools are exposed:

| Tool | What it does |
|---|---|
| `query_players(sql)` | Runs a single read-only `SELECT` against `chatbot/players.db`. Read-only + statement allowlist guards. |
| `search_docs(query)` | BM25 over `chatbot/docs/*.md`. Returns the top passages. |

The system prompt — generated at runtime — includes the live `players`
column list so the model always sees the actual schema.

The model routes by the kind of question:

- **Numbers, rankings, counts, filters, specific players** → `query_players`.
- **Method, rationale, caveats, what a cluster means** → `search_docs`.
- Many questions need both: get the number, then the interpretation.

### 5.2 The CLI

```bash
python chatbot/chat.py             # quiet
python chatbot/chat.py -v          # verbose: prints tool calls + raw results
VERBOSE=1 python chatbot/chat.py   # verbose via env
```

`Ctrl-C` or `quit` to exit.

### 5.3 The web UI

```bash
python chatbot/api.py
# → http://127.0.0.1:8000
```

Two tabs:

#### **Notebook** (default)

A chat surface styled like a notebook. The sidebar shows the session id,
question count, and the routing rule. The welcome card has four numbered
example prompts (`01 · Numeric`, `02 · Method`, `03 · Both`, `04 · Aggregate`).

**Behaviour**:

- **Enter** sends. **Shift+Enter** inserts a newline.
- Question count persists in `localStorage`.
- Session id persists in `localStorage`; the same id is sent to the server
  with every chat call so the server-side engine keeps the conversation.
- "Clear notes" button resets the session and reloads the page.
- Tool calls are rendered as a collapsible `+ N tool calls` block under the
  assistant message, showing the SQL or query the model ran and the result
  preview.

#### **Tables**

A browser for the `outputs/*.csv` files. The left rail lists each file
with row count and size. The right pane shows a paginated, sortable,
filterable table with a download button.

- **Sort** any column by clicking its header (↑ / ↓).
- **Filter** is debounced client-side over the current page.
- **Pager** walks 200 rows at a time; the 19,729-row labels file paged
  end-to-end takes 99 clicks (or paste a `?limit=...&offset=...` URL).
- **Download** fetches the raw CSV via the same endpoint the CLI uses.

---

## 6. The web API

The FastAPI server in [`chatbot/api.py`](chatbot/api.py) exposes:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | The UI (static `index.html`). |
| `GET` | `/api/health` | `{ ok: true }` liveness check. |
| `POST` | `/api/chat` | `{ session_id, message }` → `{ reply, steps }`. |
| `POST` | `/api/reset` | `{ session_id }` → clears that session's history. |
| `GET` | `/api/outputs` | List CSVs in `outputs/` with row counts and descriptions. |
| `GET` | `/api/outputs/{name}` | Paged CSV rows: `?limit=200&offset=0`. |
| `GET` | `/api/outputs/{name}/download` | Raw CSV as `text/csv` attachment. |

The `/api/outputs/{name}` endpoint only resolves files inside `outputs/`
(path-traversal guarded).

### Example chat call

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","message":"Top 10 strikers by xG per 90"}'
```

The response carries the assistant's prose plus a `steps` array showing
the SQL the model ran and a preview of the rows.

---

## 7. Rebuilding the chatbot's knowledge store

The chatbot reads `outputs/*.csv` and the script docstrings into a SQLite
database at `chatbot/players.db`. **Re-run this every time the pipeline
re-writes the outputs**:

```bash
python chatbot/build_db.py
```

What it does:
1. Loads `clean_player_features_categorised.csv` and `outputs/kmeans_labels.csv`.
2. Builds `chatbot/players.db` with the `players` table (one row per player)
   plus the four side tables (`kmeans_summary`, `kmeans_evaluation`,
   `kmeans_k_sweep`, `kmeans_cluster_stability`).
3. Walks the named source files (`common.py`, `check_market_value.py`,
   `evaluate.py`, `run_kmeans.py`, `classify_players.py`) and writes their
   docstrings into `chatbot/docs/pipeline_internals.md`.
4. Writes one `(position, cluster)` profile per row into
   `chatbot/docs/cluster_profiles.md`.

If the database or docs are out of date, the model will quote stale
numbers. There is no automatic refresh — by design, so the data the model
sees is reproducible.

---

## 8. The docs the chatbot reads

`chatbot/docs/`:

- `summary.md` — copy of the project-level summary, with a small note on
  the chat engine.
- `cluster_profiles.md` — generated by `build_db.py`. One section per
  `(position_group, kmeans_cluster)` pair with the centroid stats and the
  example players. The model uses this to answer "what does a Centre Back
  in cluster 2 look like?"-style questions.
- `pipeline_internals.md` — the **docstrings** of the pipeline scripts.
  This is the only place the Elite / Good / Average / Poor quantile rule
  is written out, so the model reads it to justify the category labels.

All three are retrieved by BM25 (`chatbot/retriever.py`). Lexical matching
wins here because the queries are dense with exact tokens the docs also
use (`bootstrap_ari`, `RobustScaler`, "Striker cluster 3"…).

---

## 9. Dependencies

[`requirements.txt`](requirements.txt):

```
pandas==3.0.5
numpy==2.5.1
scikit-learn==1.9.0
openai>=1.0
python-dotenv>=1.0
fastapi>=0.110
uvicorn>=0.27
pydantic>=2.5
```

Tested on Python 3.14.4. The OpenAI client is `openai>=1.0` (the new
`OpenAI()` constructor); the model in use is `gpt-4o-mini`.

---

## 10. Troubleshooting

**`OPENAI_API_KEY is not set.`**
Put `OPENAI_API_KEY=sk-...` in `chatbot/.env` (preferred) or `chatbot/../.env`.

**`chatbot/players.db not found.`**
Run `python chatbot/build_db.py` once.

**`ModuleNotFoundError: fastapi` / `uvicorn`.**
`pip install -r requirements.txt` — they're listed but not in the venv by
default.

**Stale data in the chatbot.**
The chatbot reads from `chatbot/players.db`, not from `outputs/` directly.
After re-running the pipeline, re-run `python chatbot/build_db.py`.

**Chat returns 503.**
The server couldn't construct a `ChatEngine` — usually the missing
`OPENAI_API_KEY` or the missing `players.db`. The HTTP response body has
the reason.

**UI loads but the file browser is empty.**
The `outputs/` folder is missing or empty. Run the pipeline first.

**Want to restart the conversation?**
Click "Clear notes" in the header, or `Ctrl-C` + restart the CLI.

---

## 11. Read next

- [`summary.md`](summary.md) — the methodology writeup. The full argument
  for stability-over-silhouette, why RobustScaler not StandardScaler, the
  market-value diagnosis, and the per-position k picks.
- [`chatbot/docs/cluster_profiles.md`](chatbot/docs/cluster_profiles.md)
  — the cluster centroids. The single best file for "what did the model
  actually decide".
- [`chatbot/docs/pipeline_internals.md`](chatbot/docs/pipeline_internals.md)
  — the docstrings of the pipeline scripts. The only place the
  Elite / Good / Average / Poor rule is written out.
