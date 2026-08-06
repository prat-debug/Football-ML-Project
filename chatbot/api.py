"""FastAPI server that exposes the chatbot engine over HTTP and serves the
Tailwind UI from chatbot/static/.

    pip install fastapi uvicorn
    python chatbot/api.py            # serves on http://localhost:8000

Endpoints:
    GET  /            -> the chat UI (static/index.html)
    POST /api/chat    -> { session_id, message } -> { reply, steps }
    POST /api/reset   -> { session_id } -> clears that session's history
    GET  /api/health  -> { ok: true }
    GET  /api/outputs -> list CSVs in outputs/ with row counts
    GET  /api/outputs/{name}?limit&offset -> paged rows of a CSV
    GET  /api/outputs/{name}/download -> raw CSV file

Sessions are kept in-process in a dict keyed by session_id (the client picks
one, e.g. a UUID stored in localStorage). Switching to Redis or a cookie-based
store is a one-line change in _get_engine().
"""

from __future__ import annotations

import csv
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import ChatEngine

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "static"

# Load .env same as the CLI so OPENAI_API_KEY works without manual export.
load_dotenv(HERE / ".env")
load_dotenv(HERE.parent / ".env")


class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str


class ToolStep(BaseModel):
    tool: str
    args: dict
    result_preview: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    steps: list[ToolStep] = []


class ResetRequest(BaseModel):
    session_id: str


app = FastAPI(title="Player Clustering Chatbot")

# CORS is wide open because the UI is served by the same origin in production.
# If you split static hosting onto a different domain, restrict this to that
# origin -- the dev server at http://localhost:5173 etc.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Per-session engine registry. Engines are cheap to keep around: an OpenAI
# client plus a list of dictionaries. The conversation history can grow over
# a long session, so a real deployment would cap or persist this.
_engines: dict[str, ChatEngine] = {}


def _get_engine(session_id: str) -> ChatEngine:
    if session_id not in _engines:
        try:
            _engines[session_id] = ChatEngine()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _engines[session_id]


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    engine = _get_engine(req.session_id)
    result = engine.send(req.message)
    return ChatResponse(
        session_id=req.session_id,
        reply=result.text,
        steps=[
            ToolStep(tool=s.tool, args=s.args, result_preview=s.result_preview)
            for s in result.steps
        ],
    )


@app.post("/api/reset")
def reset(req: ResetRequest) -> dict:
    if req.session_id in _engines:
        _engines[req.session_id].reset()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Outputs CSV panel
# ---------------------------------------------------------------------------

# Friendly descriptions for each CSV. The UI shows these under the table so
# the user knows what the columns mean.
_OUTPUT_DESCRIPTIONS: dict[str, str] = {
    "kmeans_summary": (
        "One row per position group: chosen k, PCA dimensions, silhouette, "
        "and the smallest cluster size. The headline numbers."
    ),
    "kmeans_evaluation": (
        "Stability sweep at each k: bootstrap ARI, min cluster Jaccard, and "
        "market-value-tercile prediction accuracy (the model's external "
        "validity check, never seen by KMeans)."
    ),
    "kmeans_k_sweep": (
        "Internal quality metrics (silhouette, Davies-Bouldin, Calinski-Harabasz, "
        "inertia) for k in 2..10. Used to compare against the stability pick."
    ),
    "kmeans_cluster_stability": (
        "Per-cluster bootstrap Jaccard at the chosen k. Every cluster should "
        "be >= 0.75 for the run to be accepted."
    ),
    "kmeans_labels": (
        "Final assignment: every player with their cluster id and tier (1 = "
        "best within the position). 19,729 rows."
    ),
}

OUTPUTS_DIR = HERE.parent / "outputs"


def _list_outputs() -> list[dict]:
    """List every *.csv in outputs/ with row count, size, and description.

    Row count is computed from line count. Cheap for the 19k-row labels file
    and avoids loading the whole CSV into Python just to count rows.
    """
    if not OUTPUTS_DIR.exists():
        return []
    out: list[dict] = []
    for path in sorted(OUTPUTS_DIR.glob("*.csv")):
        with path.open() as f:
            row_count = max(0, sum(1 for _ in f) - 1)
        out.append({
            "name": path.stem,
            "filename": path.name,
            "rows": row_count,
            "size_bytes": path.stat().st_size,
            "description": _OUTPUT_DESCRIPTIONS.get(path.stem, ""),
        })
    return out


@app.get("/api/outputs")
def list_outputs() -> dict:
    return {"outputs": _list_outputs()}


@app.get("/api/outputs/{name}")
def get_output(
    name: str,
    limit: int = Query(200, ge=1, le=5000),
    offset: int = Query(0, ge=0),
) -> dict:
    """Return a chunk of a CSV: columns, rows, and total row count.

    The 19,729-row labels file is paged rather than sent whole -- the JSON
    payload would be roughly 3 MB and the table only shows ~50 rows at a
    time. The frontend re-fetches with a new offset when scrolling.
    """
    path = (OUTPUTS_DIR / f"{name}.csv").resolve()
    # Containment check: prevent ../ traversal from leaving outputs/.
    if OUTPUTS_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail=f"output {name} not found")

    with path.open() as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return {"name": name, "columns": [], "rows": [], "total": 0, "offset": 0}

        all_rows = list(reader)
        total = len(all_rows)
        chunk = all_rows[offset : offset + limit]

    return {
        "name": name,
        "description": _OUTPUT_DESCRIPTIONS.get(name, ""),
        "columns": header,
        "rows": chunk,
        "total": total,
        "offset": offset,
        "limit": limit,
    }


@app.get("/api/outputs/{name}/download")
def download_output(name: str) -> Response:
    """Raw CSV download -- used by the 'Download' button on the outputs panel."""
    path = (OUTPUTS_DIR / f"{name}.csv").resolve()
    if OUTPUTS_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail=f"output {name} not found")
    return Response(
        content=path.read_text(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
    )


# Static UI. Mounted last so /api/* keeps priority.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
