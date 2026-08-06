"""Keyword (BM25) retrieval over the markdown docs in chatbot/docs/.

Why BM25 and not embeddings: the whole prose corpus here is a few hundred KB of
methodology text and cluster profiles, and the queries are dense with exact
tokens the corpus also uses -- position names, cluster ids, metric names like
`bootstrap_ari`. Lexical matching wins on that shape and needs no model call, no
index server, and no re-embedding when run_kmeans.py reruns.

Swap in vectors when the corpus grows past a few thousand chunks or starts
carrying paraphrase-heavy prose (scouting reports, match notes) where the user's
words won't match the document's words. The `search()` signature is the seam:
keep it, replace the body.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DOCS = Path(__file__).resolve().parent / "docs"

# BM25 parameters. k1 damps term-frequency saturation, b controls how hard long
# documents are penalised. These are the standard defaults.
K1 = 1.5
B = 0.75

# Extra credit for a term appearing in the chunk's heading. Without it, a short
# "Striker — cluster 3" profile loses to a long summary table that happens to
# repeat "striker" and "cluster" in every row -- exactly backwards for the
# lookup the heading was written to serve.
HEADING_WEIGHT = 2.0

_TOKEN = re.compile(r"[a-z0-9_]+")

# In a corpus this small and this technical, ordinary function words end up with
# deceptively high IDF -- "how" and "is" appear in few enough chunks to look
# discriminating, so "how is a player labelled Elite" scored a methodology
# section above the function that actually assigns the label. Dropping them from
# both the index and the query leaves the content words to decide.
# Note "k" is deliberately absent: it is a real term here (choosing k).
STOPWORDS = frozenset("""
a an the this that these those it its is are was were be been being
of to in on for with by at as from into about over under than then
and or but not no if so such each any all both
do does did doing done can could should would will shall may might must
i me my we our you your he she they them their there here
what which who whom whose when where why how
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Chunk:
    source: str
    heading: str
    text: str
    tokens: Counter
    length: int
    head_tokens: set[str]


def _split_markdown(path: Path) -> list[tuple[str, str]]:
    """Split a markdown file on headings into (heading, body) pairs.

    Headings carry most of the retrievable signal here ("### Striker — cluster
    3"), so each chunk keeps its own heading text prepended for matching.
    """
    # lines = path.read_text().splitlines()
    lines = path.read_text(encoding="utf-8").splitlines()
    sections: list[tuple[str, list[str]]] = [("(preamble)", [])]
    for line in lines:
        if line.startswith("#"):
            sections.append((line.lstrip("# ").strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections if "".join(body).strip()]


class Retriever:
    def __init__(self, docs_dir: Path = DOCS):
        self.chunks: list[Chunk] = []
        for path in sorted(docs_dir.glob("*.md")):
            for heading, body in _split_markdown(path):
                text = f"{heading}\n{body}"
                toks = tokenize(text)
                self.chunks.append(
                    Chunk(
                        path.name,
                        heading,
                        body,
                        Counter(toks),
                        len(toks),
                        set(tokenize(heading)),
                    )
                )

        self.avg_len = (
            sum(c.length for c in self.chunks) / len(self.chunks) if self.chunks else 0.0
        )
        # Document frequency per term, for the IDF component.
        self.df: Counter = Counter()
        for chunk in self.chunks:
            self.df.update(chunk.tokens.keys())
        self.n = len(self.chunks)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, Chunk]]:
        terms = tokenize(query)
        scored: list[tuple[float, Chunk]] = []
        for chunk in self.chunks:
            score = 0.0
            for term in terms:
                if term in chunk.head_tokens:
                    score += HEADING_WEIGHT * self._idf(term)
                tf = chunk.tokens.get(term, 0)
                if not tf:
                    continue
                norm = 1 - B + B * (chunk.length / self.avg_len) if self.avg_len else 1
                score += self._idf(term) * (tf * (K1 + 1)) / (tf + K1 * norm)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:top_k]


if __name__ == "__main__":
    import sys

    r = Retriever()
    print(f"{r.n} chunks indexed from {DOCS}")
    q = " ".join(sys.argv[1:]) or "how was k chosen"
    for score, chunk in r.search(q):
        print(f"\n[{score:.2f}] {chunk.source} :: {chunk.heading}")
        print(chunk.text[:300])
