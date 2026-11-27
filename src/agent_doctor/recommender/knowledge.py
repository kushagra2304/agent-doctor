"""The fix playbooks and a small BM25 index over them.

Each playbook is a markdown file with front matter (``id``, ``title``, ``problems``,
optional ``providers`` and ``effort``). Add your own with ``KnowledgeBase.load(extra_dir)``:
your playbooks are searched alongside the built-in ones, and win on a duplicate ``id``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.S)
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be by for from has have in is it its of on or so that the "
                  "their them then this to too was were when which with you your".split())


def tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower().replace("_", " ")) if w not in _STOP]


class Playbook(BaseModel):
    id: str
    title: str
    problems: list[str] = Field(description="Finding kinds this playbook fixes.")
    providers: list[str] = Field(default_factory=list, description="Empty = any provider.")
    effort: str = "medium"
    body: str
    source: str

    def section(self, name: str) -> str:
        """The text under ``## name``, or "" if the playbook has no such section."""
        m = re.search(rf"^## {re.escape(name)}\s*\n(.*?)(?=^## |\Z)", self.body, re.S | re.M)
        return m.group(1).strip() if m else ""


def parse_playbook(text: str, source: str) -> Playbook:
    m = _FRONT.match(text.replace("\r\n", "\n"))
    if not m:
        raise ValueError(f"{source}: a playbook must start with --- front matter ---")
    meta = yaml.safe_load(m.group(1)) or {}
    return Playbook(body=m.group(2).strip(), source=source, **meta)


class KnowledgeBase:
    """Playbooks plus a BM25 index over their text."""

    def __init__(self, playbooks: list[Playbook], k1: float = 1.5, b: float = 0.75):
        self.playbooks = playbooks
        self.k1, self.b = k1, b
        self._docs = [Counter(tokens(f"{p.title} {p.title} {' '.join(p.problems)} {p.body}"))
                      for p in playbooks]
        self._len = [sum(d.values()) for d in self._docs]
        self._avg = sum(self._len) / len(self._len) if self._len else 0.0
        df: Counter[str] = Counter()
        for d in self._docs:
            df.update(d.keys())
        n = len(playbooks)
        self._idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    @classmethod
    def load(cls, *extra_dirs: str | Path) -> KnowledgeBase:
        books: dict[str, Playbook] = {}
        for f in sorted(resources.files("agent_doctor.recommender").joinpath("playbooks").iterdir(),
                        key=lambda f: f.name):
            if f.name.endswith(".md"):
                p = parse_playbook(f.read_text(encoding="utf-8"), f"built-in:{f.name}")
                books[p.id] = p
        for d in extra_dirs:
            for f in sorted(Path(d).glob("*.md")):
                p = parse_playbook(f.read_text(encoding="utf-8"), str(f))
                books[p.id] = p
        return cls(list(books.values()))

    def get(self, playbook_id: str) -> Playbook | None:
        return next((p for p in self.playbooks if p.id == playbook_id), None)

    def bm25(self, query: str) -> list[float]:
        q = tokens(query)
        scores = []
        for doc, length in zip(self._docs, self._len, strict=True):
            s = 0.0
            for w in q:
                f = doc.get(w, 0)
                if f:
                    s += self._idf[w] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * length / self._avg))
            scores.append(s)
        return scores

    def search(self, query: str, limit: int = 5) -> list[tuple[Playbook, float]]:
        scored = sorted(zip(self.playbooks, self.bm25(query), strict=True), key=lambda x: -x[1])
        return [(p, round(s, 4)) for p, s in scored[:limit] if s > 0]


@lru_cache(maxsize=1)
def default_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase.load()
