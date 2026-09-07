"""Offline TF-IDF retrieval over normalized, processed evidence records."""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(text: str) -> str:
    """Normalize whitespace without changing evidence meaning."""
    return " ".join((text or "").split())


def load_processed_evidence(paths: Iterable[str | Path], max_documents: Optional[int] = None) -> List[Dict[str, str]]:
    """Load and de-duplicate evidence only from processed JSONL files."""
    corpus, seen = [], set()
    for path in paths:
        with Path(path).open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                evidence = normalize_text(record.get("evidence", ""))
                if not evidence or evidence in seen:
                    continue
                seen.add(evidence)
                evidence_id = record.get("sample_id") or hashlib.sha1(evidence.encode("utf-8")).hexdigest()[:16]
                corpus.append({"evidence_id": str(evidence_id), "evidence": evidence})
                if max_documents is not None and len(corpus) >= max_documents:
                    return corpus
    return corpus


class TfidfEvidenceRetriever:
    """Reusable in-memory lexical evidence retriever."""

    def __init__(self, corpus: List[Dict[str, str]]):
        if not corpus:
            raise ValueError("Evidence corpus must contain at least one record")
        self.corpus = [{"evidence_id": str(item["evidence_id"]), "evidence": normalize_text(item["evidence"])} for item in corpus]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([item["evidence"] for item in self.corpus])

    def retrieve(self, claim: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return up to ``top_k`` evidence records ranked by cosine similarity."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query = normalize_text(claim)
        if not query:
            return []
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        indices = sorted(range(len(scores)), key=lambda index: (-scores[index], index))[:top_k]
        return [{**self.corpus[index], "score": float(scores[index])} for index in indices]


class RobustTfidfEvidenceRetriever(TfidfEvidenceRetriever):
    """Versioned offline variant with English stop-word filtering for query noise."""
    def __init__(self, corpus: List[Dict[str, str]]):
        if not corpus:
            raise ValueError("Evidence corpus must contain at least one record")
        self.corpus = [{"evidence_id": str(item["evidence_id"]), "evidence": normalize_text(item["evidence"])} for item in corpus]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", strip_accents="unicode")
        self.matrix = self.vectorizer.fit_transform([item["evidence"] for item in self.corpus])
