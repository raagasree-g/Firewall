"""Sentence-based claim extraction.

This is intentionally a conservative starting point, not an atomic-claim
decomposer.  A future extractor can implement the same ``extract`` method.
"""
import re
from typing import Any, Dict, List


class SentenceClaimExtractor:
    """Extract non-duplicate declarative sentence claims from a response."""

    _sentence_boundary = re.compile(r"(?<=[.!?])\s+|\n+")
    _word = re.compile(r"\w+", re.UNICODE)

    @classmethod
    def _canonical(cls, sentence: str) -> str:
        return " ".join(cls._word.findall(sentence.lower()))

    def extract(self, response: str) -> List[Dict[str, Any]]:
        """Return structured sentence claims in their original order."""
        if not response or not response.strip():
            return []
        claims, seen = [], set()
        for position, raw_sentence in enumerate(self._sentence_boundary.split(response.strip())):
            sentence = " ".join(raw_sentence.split())
            canonical = self._canonical(sentence)
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            claims.append({
                "claim_id": f"claim_{len(claims) + 1:03d}",
                "claim": sentence,
                "source_sentence": sentence,
                "position": position,
            })
        return claims
