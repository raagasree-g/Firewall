from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import TfidfEvidenceRetriever, load_processed_evidence
from src.verillm_pipeline import analyze_response
from src.verification.nli_verifier import NLIVerifier

app = FastAPI(
    title="VeriLLM",
    description="Explainable RAG + ML framework for LLM reliability and hallucination detection.",
    version="0.1.0",
)
cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "VERILLM_CORS_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

FRONTEND_DIR = PROJECT_ROOT / "frontend"
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")


class VerifyRequest(BaseModel):
    response: str = Field(..., min_length=1, description="LLM response to audit.")
    evidence: List[Dict[str, str]] | None = Field(default=None, description="Optional evidence corpus.")


class OfflineFallbackVerifier:
    """Gracefully support UI use without the full NLI model cache."""

    @staticmethod
    def _score_pair(claim: str, evidence: str) -> tuple[str, float]:
        claim_text = (claim or "").strip().lower()
        evidence_text = (evidence or "").strip().lower()
        if not evidence_text:
            return "UNSUPPORTED", 0.12

        if claim_text in evidence_text:
            return "SUPPORTED", 0.9

        if any(token in evidence_text for token in ("not", "never", "cannot", "denied", "refuted", "false", "incorrect")):
            return "CONTRADICTED", 0.76

        if any(token in evidence_text for token in ("likely", "possibly", "probably", "may", "reportedly")):
            return "UNSUPPORTED", 0.46

        support_tokens = set(re.findall(r"\b[a-z]+\b", claim_text))
        evidence_tokens = set(re.findall(r"\b[a-z]+\b", evidence_text))
        overlap = len(support_tokens.intersection(evidence_tokens))
        if overlap:
            return "SUPPORTED", min(0.8, 0.55 + (overlap * 0.08))
        return "UNSUPPORTED", 0.32

    def predict_many(self, records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for record in records:
            claim = str(record.get("claim", "")).strip()
            evidence = str(record.get("evidence", "")).strip()
            verdict, confidence = self._score_pair(claim, evidence)
            outputs.append(
                {
                    "claim": claim,
                    "evidence": evidence,
                    "nli_label": verdict,
                    "verdict": verdict,
                    "confidence": float(confidence),
                }
            )
        return outputs


def _resolve_processed_corpus() -> List[Dict[str, str]]:
    candidates = [
        PROJECT_ROOT / "datasets" / "processed" / "AVeriTeC" / "averitec_normalized.jsonl",
        PROJECT_ROOT / "datasets" / "processed" / "FEVER" / "fever_normalized.jsonl",
        PROJECT_ROOT / "datasets" / "processed" / "HaluEval" / "halueval_normalized.jsonl",
        PROJECT_ROOT / "datasets" / "processed" / "RAGTruth" / "ragtruth_normalized.jsonl",
    ]
    files = [path for path in candidates if path.exists()]
    if files:
        try:
            return load_processed_evidence(files, max_documents=250)
        except Exception:
            pass
    return [
        {"evidence_id": "earth", "evidence": "Earth is the third planet from the Sun in our solar system."},
        {"evidence_id": "paris", "evidence": "Paris is the capital city of France."},
        {"evidence_id": "germany", "evidence": "Berlin is the capital city of Germany."},
        {"evidence_id": "mars", "evidence": "Mars is a cold desert planet and is known as the Red Planet."},
        {"evidence_id": "wall", "evidence": "The Great Wall of China is a series of fortifications built across northern China."},
        {"evidence_id": "eiffel", "evidence": "The Eiffel Tower is a landmark in Paris, France."},
        {"evidence_id": "company", "evidence": "OpenAI is an AI research and deployment company founded in 2015."},
    ]


def _run_analysis(response: str, corpus: List[Dict[str, str]]) -> Dict[str, Any]:
    try:
        verifier = NLIVerifier()
        mode = "nli_model"
    except Exception:
        verifier = OfflineFallbackVerifier()
        mode = "offline_fallback"

    result = analyze_response(response, corpus, top_k=2, verifier=verifier)
    result["mode"] = mode
    return result


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "project": "VeriLLM", "mode": "ready"}


@app.get("/")
def serve_frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/overview")
def overview() -> Dict[str, Any]:
    sample_response = (
        "The Eiffel Tower is in Paris. OpenAI was founded in 2015. "
        "The Great Wall of China is a series of fortifications across northern China."
    )
    result = _run_analysis(sample_response, _resolve_processed_corpus())
    return {
        "summary": result["summary"],
        "claims": result["claims"],
        "response": result["response"],
        "mode": result["mode"],
    }


@app.post("/api/verify")
def verify(payload: VerifyRequest) -> Dict[str, Any]:
    response = (payload.response or "").strip()
    if not response:
        raise HTTPException(status_code=400, detail="A response to analyze is required.")

    corpus = payload.evidence or _resolve_processed_corpus()
    if not corpus:
        raise HTTPException(status_code=400, detail="No evidence corpus is available for verification.")

    result = _run_analysis(response, corpus)
    return {
        "summary": result["summary"],
        "claims": result["claims"],
        "response": result["response"],
        "mode": result["mode"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
