from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.dashboard_data import page_data
from src.retrieval import RobustTfidfEvidenceRetriever, load_processed_evidence
from src.verillm_pipeline import analyze_response
from src.verification.nli_verifier import NLIVerifier, resolve_offline_model

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


class EvidenceInput(BaseModel):
    evidence_id: str = Field(..., min_length=1)
    evidence: str = Field(..., min_length=1)


class VerifyRequest(BaseModel):
    response: str = Field(..., min_length=1, description="LLM response to audit.")
    evidence: List[EvidenceInput] | None = Field(default=None, description="Optional evidence corpus.")
    top_k: int = Field(default=3, ge=1, le=10)
    min_retrieval_score: float = Field(default=0.50, ge=0, le=1)
    low_confidence_review_threshold: float = Field(default=0.50, ge=0, le=1)
    model_name: str | None = None
    application: str | None = None
    version: str | None = None


@lru_cache(maxsize=1)
def _resolve_processed_corpus() -> List[Dict[str, str]]:
    corpus_path = PROJECT_ROOT / "datasets" / "processed" / "AVeriTeC" / "averitec_normalized.jsonl"
    if not corpus_path.is_file():
        raise FileNotFoundError(
            "Processed AVeriTeC evidence is not available at "
            "datasets/processed/AVeriTeC/averitec_normalized.jsonl."
        )
    return load_processed_evidence([corpus_path])


@lru_cache(maxsize=1)
def _load_verifier() -> NLIVerifier:
    return NLIVerifier()


@lru_cache(maxsize=1)
def _runtime_components() -> tuple[RobustTfidfEvidenceRetriever, NLIVerifier]:
    corpus = _resolve_processed_corpus()
    return RobustTfidfEvidenceRetriever(corpus), _load_verifier()


def _run_analysis(
    response: str,
    *,
    evidence: List[Dict[str, str]] | None = None,
    top_k: int = 3,
    min_retrieval_score: float = 0.50,
    low_confidence_review_threshold: float = 0.50,
) -> Dict[str, Any]:
    if evidence is None:
        retriever, verifier = _runtime_components()
    else:
        retriever = RobustTfidfEvidenceRetriever(evidence)
        verifier = _load_verifier()
    result = analyze_response(
        response,
        evidence if evidence is not None else _resolve_processed_corpus(),
        top_k=top_k,
        min_retrieval_score=min_retrieval_score,
        retriever=retriever,
        verifier=verifier,
        review_config={"low_confidence": low_confidence_review_threshold},
    )
    result["mode"] = "nli_model"
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
    try:
        result = _run_analysis(sample_response)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (RuntimeError, OSError, ValueError, ImportError) as error:
        raise HTTPException(status_code=503, detail=f"Live verification is unavailable: {error}") from error
    return {
        "summary": result["summary"],
        "claims": result["claims"],
        "response": result["response"],
        "mode": result["mode"],
    }


@app.get("/api/page/{page}")
def dashboard_page(
    page: str,
    dataset: str | None = None,
    model: str | None = None,
    limit: int = Query(default=200, ge=1, le=500),
) -> Dict[str, Any]:
    try:
        return page_data(page, dataset=dataset, model=model, limit=limit)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail=f"Unable to load {page} page data: {error}") from error


@app.post("/api/verify")
def verify(payload: VerifyRequest) -> Dict[str, Any]:
    response = (payload.response or "").strip()
    if not response:
        raise HTTPException(status_code=400, detail="A response to analyze is required.")

    try:
        result = _run_analysis(
            response,
            evidence=[item.model_dump() for item in payload.evidence] if payload.evidence is not None else None,
            top_k=payload.top_k,
            min_retrieval_score=payload.min_retrieval_score,
            low_confidence_review_threshold=payload.low_confidence_review_threshold,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (RuntimeError, OSError, ValueError, ImportError) as error:
        raise HTTPException(status_code=503, detail=f"Live verification is unavailable: {error}") from error
    return {
        "summary": result["summary"],
        "claims": result["claims"],
        "response": result["response"],
        "mode": result["mode"],
        "metadata": {
            "model_name": payload.model_name,
            "application": payload.application,
            "version": payload.version,
            "top_k": payload.top_k,
            "min_retrieval_score": payload.min_retrieval_score,
        },
    }


@app.get("/api/health")
def detailed_health() -> Dict[str, Any]:
    corpus_path = PROJECT_ROOT / "datasets" / "processed" / "AVeriTeC" / "averitec_normalized.jsonl"
    nli_config_path = PROJECT_ROOT / "configs" / "verification_config.yaml"
    model_name = "unknown"
    if nli_config_path.is_file():
        import yaml

        with nli_config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        model_name = config.get("nli_model", {}).get("name", "unknown")
    model_path = Path(resolve_offline_model(model_name)) if model_name != "unknown" else None
    model_available = bool(model_path and model_path.is_dir())
    corpus_available = corpus_path.is_file()
    checks = {
        "backend": "ONLINE",
        "pipeline": "ONLINE",
        "retriever": "ONLINE" if corpus_available else "OFFLINE",
        "processed_evidence": {
            "status": "AVAILABLE" if corpus_available else "DATA NOT AVAILABLE",
            "path": str(corpus_path.relative_to(PROJECT_ROOT)),
        },
        "nli_model": {
            "status": "AVAILABLE" if model_available else "WARNING",
            "name": model_name,
            "local_cache_path": str(model_path) if model_available else None,
            "detail": None if model_available else "No local model snapshot found; local-only inference cannot start.",
        },
    }
    has_offline_dependencies = corpus_available and model_available
    return {
        "status": "ONLINE" if has_offline_dependencies else "WARNING",
        "checks": checks,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
