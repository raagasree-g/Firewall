"""Run an offline VeriLLM core-pipeline demonstration."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import TfidfEvidenceRetriever, load_processed_evidence
from src.verillm_pipeline import analyze_response
from src.verification.nli_verifier import NLIVerifier


def main() -> None:
    corpus = load_processed_evidence([PROJECT_ROOT / "datasets" / "processed" / "AVeriTeC" / "averitec_normalized.jsonl"])
    retriever = TfidfEvidenceRetriever(corpus)
    verifier = NLIVerifier()
    # These AVeriTeC-dev candidates exercise all three previously observed NLI
    # outcome categories. Their verdicts are not hardcoded: NLI decides them.
    responses = [
        "In a letter to Steve Jobs, Sean Connery refused to appear in an apple commercial.",
        "Trump Administration claimed songwriter Billie Eilish Is Destroying Our Country In Leaked Documents.",
        "Due to Imran Khan's criticism of Macron's comments on Islam, French authorities cancelled the visas of 183 Pakistani citizens and deported 118 from the country.",
    ]
    for response in responses:
        print(json.dumps(analyze_response(response, corpus, top_k=1, retriever=retriever, verifier=verifier), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
