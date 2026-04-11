# qwen_reranker_api.py
import os
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

# --------------------------------------------------
# Config
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.environ.get(
    "RERANKER_MODEL_PATH",
    os.path.join(BASE_DIR, "models", "qwen3-reranker-0.6b"),
)

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI(title="Qwen External Reranker API")

reranker = CrossEncoder(
    DEFAULT_MODEL,
    trust_remote_code=True,
)

# --------------------------------------------------
# Schema
# --------------------------------------------------
class RerankRequest(BaseModel):
    context: str = ""
    candidates: List[str] = Field(default_factory=list)
    top_k: Optional[int] = None
    return_documents: bool = True


class ScoredCandidate(BaseModel):
    text: str
    score: float
    rank: int


class RerankResponse(BaseModel):
    model: str
    context: str
    scored_candidates: List[ScoredCandidate]
    ranked_candidates: List[str]


def build_pairs(context: str, candidates: List[str]) -> List[List[str]]:
    query = context or ""
    return [[query, candidate] for candidate in candidates]


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": DEFAULT_MODEL,
    }


@app.get("/models")
def models() -> Dict[str, Any]:
    return {
        "available_models": [
            os.path.join(BASE_DIR, "models", "qwen3-reranker-0.6b"),
            os.path.join(BASE_DIR, "models", "qwen3-reranker-4b"),
            os.path.join(BASE_DIR, "models", "qwen3-reranker-8b"),
        ]
    }


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    candidates = req.candidates or []
    context = req.context or ""

    if not candidates:
        return RerankResponse(
            model=DEFAULT_MODEL,
            context=context,
            scored_candidates=[],
            ranked_candidates=[],
        )

    if len(candidates) == 1:
        return RerankResponse(
            model=DEFAULT_MODEL,
            context=context,
            scored_candidates=[
                ScoredCandidate(text=candidates[0], score=0.0, rank=1)
            ],
            ranked_candidates=candidates,
        )

    try:
        pairs = build_pairs(context, candidates)
        scores = reranker.predict(pairs)

        ranked_rows = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        if req.top_k is not None and req.top_k > 0:
            ranked_rows = ranked_rows[: req.top_k]

        scored_candidates = [
            ScoredCandidate(text=text, score=float(score), rank=idx + 1)
            for idx, (text, score) in enumerate(ranked_rows)
        ]

        ranked_candidates = [text for text, _ in ranked_rows]

        return RerankResponse(
            model=DEFAULT_MODEL,
            context=context,
            scored_candidates=scored_candidates,
            ranked_candidates=ranked_candidates,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))