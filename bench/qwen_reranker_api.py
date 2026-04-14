import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qwen_reranker_api")

DEFAULT_MODEL = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
reranker: Optional[CrossEncoder] = None


def load_reranker() -> CrossEncoder:
    global reranker
    if reranker is None:
        logger.info("Loading reranker model: %s", DEFAULT_MODEL)
        reranker = CrossEncoder(DEFAULT_MODEL, trust_remote_code=True)
        logger.info("Reranker loaded successfully")
    return reranker


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_reranker()
    yield


app = FastAPI(
    title="Qwen External Reranker API",
    version="1.0.0",
    lifespan=lifespan,
)


class CandidateItem(BaseModel):
    id: int
    text: str


class RerankRequest(BaseModel):
    context: str = ""
    candidates: List[CandidateItem] = Field(default_factory=list)
    top_k: Optional[int] = None


class ScoredCandidate(BaseModel):
    id: int
    text: str
    score: float
    rank: int


class RerankResponse(BaseModel):
    model: str
    context: str
    scored_candidates: List[ScoredCandidate]
    ranked_candidate_ids: List[int]


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model": DEFAULT_MODEL,
        "loaded": reranker is not None,
    }


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    context = (req.context or "").strip()

    candidates = [
        CandidateItem(id=item.id, text=(item.text or "").strip())
        for item in (req.candidates or [])
    ]

    # Remove empty candidates
    candidates = [item for item in candidates if item.text]

    if not candidates:
        return RerankResponse(
            model=DEFAULT_MODEL,
            context=context,
            scored_candidates=[],
            ranked_candidate_ids=[],
        )

    if len(candidates) == 1:
        item = candidates[0]
        return RerankResponse(
            model=DEFAULT_MODEL,
            context=context,
            scored_candidates=[
                ScoredCandidate(
                    id=item.id,
                    text=item.text,
                    score=0.0,
                    rank=1,
                )
            ],
            ranked_candidate_ids=[item.id],
        )

    try:
        model = load_reranker()

        # CrossEncoder expects pairs: [query/context, candidate_text]
        pairs = [[context, item.text] for item in candidates]

        scores = model.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
        )

        ranked_rows = sorted(
            zip(candidates, scores),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )

        if req.top_k is not None and req.top_k > 0:
            ranked_rows = ranked_rows[: req.top_k]

        scored_candidates = [
            ScoredCandidate(
                id=item.id,
                text=item.text,
                score=float(score),
                rank=index + 1,
            )
            for index, (item, score) in enumerate(ranked_rows)
        ]

        ranked_candidate_ids = [item.id for item, _ in ranked_rows]

        return RerankResponse(
            model=DEFAULT_MODEL,
            context=context,
            scored_candidates=scored_candidates,
            ranked_candidate_ids=ranked_candidate_ids,
        )

    except Exception as exc:
        logger.exception("Rerank failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "qwen_reranker_api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )