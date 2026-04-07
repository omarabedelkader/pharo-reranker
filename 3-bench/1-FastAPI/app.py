import os
import json
from collections import Counter
from typing import List, Optional, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostRanker, Pool

# ==================================================
# Paths
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

TFIDF_PATH = os.path.join(BASE_DIR, "artifacts", "tfidf.joblib")
EMB_PATH = os.path.join(BASE_DIR, "artifacts", "embedder")

# ==================================================
# External feature models
# ==================================================
from joblib import load
from sentence_transformers import SentenceTransformer

tfidf_vectorizer = load(TFIDF_PATH)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# ==================================================
# Model registry
# ==================================================
MODEL_SPECS = {
    "catboost-base": {
        "dir": "catboost",
        "model": "ranker_v1.cbm",
        "meta": "ranker_v1.meta.json",
    },
    "catboost-bm25": {
        "dir": "catboost-bm25",
        "model": "ranker_bm25.cbm",
        "meta": "ranker_bm25.meta.json",
    },
    "catboost-emb": {
        "dir": "catboost-emb",
        "model": "ranker_emb.cbm",
        "meta": "ranker_emb.meta.json",
    },
    "catboost-tfidf": {
        "dir": "catboost-tfidf",
        "model": "ranker_tfidf.cbm",
        "meta": "ranker_tfidf.meta.json",
    },
    "catboost-full": {
        "dir": "catboost-full",
        "model": "ranker_full.cbm",
        "meta": "ranker_full.meta.json",
    },
}

# ==================================================
# Runtime feature registry (authoritative)
# ==================================================
RUNTIME_FEATURES = {
    "tok_overlap",
    "char_jaccard_3",
    "ctx_len",
    "cand_len",
    "len_diff",
    "len_ratio",
    "ctx_tokens",
    "cand_tokens",
    "bm25",
    "bm25_norm",
    "emb_cos",
    "tfidf_cos",
}

# ==================================================
# Load models + validate meta
# ==================================================
MODELS: Dict[str, Dict[str, Any]] = {}

for name, spec in MODEL_SPECS.items():
    model_dir = os.path.join(MODELS_DIR, spec["dir"])
    model_path = os.path.join(model_dir, spec["model"])
    meta_path = os.path.join(model_dir, spec["meta"])

    if not os.path.isfile(model_path):
        raise RuntimeError(f"[{name}] Missing model file: {model_path}")

    if not os.path.isfile(meta_path):
        raise RuntimeError(f"[{name}] Missing meta file: {meta_path}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    feature_names = meta.get("feature_names")
    if not feature_names:
        raise RuntimeError(f"[{name}] meta.json missing feature_names")

    unknown = set(feature_names) - RUNTIME_FEATURES
    if unknown:
        raise RuntimeError(f"[{name}] Unknown features: {sorted(unknown)}")

    model = CatBoostRanker()
    model.load_model(model_path)

    MODELS[name] = {
        "model": model,
        "feature_names": feature_names,
    }

print("Loaded models:")
for k, v in MODELS.items():
    print(f" - {k}: {v['feature_names']}")

# ==================================================
# FastAPI app
# ==================================================
app = FastAPI(title="Multi-Model CatBoost Ranker API")

# ==================================================
# Request schema
# ==================================================
class RankRequest(BaseModel):
    context: Optional[str] = ""
    candidates: List[str]
    model: Optional[str] = "catboost-base"

# ==================================================
# Tokenization + similarity
# ==================================================
def tokenize(text: str) -> List[str]:
    return text.lower().split()


def overlap_ratio(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    return len(ta & tb) / len(ta) if ta else 0.0


def char_ngrams(s: str, n: int = 3) -> set:
    if len(s) < n:
        return set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def char_jaccard(a: str, b: str, n: int = 3) -> float:
    na, nb = char_ngrams(a, n), char_ngrams(b, n)
    return len(na & nb) / len(na) if na else 0.0


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

# ==================================================
# BM25 (single-query)
# ==================================================
def bm25_score(query: str, doc: str, k1: float = 1.5, b: float = 0.75) -> float:
    q_terms = tokenize(query)
    d_terms = tokenize(doc)

    if not q_terms or not d_terms:
        return 0.0

    tf = Counter(d_terms)
    doc_len = len(d_terms)
    avgdl = max(doc_len, 1)

    score = 0.0
    for term in q_terms:
        f = tf.get(term, 0)
        if f == 0:
            continue
        denom = f + k1 * (1 - b + b * doc_len / avgdl)
        score += (f * (k1 + 1)) / denom

    return score

# ==================================================
# Feature builder (model-aware)
# ==================================================
def build_feature_matrix(
    context: str,
    candidates: List[str],
    feature_names: List[str],
) -> np.ndarray:
    ctx = context or ""

    needs_bm25 = {"bm25", "bm25_norm"} & set(feature_names)
    needs_emb = "emb_cos" in feature_names
    needs_tfidf = "tfidf_cos" in feature_names

    if needs_emb:
        ctx_emb = embedder.encode(ctx, normalize_embeddings=True)
        cand_embs = embedder.encode(candidates, normalize_embeddings=True)

    if needs_tfidf:
        tfidf = tfidf_vectorizer.transform([ctx] + candidates)
        ctx_tfidf = tfidf[0].toarray()[0]
        cand_tfidf = tfidf[1:]

    bm25_scores = [bm25_score(ctx, c) for c in candidates] if needs_bm25 else []
    max_bm25 = max(bm25_scores) if bm25_scores else 1.0
    max_bm25 = max(max_bm25, 1e-8)

    rows = []

    for i, cand in enumerate(candidates):
        feats = {
            "tok_overlap": overlap_ratio(ctx, cand),
            "char_jaccard_3": char_jaccard(ctx, cand),
            "ctx_len": len(ctx),
            "cand_len": len(cand),
            "len_diff": abs(len(ctx) - len(cand)),
            "len_ratio": len(cand) / max(len(ctx), 1),
            "ctx_tokens": len(tokenize(ctx)),
            "cand_tokens": len(tokenize(cand)),
        }

        if needs_bm25:
            feats["bm25"] = bm25_scores[i]
            feats["bm25_norm"] = bm25_scores[i] / max_bm25

        if needs_emb:
            feats["emb_cos"] = cosine_sim(ctx_emb, cand_embs[i])

        if needs_tfidf:
            feats["tfidf_cos"] = cosine_sim(
                ctx_tfidf, cand_tfidf[i].toarray()[0]
            )

        rows.append([feats[name] for name in feature_names])

    return np.asarray(rows, dtype=np.float32)

# ==================================================
# Ranking endpoint
# ==================================================
@app.post("/rank")
def rank(req: RankRequest) -> Dict[str, Any]:
    model_name = req.model or "catboost-base"

    if model_name not in MODELS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Unknown model '{model_name}'",
                "available_models": list(MODELS.keys()),
            },
        )

    candidates = req.candidates or []
    context = req.context or ""

    if len(candidates) <= 1:
        return {
            "model": model_name,
            "ranked_candidates": candidates,
            "scores": [0.0] * len(candidates),
        }

    pruned = sorted(candidates, key=lambda c: (len(c), c))[:20]

    entry = MODELS[model_name]
    X = build_feature_matrix(context, pruned, entry["feature_names"])

    pool = Pool(data=X, feature_names=entry["feature_names"])
    scores = entry["model"].predict(pool)

    ranked = [
        c for c, _ in sorted(
            zip(pruned, scores),
            key=lambda x: x[1],
            reverse=True,
        )
    ]

    return {
        "model": model_name,
        "ranked_candidates": ranked,
        "scores": scores.tolist(),
    }

# ==================================================
# Model listing
# ==================================================
@app.get("/models")
def list_models():
    return {"available_models": list(MODELS.keys())}
