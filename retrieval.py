"""
retrieval.py – Hybrid Neural Dense + BM25 schema retriever.

Architecture:
  - BM25 (sparse):   Okapi BM25 over tokenised schema text.
                     Excels at exact keyword hits — column codes (B19013),
                     known terms ("income", "population"), table codes.
  - Neural Dense:    Sentence-transformer embeddings (all-MiniLM-L6-v2).
                     Maps user language into a shared semantic space so
                     "home ownership" retrieves "owner occupied tenure" even
                     with zero lexical overlap.
  - RRF Fusion:      Reciprocal Rank Fusion merges both ranked lists without
                     score normalisation. Standard constant k=60.

Why this beats BM25-only or TF-IDF hybrid:
  Pure BM25 failure example — "home ownership":
    BM25 matched "home" in "Mobile home" and "Worked from home"
    → LLM received wrong context → reported "no available tables"
  TF-IDF bridged vocabulary but missed deep semantic synonyms.
  Neural dense embeddings capture the full semantic similarity:
    "commute time" ↔ "travel time to work"  (cosine ~0.82)
    "poverty rate"  ↔ "below poverty level"  (cosine ~0.91)
"""

import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy sentence-transformer import — avoids hard crash if torch isn't installed
# ---------------------------------------------------------------------------

def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        logger.info("Loaded sentence-transformer model: %s", model_name)
        return model
    except ImportError:
        logger.warning(
            "sentence-transformers not installed — neural dense retrieval disabled. "
            "Run: pip install sentence-transformers"
        )
        return None
    except Exception as exc:
        logger.warning("Failed to load sentence-transformer (%s): %s — falling back to BM25 only.", model_name, exc)
        return None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SchemaEntry:
    table: str
    column: str
    data_type: str
    comment: str

    def as_text(self) -> str:
        """Full text used for indexing."""
        parts = [self.table, self.column, self.comment]
        return " ".join(p for p in parts if p)

    def as_display(self) -> str:
        """One-line display for LLM schema context."""
        c = f" -- {self.comment}" if self.comment else ""
        return f"{self.table}.{self.column} ({self.data_type}){c}"


# ---------------------------------------------------------------------------
# BM25 (sparse retrieval)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class BM25Index:
    """Minimal Okapi BM25 implementation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[List[str]] = []
        self._df: Dict[str, int] = {}
        self._avg_dl: float = 0.0
        self._N: int = 0

    def fit(self, documents: List[str]) -> None:
        self._docs = [_tokenize(d) for d in documents]
        self._N = len(self._docs)
        self._df = {}
        for tokens in self._docs:
            for t in set(tokens):
                self._df[t] = self._df.get(t, 0) + 1
        self._avg_dl = sum(len(d) for d in self._docs) / max(self._N, 1)

    def score(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if not self._docs:
            return []
        q_tokens = _tokenize(query)
        scores = []
        for idx, doc in enumerate(self._docs):
            dl = len(doc)
            tf_map: Dict[str, int] = {}
            for t in doc:
                tf_map[t] = tf_map.get(t, 0) + 1
            s = 0.0
            for t in q_tokens:
                if t not in tf_map:
                    continue
                tf = tf_map[t]
                df = self._df.get(t, 0)
                idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1))
                s += idf * num / den
            scores.append((idx, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# Census vocabulary expansion
# (bridges user language → government terminology before retrieval)
# ---------------------------------------------------------------------------

CENSUS_SYNONYMS: Dict[str, str] = {
    # Housing tenure
    "home ownership":    "owner occupied tenure housing",
    "own home":          "owner occupied tenure",
    "own their home":    "owner occupied tenure",
    "homeowner":         "owner occupied tenure",
    "renter":            "renter occupied tenant",
    "rental":            "renter occupied tenant",
    "rent":              "renter occupied tenant",
    # Commute / transportation
    "commute":           "means transportation work journey",
    "drive to work":     "means transportation automobile car drove",
    "public transit":    "means transportation bus subway public",
    "work from home":    "worked from home means transportation",
    # Education
    "college":           "educational attainment bachelor degree",
    "university":        "educational attainment bachelor degree",
    "high school":       "educational attainment high school diploma",
    "dropout":           "educational attainment no diploma",
    # Poverty
    "poverty":           "below poverty level income",
    "poor":              "below poverty level",
    "low income":        "below poverty level income",
    # Age / demographics
    "elderly":           "age 65 years over",
    "senior":            "age 65 years over",
    "children":          "age under 5 years",
    "kids":              "age under 18 years",
    "working age":       "age 18 to 64 years",
    # Race / ethnicity
    "hispanic":          "hispanic latino",
    "latino":            "hispanic latino",
    "black":             "african american black race",
    "asian":             "asian race alone",
    # Health
    "uninsured":         "no health insurance coverage",
    "health coverage":   "health insurance coverage types",
    # Employment
    "unemployed":        "unemployment labor force employment",
    "jobs":              "occupation employed industry",
    # Citizenship / immigration
    "immigrant":         "foreign born nativity citizenship",
    "citizen":           "citizen voting age citizenship",
}


def expand_query(query: str) -> str:
    """
    Expand user query with Census-specific synonyms.
    Appends equivalent Census terms so both BM25 and neural dense can match them.
    """
    q_lower = query.lower()
    expansions = []
    for user_term, census_term in CENSUS_SYNONYMS.items():
        if user_term in q_lower:
            expansions.append(census_term)
    if expansions:
        return query + " " + " ".join(expansions)
    return query


# ---------------------------------------------------------------------------
# Hybrid retriever: Neural Dense + BM25 with RRF
# ---------------------------------------------------------------------------

_DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60  # standard RRF constant


class HybridRetriever:
    """
    Combines BM25 (sparse exact-match) with neural dense embeddings (semantic)
    using Reciprocal Rank Fusion.

    Dense model: all-MiniLM-L6-v2 — 22M params, 384-dim embeddings, ~80ms/query
    on CPU. Downloads once (~90 MB) and is cached by sentence-transformers.

    Graceful degradation: if sentence-transformers is unavailable, falls back
    to BM25-only mode transparently.
    """

    def __init__(self, entries: List[SchemaEntry], model_name: str = _DENSE_MODEL_NAME) -> None:
        self.entries = entries
        self._model_name = model_name
        self._bm25 = BM25Index()
        self._dense_model = None
        self._dense_matrix: Optional[np.ndarray] = None
        self._indexed = False

    def build_index(self) -> None:
        if not self.entries:
            logger.warning("HybridRetriever: no entries to index.")
            return

        texts = [e.as_text() for e in self.entries]

        # --- Sparse BM25 index ---
        self._bm25.fit(texts)
        logger.info("BM25 index built over %d schema entries.", len(texts))

        # --- Neural Dense index ---
        self._dense_model = _load_sentence_transformer(self._model_name)
        if self._dense_model is not None:
            logger.info("Encoding %d schema entries with %s …", len(texts), self._model_name)
            # encode returns (N, 384) float32 numpy array
            self._dense_matrix = self._dense_model.encode(
                texts,
                batch_size=256,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,   # L2-normalised → dot product == cosine similarity
            )
            logger.info(
                "Dense index built: shape=%s, dtype=%s",
                self._dense_matrix.shape,
                self._dense_matrix.dtype,
            )
        else:
            logger.warning("Dense retrieval unavailable — using BM25 only.")

        self._indexed = True

    def retrieve(self, query: str, top_k: int = 25) -> List[SchemaEntry]:
        if not self._indexed or not self.entries:
            return self.entries[:top_k]

        # Expand query with Census-specific synonyms
        expanded = expand_query(query)

        n = len(self.entries)
        k = min(top_k * 4, n)  # over-fetch before RRF fusion

        # --- BM25 sparse ranks ---
        bm25_hits = self._bm25.score(expanded, k)
        bm25_rank: Dict[int, int] = {idx: rank for rank, (idx, _) in enumerate(bm25_hits)}

        # --- Neural dense ranks ---
        dense_rank: Dict[int, int] = {}
        if self._dense_model is not None and self._dense_matrix is not None:
            q_emb = self._dense_model.encode(
                [expanded],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            # dot product of L2-normalised vectors == cosine similarity
            sims = (self._dense_matrix @ q_emb.T).flatten()
            dense_order = sims.argsort()[::-1][:k]
            dense_rank = {int(idx): rank for rank, idx in enumerate(dense_order)}

        # --- Reciprocal Rank Fusion ---
        all_idx = set(bm25_rank) | set(dense_rank)
        rrf_scores: Dict[int, float] = {}
        for idx in all_idx:
            bm25_rrf  = 1.0 / (RRF_K + bm25_rank.get(idx,  k + RRF_K))
            dense_rrf = 1.0 / (RRF_K + dense_rank.get(idx, k + RRF_K))
            rrf_scores[idx] = bm25_rrf + dense_rrf

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [self.entries[idx] for idx, _ in ranked[:top_k]]

    def schema_context(self, query: str, top_k: int = 25) -> str:
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return "(no matching schema entries)"
        return "\n".join(e.as_display() for e in hits)


# ---------------------------------------------------------------------------
# Singleton retriever (cached per process)
# ---------------------------------------------------------------------------

_retriever: Optional[HybridRetriever] = None


def bootstrap_retriever(schema: List[dict], model_name: str = _DENSE_MODEL_NAME) -> HybridRetriever:
    """Build and cache a HybridRetriever from the schema metadata list."""
    global _retriever
    entries = [
        SchemaEntry(
            table=row.get("table", ""),
            column=row.get("column", ""),
            data_type=row.get("data_type", ""),
            comment=row.get("comment", ""),
        )
        for row in schema
    ]
    ret = HybridRetriever(entries, model_name=model_name)
    ret.build_index()
    _retriever = ret
    return ret


def get_retriever() -> HybridRetriever:
    """Return the cached retriever (must call bootstrap_retriever first)."""
    if _retriever is None:
        raise RuntimeError("Retriever not initialised — call bootstrap_retriever() first.")
    return _retriever
