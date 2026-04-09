"""
retrieval.py – Hybrid BM25 + TF-IDF schema retriever.

Why hybrid is necessary for this schema:
  - Census column descriptions use bureaucratic terminology ("owner occupied",
    "journey to work", "allocation of travel time") that users never say.
  - BM25 (sparse): excellent at exact keyword matching — great for column codes
    (B19013), known terms ("income", "population"), table codes.
  - TF-IDF cosine (dense-ish): captures vocabulary overlap with IDF weighting —
    bridges gaps like "home ownership" → "owner occupied", "commute" → "journey
    to work", "college" → "educational attainment".
  - Reciprocal Rank Fusion (RRF): combines both rank lists without needing to
    normalize scores. No model download required — pure sklearn.

Failure mode of pure BM25 (why we switched):
  "home ownership" → BM25 matched "home" in "Mobile home" and "Worked from home"
  → LLM received wrong context → reported "no available tables"
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

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
        """Full text used for indexing (rich, for TF-IDF)."""
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
    """Minimal BM25 implementation (Okapi BM25)."""

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
        # Document frequency
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
    Appends equivalent Census terms so both BM25 and TF-IDF can match them.
    """
    q_lower = query.lower()
    expansions = []
    for user_term, census_term in CENSUS_SYNONYMS.items():
        if user_term in q_lower:
            expansions.append(census_term)
    if expansions:
        return query + " " + " ".join(expansions)
    return query



RRF_K = 60  # standard RRF constant


class HybridRetriever:
    """
    Combines BM25 (exact keyword) with TF-IDF cosine similarity (soft matching)
    using Reciprocal Rank Fusion.

    No neural model download required — sklearn TF-IDF handles vocabulary
    bridging (e.g. "home ownership" ↔ "owner occupied") through shared subwords
    and IDF-weighted term overlap.
    """

    def __init__(self, entries: List[SchemaEntry]) -> None:
        self.entries = entries
        self._bm25 = BM25Index()
        self._tfidf: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        self._indexed = False

    def build_index(self) -> None:
        if not self.entries:
            return
        texts = [e.as_text() for e in self.entries]
        # Sparse BM25
        self._bm25.fit(texts)
        # TF-IDF with word n-grams (handles partial word overlap)
        self._tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),       # unigrams + bigrams
            sublinear_tf=True,        # log-normalise term frequencies
            min_df=1,
            max_features=50_000,
        )
        try:
            self._tfidf_matrix = self._tfidf.fit_transform(texts)
            self._indexed = True
        except ValueError:
            # Vocabulary is empty (e.g. all stop words / single-char entries)
            # Fall back to BM25-only mode
            logger.warning("TF-IDF vocabulary empty — falling back to BM25 only.")
            self._tfidf = None
            self._tfidf_matrix = None
            self._indexed = True  # BM25 still works
        logger.info("HybridRetriever indexed %d schema entries.", len(self.entries))

    def retrieve(self, query: str, top_k: int = 25) -> List[SchemaEntry]:
        if not self._indexed or not self.entries:
            return self.entries[:top_k]

        # Expand query with Census-specific synonyms before retrieval
        expanded = expand_query(query)

        n = len(self.entries)
        k = min(top_k * 4, n)  # over-fetch before fusion

        # --- BM25 ranks ---
        bm25_hits = self._bm25.score(expanded, k)
        bm25_rank: Dict[int, int] = {idx: rank for rank, (idx, _) in enumerate(bm25_hits)}

        # --- TF-IDF cosine ranks (if available) ---
        if self._tfidf is not None and self._tfidf_matrix is not None:
            q_vec = self._tfidf.transform([expanded])
            sims = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
            tfidf_order = sims.argsort()[::-1][:k]
            tfidf_rank = {int(idx): rank for rank, idx in enumerate(tfidf_order)}
        else:
            tfidf_rank = {}

        # --- Reciprocal Rank Fusion ---
        all_idx = set(bm25_rank) | set(tfidf_rank)
        rrf_scores: Dict[int, float] = {}
        for idx in all_idx:
            bm25_rrf  = 1.0 / (RRF_K + bm25_rank.get(idx,  k + RRF_K))
            tfidf_rrf = 1.0 / (RRF_K + tfidf_rank.get(idx, k + RRF_K))
            rrf_scores[idx] = bm25_rrf + tfidf_rrf

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


def bootstrap_retriever(schema: List[dict]) -> HybridRetriever:
    """Build and cache a HybridRetriever from the schema metadata list."""
    global _retriever
    entries = []
    for row in schema:
        entries.append(SchemaEntry(
            table=row.get("table", ""),
            column=row.get("column", ""),
            data_type=row.get("data_type", ""),
            comment=row.get("comment", ""),
        ))
    ret = HybridRetriever(entries)
    ret.build_index()
    _retriever = ret
    return ret


def get_retriever() -> HybridRetriever:
    """Return the cached retriever (must call bootstrap_retriever first)."""
    if _retriever is None:
        raise RuntimeError("Retriever not initialised — call bootstrap_retriever() first.")
    return _retriever
