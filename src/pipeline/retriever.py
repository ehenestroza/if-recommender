"""Stage 1: ANN retrieval via FAISS."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.index.faiss_index import GameIndex

logger = logging.getLogger(__name__)


class Retriever:
    """
    Encode a query and retrieve the nearest game embeddings from FAISS.

    Supports three query modes:
      text     – encode a free-text string directly
      userid   – encode the user's pre-built profile text
      game_ids – average the embeddings of seed games ("more like these")
    """

    def __init__(
        self,
        model: Optional[SentenceTransformer],
        index: GameIndex,
        user_profiles: Dict[str, str],
        game_embeddings: Dict[str, np.ndarray],
        game_query_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.bi_encoder = model
        self.index = index
        self.user_profiles = user_profiles
        self.game_embeddings = game_embeddings
        # Query-encoder embeddings of game profile texts (no description).
        # Used for game_id queries so we search doc-space from the query side.
        self.game_query_embeddings: Dict[str, np.ndarray] = game_query_embeddings or {}

    # ------------------------------------------------------------------
    # Query encoders
    # ------------------------------------------------------------------

    def _encode_text(self, text: str) -> np.ndarray:
        emb = self.bi_encoder.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return emb[0]

    def _encode_userid(self, userid: str) -> Optional[np.ndarray]:
        profile = self.user_profiles.get(userid)
        if not profile:
            logger.warning("No profile found for user '%s'", userid)
            return None
        return self._encode_text(profile)

    def _encode_game_ids(self, game_ids: List[str]) -> Optional[np.ndarray]:
        # Prefer query-space embeddings (encoded by the query encoder) so that
        # the search vector lives in the same space as user-profile queries.
        # Fall back to doc-space embeddings if query embeddings aren't available.
        lookup = self.game_query_embeddings if self.game_query_embeddings else self.game_embeddings
        embs = [lookup[gid] for gid in game_ids if gid in lookup]
        if not embs:
            logger.warning("None of the seed game IDs found in embeddings")
            return None
        avg = np.mean(embs, axis=0).astype(np.float32)
        norm = np.linalg.norm(avg)
        return avg / norm if norm > 0 else avg

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query,
        query_type: str = "text",
        top_k: int = 100,
        min_score: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """
        Return a ranked list of (gameid, score) pairs.

        query:     str for text/userid, list[str] for game_ids
        min_score: when set, returns all candidates above this cosine similarity
                   threshold (top_k is ignored for the FAISS call)
        """
        if query_type == "text":
            emb = self._encode_text(query)
        elif query_type == "userid":
            emb = self._encode_userid(query)
        elif query_type == "game_ids":
            emb = self._encode_game_ids(query)
        else:
            raise ValueError(f"Unknown query_type: {query_type!r}")

        if emb is None:
            return []

        return self.index.search(emb, top_k=top_k, min_score=min_score)


# ---------------------------------------------------------------------------
# Hard filtering
# ---------------------------------------------------------------------------

def apply_hard_filters(
    candidates: List[Tuple[str, float]],
    game_info_map: Dict[str, dict],
    year_range: Optional[str] = None,
    author: Optional[str] = None,
    system: Optional[str] = None,
    tags: Optional[str] = None,
    min_rating: Optional[float] = None,
    min_rating_count: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    Filter a candidate list by optional hard constraints (all are AND-ed together).

    year_range:       "YYYY-YYYY" — game's publication year must fall within the range
    author:           substring match (lowercase) against any individual author name
    system:           substring match (lowercase) against any individual system name
    tags:             comma-separated; every listed tag must appear in the game's tag set
    min_rating:       lower bound on the game's Bayesian-average rating
    min_rating_count: lower bound on the number of ratings the game has received
    """
    if not any([year_range, author, system, tags, min_rating is not None, min_rating_count is not None]):
        return candidates

    year_lo: Optional[int] = None
    year_hi: Optional[int] = None
    if year_range:
        parts = year_range.split("-")
        if len(parts) == 2:
            try:
                year_lo, year_hi = int(parts[0]), int(parts[1])
            except ValueError:
                logger.warning("Invalid year_range %r — ignoring", year_range)

    author_q = author.strip().lower() if author else None
    system_q = system.strip().lower() if system else None
    tag_queries: set = (
        {t.strip().lower() for t in tags.split(",") if t.strip()} if tags else set()
    )

    filtered: List[Tuple[str, float]] = []
    for gid, score in candidates:
        info = game_info_map.get(gid, {})

        if year_lo is not None or year_hi is not None:
            year_str = str(info.get("year", "")).strip()
            if not year_str:
                continue
            try:
                year = int(year_str)
            except ValueError:
                continue
            if year_lo is not None and year < year_lo:
                continue
            if year_hi is not None and year > year_hi:
                continue

        if author_q:
            game_authors = {
                a.strip().lower()
                for a in str(info.get("author", "")).split(",")
                if a.strip()
            }
            if not any(author_q in a for a in game_authors):
                continue

        if system_q:
            game_systems = {
                s.strip().lower()
                for s in str(info.get("system", "")).split(",")
                if s.strip()
            }
            if not any(system_q in s for s in game_systems):
                continue

        if tag_queries:
            game_tags = {
                t.strip().lower()
                for t in str(info.get("tags", "")).split(",")
                if t.strip()
            }
            if not tag_queries.issubset(game_tags):
                continue

        if min_rating is not None:
            try:
                if float(info.get("bayesian_avg", 0)) < min_rating:
                    continue
            except (TypeError, ValueError):
                continue

        if min_rating_count is not None:
            try:
                if int(info.get("review_count", 0)) < min_rating_count:
                    continue
            except (TypeError, ValueError):
                continue

        filtered.append((gid, score))

    return filtered


def cap_candidates_by_author(
    candidates: List[Tuple[str, float]],
    game_info_map: Dict[str, dict],
    max_per_author: int = 2,
) -> List[Tuple[str, float]]:
    """
    Retain at most max_per_author games per individual author, preserving
    the order of candidates (highest retrieval score first).

    This runs before top_k_retrieve truncation so that a prolific author
    cannot crowd out the reranker input even when they dominate cosine scores.
    """
    author_counts: Dict[str, int] = {}
    result: List[Tuple[str, float]] = []
    for gid, score in candidates:
        info = game_info_map.get(gid, {})
        authors = {
            a.strip().lower()
            for a in str(info.get("author", "")).split(",")
            if a.strip()
        }
        if any(author_counts.get(a, 0) >= max_per_author for a in authors):
            continue
        result.append((gid, score))
        for a in authors:
            author_counts[a] = author_counts.get(a, 0) + 1
    return result
