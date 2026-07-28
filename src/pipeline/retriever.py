"""Stage 1: ANN retrieval via FAISS."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.data.columns import split_clean
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

def _original_values(info: dict, field: str) -> set:
    """
    Lowercased comma-separated values of an original IFDB column.

    Deliberately *not* the `_clean` variant: filters must match what the results
    table displays, or a filter appears to return games that don't have the thing
    the user filtered for.
    """
    return {v.strip().lower() for v in str(info.get(field, "") or "").split(",") if v.strip()}


def apply_hard_filters(
    candidates: List[Tuple[str, float]],
    game_info_map: Dict[str, dict],
    year_range: Optional[str] = None,
    author: Optional[str] = None,
    system: Optional[str] = None,
    tags: Optional[str] = None,
    genre: Optional[str] = None,
    min_rating: Optional[float] = None,
    min_rating_count: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    Filter a candidate list by optional hard constraints (all are AND-ed together).

    year_range:       "YYYY-YYYY" — game's publication year must fall within the range
    author:           substring match (lowercase) against any individual author name
    system:           substring match (lowercase) against any individual system name
    tags:             comma-separated; each must substring-match one of the game's tags
    genre:            substring match against the game's genre values
    min_rating:       lower bound on the game's raw community average rating;
                      also excludes unrated games, whose average is only the prior
    min_rating_count: lower bound on the number of ratings the game has received

    Matching runs against the **original IFDB values**, not the normalised `_clean`
    ones, so a filter only ever matches something the user can actually see in the
    results. Two consequences that motivated this:

      * `tags:IFComp 2025` works. Competition tags are stripped from `tags_clean`,
        so filtering the cleaned values could never match them.
      * `tags:slice of life` no longer returns games that merely have *genre*
        "Slice of life" — genre is folded into `tags_clean` but is not shown in
        the Tags column, which made those matches look like false positives.
        Use `genre:` to search that field explicitly.

    Substring matching keeps it forgiving: "inform" matches "Inform 7", "horror"
    matches "cosmic horror".
    """
    if not any([year_range, author, system, tags, genre,
                min_rating is not None, min_rating_count is not None]):
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
    genre_q = genre.strip().lower() if genre else None
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
            if not any(author_q in a for a in _original_values(info, "author")):
                continue

        if system_q:
            if not any(system_q in s for s in _original_values(info, "system")):
                continue

        if genre_q:
            if not any(genre_q in g for g in _original_values(info, "genre")):
                continue

        if tag_queries:
            game_tags = _original_values(info, "tags")
            if not all(any(q in t for t in game_tags) for q in tag_queries):
                continue

        if min_rating is not None:
            # Filter on the raw community average, not bayesian_avg: someone
            # asking for ≥3.0 should not be handed a game whose actual average is
            # 2.4 and which only cleared the bar because smoothing pulled it
            # toward the 3.5 prior. Unrated games go too — their "average" is
            # nothing but that prior.
            count = info.get("review_count")
            if count is not None:
                try:
                    if int(count) == 0:
                        continue
                except (TypeError, ValueError):
                    pass  # unknown count: fall through to the score comparison
            try:
                # Fall back to bayesian_avg only if the raw average is absent,
                # for game_docs files written before avg_rating was surfaced.
                rating = float(info.get("avg_rating", info.get("bayesian_avg", float("nan"))))
            except (TypeError, ValueError):
                continue
            if not rating >= min_rating:   # the negation also rejects NaN
                continue

        if min_rating_count is not None:
            try:
                if int(info.get("review_count", 0)) < min_rating_count:
                    continue
            except (TypeError, ValueError):
                continue

        filtered.append((gid, score))

    return filtered


def filter_by_tag_overlap(
    candidates: List[Tuple[str, float]],
    game_info_map: Dict[str, dict],
    query_tags: set,
) -> List[Tuple[str, float]]:
    """
    Drop candidates sharing no tag with the query, before the reranker sees them.

    Cheap and, measured over 300 held-out users, free: it removes 11% of the pool
    for full profiles and 39% for short menu-style queries while changing
    Recall@10/25 and NDCG@10/25 by exactly zero. The highest-ranked candidate it
    removes sits around rank 100-400, well below anything displayed — a game that
    shares no tag with the query never reaches the top of the ranking anyway.

    Unlike truncating by cosine rank, this prunes on a signal the query is
    actually made of, which is why it costs nothing.

    Returns the input untouched when there are no query tags to match against, or
    when filtering would empty the pool.
    """
    if not query_tags:
        return candidates
    kept = [
        (gid, score) for gid, score in candidates
        if split_clean(game_info_map.get(gid, {}), "tags") & query_tags
    ]
    return kept or candidates


# Author capping used to live here, applied before truncating the reranker's
# input. Now that the whole candidate pool is reranked there is nothing to
# protect that budget from, and `diversify_results` caps repeat authors on the
# scored list — where it can keep an author's *best* games rather than whichever
# ones happened to rank highest by cosine.
