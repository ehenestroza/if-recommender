"""Stage 2: cross-encoder reranking and offline evaluation metrics."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from src.data.columns import clean_value, split_clean

logger = logging.getLogger(__name__)


class Reranker:
    """Score (query, document) pairs with a cross-encoder and return top-K."""

    def __init__(self, model_name: str) -> None:
        logger.info("Loading cross-encoder: %s", model_name)
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query_text: str,
        candidates: List[Tuple[str, float]],
        game_doc_lookup: Dict[str, str],
        top_k: int = 10,
        bayesian_avg_map: Optional[Dict[str, float]] = None,
        rating_weight: float = 0.5,
        max_rating: float = 5.0,
        min_ce_score: Optional[float] = None,
    ) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
        """
        Re-score candidates with the cross-encoder and return the top-K.

        Raw cross-encoder logits are passed through sigmoid to yield 0–1 relevance
        probabilities. Candidates below min_ce_score are dropped before anything else.

        If bayesian_avg_map is given, the final score blends relevance with rating
        on their own absolute scales:

            score = (1 - rating_weight) * ce_prob + rating_weight * (bay_avg / max_rating)

        Both terms are absolute rather than pool-relative, deliberately. Rescaling
        within the candidate pool would force the best candidate toward 1.0 even
        for a niche query with nothing genuinely relevant in it, overstating the
        match; keeping absolute scales lets a weak pool score like a weak pool,
        and lets scores mean the same thing across queries.

        Note that Bayesian smoothing leaves the rating term spanning only about
        0.54–0.81 against relevance's ~0–1, so rating moves the score roughly
        three times less than `rating_weight` suggests. That imbalance is load
        bearing: equalising the two terms measurably *hurts* ranking quality
        (−0.015 NDCG@10 over 300 held-out users), because rating is the weaker
        and noisier signal.

        candidates:       list of (gameid, retrieval_score) from the bi-encoder stage
        game_doc_lookup:  gameid → document text
        bayesian_avg_map: optional gameid → bayesian_avg for rating blending
        min_ce_score:     drop candidates with sigmoid cross-encoder score below this

        Returns (ranked [(gameid, score)], {gameid: relevance}) — relevance is the
        unblended cross-encoder probability, kept for display.
        """
        if not candidates:
            return [], {}

        game_ids = [gid for gid, _ in candidates]
        pairs = [(query_text, game_doc_lookup.get(gid, "")) for gid in game_ids]

        raw = self.model.predict(pairs, show_progress_bar=False)
        ce_scores = torch.sigmoid(torch.tensor(raw, dtype=torch.float32)).numpy()
        relevance = {gid: float(s) for gid, s in zip(game_ids, ce_scores)}

        # Filter by cross-encoder score before blending
        if min_ce_score is not None:
            kept = [(gid, float(s)) for gid, s in zip(game_ids, ce_scores) if s >= min_ce_score]
        else:
            kept = [(gid, float(s)) for gid, s in zip(game_ids, ce_scores)]

        if bayesian_avg_map is not None:
            results = []
            for gid, ce_s in kept:
                bay_s = bayesian_avg_map.get(gid, max_rating / 2) / max_rating
                score = (1.0 - rating_weight) * ce_s + rating_weight * bay_s
                results.append((gid, score))
        else:
            results = kept

        return sorted(results, key=lambda x: -x[1])[:top_k], relevance


# ---------------------------------------------------------------------------
# Post-rerank diversity
# ---------------------------------------------------------------------------

def order_by_relevance(
    scored: List[Tuple[str, float]],
    relevance: Dict[str, float],
) -> List[Tuple[str, float]]:
    """
    Re-key a score-ordered pool onto relevance, for display.

    The blended score still decides *which* candidates exist: it is what
    `min_rerank_score` filtered on and what every top-N truncation upstream was
    applied to, both here and in the precomputed tables. This decides only the
    order they are shown in.

    It swaps the value carried in each pair rather than sorting after the fact,
    so everything downstream reads relevance too — the author cap in
    `diversify_results` keeps an author's most *relevant* game rather than their
    best-rated one, and the coverage pass and its final sort agree with what the
    reader sees.

    Sorting is stable, so ties keep the pool's existing order, which is score.
    Among equally relevant games the better-rated one still comes first — a
    tiebreak rather than a thumb on the scale.

    Shared by the CLI and the web app so the two cannot show different orders.
    """
    return sorted(
        ((gid, relevance.get(gid, 0.0)) for gid, _ in scored),
        key=lambda pair: -pair[1],
    )


def select_results(
    scored: List[Tuple[str, float]],
    hard_filters: Optional[dict],
    game_info_map: Optional[Dict[str, dict]],
    top_k: int,
    use_diversity: bool = True,
    target_genres: Optional[set] = None,
    target_systems: Optional[set] = None,
) -> List[Tuple[str, float]]:
    """
    Turn a scored candidate pool into the final page: filter, then diversify.

    Shared by the CLI and the web app so the two cannot drift apart — the order
    of these steps is load-bearing (filtering before diversification is what lets
    an `author:` filter disable the variety cap).
    """
    from src.pipeline.retriever import apply_hard_filters  # local: avoids a cycle

    results = scored
    if hard_filters and game_info_map is not None:
        results = apply_hard_filters(results, game_info_map, **hard_filters)

    if game_info_map is None:
        return results[:top_k]

    genres = (target_genres or set()) if use_diversity else set()
    systems = (target_systems or set()) if use_diversity else set()
    return diversify_results(
        results, game_info_map, genres, systems, top_k,
        cap_authors="author" not in (hard_filters or {}),
    )


def _game_genres(game_info_map: Dict[str, dict], gid: str) -> set:
    return {g.strip() for g in str(game_info_map.get(gid, {}).get("genre", "")).split(",") if g.strip()}


def _game_system(game_info_map: Dict[str, dict], gid: str) -> str:
    """Normalised system string — target systems come from cleaned profile text."""
    return clean_value(game_info_map.get(gid, {}), "system").strip()


def _coverage(
    candidates: List[Tuple[str, float]],
    game_info_map: Dict[str, dict],
) -> Tuple[set, set]:
    covered_genres: set = set()
    covered_systems: set = set()
    for gid, _ in candidates:
        covered_genres |= _game_genres(game_info_map, gid)
        s = _game_system(game_info_map, gid)
        if s:
            covered_systems.add(s)
    return covered_genres, covered_systems




def diversify_results(
    candidates: List[Tuple[str, float]],
    game_info_map: Dict[str, dict],
    target_genres: set,
    target_systems: set,
    top_k: int,
    max_author_appearances: int = 2,
    cap_authors: bool = True,
) -> List[Tuple[str, float]]:
    """
    Select top_k from a fully scored + sorted candidate list with two passes:

    1. Author variety — a candidate is set aside if adding it would let one author
       appear more than max_author_appearances times, so a prolific author cannot
       fill the page. The cap yields in two situations where it would work against
       the user:

         * `cap_authors=False` — the caller filtered *by* author, so wanting more
           than two of their games is the entire point of the request.
         * Too few results — if the cap leaves fewer than top_k, the highest-scored
           set-aside candidates are added back until the page is full. Variety the
           user can see is worth less than the slots they asked for.

    2. Coverage — if the initial top_k misses a target genre or system,
       the highest-scored remaining candidate covering that target is swapped
       in, displacing the lowest-scoring initial item.

    candidates:     all scored (gameid, score) pairs, sorted descending by score
    game_info_map:  gameid → info dict; author and system are read from their
                    normalised `_clean` variants, matching the profile-derived targets
    target_genres:  genre strings to aim to cover (from user profile or seed game)
    target_systems: system strings to aim to cover
    cap_authors:    False to disable the variety cap entirely
    """
    # Pass 1: author variety. Overflow is kept rather than discarded so it can
    # backfill a short page.
    deduped: List[Tuple[str, float]] = []
    overflow: List[Tuple[str, float]] = []
    if not cap_authors:
        deduped = list(candidates)
    else:
        author_counts: Dict[str, int] = {}
        for gid, score in candidates:
            info = game_info_map.get(gid, {})
            game_authors = split_clean(info, "author")
            if any(author_counts.get(a, 0) >= max_author_appearances for a in game_authors):
                overflow.append((gid, score))
                continue
            deduped.append((gid, score))
            for a in game_authors:
                author_counts[a] = author_counts.get(a, 0) + 1

        if len(deduped) < top_k and overflow:
            shortfall = top_k - len(deduped)
            logger.debug("Author cap left %d of %d slots; backfilling %d",
                         len(deduped), top_k, min(shortfall, len(overflow)))
            deduped = sorted(deduped + overflow[:shortfall], key=lambda pair: -pair[1])

    if not target_genres and not target_systems:
        return deduped[:top_k]

    if len(deduped) <= top_k:
        return deduped

    # Pass 2: coverage enforcement on the deduped pool
    initial = deduped[:top_k]
    rest = deduped[top_k:]

    covered_genres, covered_systems = _coverage(initial, game_info_map)
    missing_genres = target_genres - covered_genres
    missing_systems = target_systems - covered_systems

    if not missing_genres and not missing_systems:
        return initial

    inserts: List[Tuple[str, float]] = []
    used: set = set()  # indices into rest already claimed

    for genre in sorted(missing_genres):
        if any(genre in _game_genres(game_info_map, gid) for gid, _ in inserts):
            continue  # already covered by a prior insert
        for i, (gid, score) in enumerate(rest):
            if i in used:
                continue
            if genre in _game_genres(game_info_map, gid):
                inserts.append((gid, score))
                used.add(i)
                break

    for system in sorted(missing_systems):
        if any(_game_system(game_info_map, gid) == system for gid, _ in inserts):
            continue
        for i, (gid, score) in enumerate(rest):
            if i in used:
                continue
            if _game_system(game_info_map, gid) == system:
                inserts.append((gid, score))
                used.add(i)
                break

    if not inserts:
        return initial

    n_keep = max(0, top_k - len(inserts))
    result = initial[:n_keep] + inserts
    result.sort(key=lambda x: -x[1])
    return result[:top_k]


# ---------------------------------------------------------------------------
# Offline evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, set],
    ks: Tuple[int, ...] = (1, 5, 10, 20, 50),
) -> Dict[str, float]:
    """
    Compute Recall@K, NDCG@K, and MRR against ground-truth positives.

    predictions:   {userid → ordered list of predicted gameids}
    ground_truth:  {userid → set of relevant gameids}
    """
    max_k = max(ks)

    recall_sums: Dict[int, float] = {k: 0.0 for k in ks}
    ndcg_sums:   Dict[int, float] = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    n_users = 0

    for uid, relevant in ground_truth.items():
        if uid not in predictions or not relevant:
            continue

        pred = predictions[uid][:max_k]
        n_users += 1

        # MRR
        for rank, gid in enumerate(pred, start=1):
            if gid in relevant:
                mrr_sum += 1.0 / rank
                break

        for k in ks:
            pred_k = pred[:k]
            hits = sum(1 for gid in pred_k if gid in relevant)

            # Recall@K
            recall_sums[k] += hits / len(relevant)

            # NDCG@K
            dcg = sum(
                1.0 / np.log2(i + 2)
                for i, gid in enumerate(pred_k)
                if gid in relevant
            )
            ideal_len = min(len(relevant), k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))
            ndcg_sums[k] += dcg / idcg if idcg > 0 else 0.0

    if n_users == 0:
        logger.warning("No users with predictions and ground truth found")
        return {}

    metrics: Dict[str, float] = {"MRR": mrr_sum / n_users}
    for k in ks:
        metrics[f"Recall@{k}"] = recall_sums[k] / n_users
        metrics[f"NDCG@{k}"]   = ndcg_sums[k]   / n_users

    return metrics
