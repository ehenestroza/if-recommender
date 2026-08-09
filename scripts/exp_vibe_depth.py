#!/usr/bin/env python
"""
Experiment: what does capping the vibe reranker's candidate pool cost?

`vibe` is the only mode scored live, and it currently reranks every candidate
above the cosine floor. This measures what a two-pass alternative — bi-encoder
retrieves the top-C, cross-encoder reranks only those — costs in quality and in
consistency, across a sweep of C.

Method
------
Vibe queries have no ground truth of their own, so they are reconstructed from
held-out users: each test user's profile is truncated to a menu-style pick
(N systems + M tags), which is exactly the shape the UI produces, and scored
against that user's test-split positives.

The cross-encoder score of a (query, document) pair does not depend on which
other candidates are in the pool, so every cap is derived from a single full
scoring per query rather than rescored. The cap variants are therefore exact,
not approximations.

Two cap orders are measured, because they are not the same thing:

  post   cap applied after the tag pre-filter — what `_rank` in app.py does today
         if `rerank_pool_cap` is set
  pre    cap applied to the raw cosine ranking, before the tag pre-filter — the
         literal "retrieve top-C, then rerank" two-pass reading

Usage
-----
    uv run scripts/exp_vibe_depth.py --users 300
    uv run scripts/exp_vibe_depth.py --users 300 --out outputs/vibe_depth.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.env import configure_logging
configure_logging()

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "recommender", Path(__file__).resolve().parent / "06_run_recommender.py"
)
recommender = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recommender)

from src.data.preprocessor import format_profile_text, parse_profile_text
from src.pipeline.ranker import select_results
from src.pipeline.retriever import filter_by_tag_overlap

logger = logging.getLogger("exp_vibe_depth")

# Caps swept. None is the current behaviour: score everything above the floor.
CAPS: Sequence[Optional[int]] = (50, 100, 150, 200, 300, 500, 750, 1000, 1500, None)

# Menu-style query shapes. A vibe query is a handful of picks, not a full
# profile, and pool size depends strongly on how many picks there are.
SHAPES = {
    "narrow": (1, 3),   # 1 system, 3 tags
    "broad":  (2, 6),   # 2 systems, 6 tags
}

TOP_K = 25          # the page the app shows
KS = (10, 25)

# Filters used for the consistency-under-filtering measurement. Chosen to be
# query-independent so the same battery applies to every query, and to bracket
# what the UI actually offers: a period, a quality bar, and a platform.
FILTERS = {
    "year:2015-2025":      {"year_range": "2015-2025"},
    "rating>=3.5 (n>=3)":  {"min_rating": 3.5, "min_rating_count": 3},
    "system:twine":        {"system": "twine"},
}


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------

def build_vibe_queries(
    profile_map: Dict[str, str],
    ground_truth: Dict[str, set],
    n_systems: int,
    n_tags: int,
    limit: int,
    seed: int = 0,
) -> List[Tuple[str, str]]:
    """
    Reconstruct menu-style vibe queries from held-out user profiles.

    Profile values are ordered by how often they appear in the games the user
    liked, so truncating from the front keeps the picks most characteristic of
    that user — the same thing someone does when they choose from a dropdown.

    Returns [(userid, query_text)], skipping users whose truncated profile is
    empty (no systems and no tags survive).
    """
    rng = np.random.default_rng(seed)
    uids = sorted(u for u in ground_truth if profile_map.get(u))
    rng.shuffle(uids)

    queries: List[Tuple[str, str]] = []
    for uid in uids:
        systems, tags = parse_profile_text(profile_map[uid])
        systems, tags = systems[:n_systems], tags[:n_tags]
        if not systems and not tags:
            continue
        queries.append((uid, format_profile_text(systems, tags)))
        if len(queries) >= limit:
            break
    return queries


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_query(
    query_text: str,
    query_encoder,
    retriever,
    reranker,
    doc_map: Dict[str, str],
    game_info_map: Dict[str, dict],
    min_retrieval_score: float,
) -> Tuple[List[Tuple[str, float]], List[str], Dict[str, float]]:
    """
    Score a query's entire candidate pool once.

    Returns (raw_candidates, tag_kept_ids, ce_scores):
      raw_candidates  every (gameid, cosine) above the retrieval floor, cosine order
      tag_kept_ids    the subset surviving the tag pre-filter, cosine order
      ce_scores       gameid → sigmoid cross-encoder score, for every raw candidate

    Every cap variant is a subset of this, so nothing is rescored downstream.
    """
    emb = query_encoder.encode([query_text], normalize_embeddings=True,
                               show_progress_bar=False)[0]
    raw = retriever.index.search(emb, min_score=min_retrieval_score)
    if not raw:
        return [], [], {}

    _, query_tags = parse_profile_text(query_text)
    kept = (filter_by_tag_overlap(raw, game_info_map, set(query_tags))
            if query_tags else raw)

    gids = [gid for gid, _ in raw]
    pairs = [(query_text, doc_map.get(gid, "")) for gid in gids]
    logits = reranker.model.predict(pairs, show_progress_bar=False)
    probs = torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).numpy()
    ce_scores = {gid: float(p) for gid, p in zip(gids, probs)}

    return raw, [gid for gid, _ in kept], ce_scores


def rank_for_cap(
    raw: List[Tuple[str, float]],
    tag_kept: List[str],
    ce_scores: Dict[str, float],
    cap: Optional[int],
    order: str,
    bayesian_avg_map: Optional[Dict[str, float]],
    rating_weight: float,
    min_ce_score: Optional[float],
) -> Tuple[List[Tuple[str, float]], int]:
    """
    Reproduce the pipeline's scored list for one cap, from the cached scores.

    order: "post" caps after the tag pre-filter, "pre" caps the raw cosine list
    before it. Returns (scored list sorted by blended score, n_pairs the
    cross-encoder would actually have had to run).
    """
    if order == "post":
        pool = tag_kept if cap is None else tag_kept[:cap]
    elif order == "pre":
        raw_ids = [gid for gid, _ in raw] if cap is None else [gid for gid, _ in raw[:cap]]
        head = set(raw_ids)
        pool = [gid for gid in tag_kept if gid in head]
    else:
        raise ValueError(f"Unknown order {order!r}")

    n_pairs = len(pool)

    scored: List[Tuple[str, float]] = []
    for gid in pool:
        ce = ce_scores.get(gid)
        if ce is None or (min_ce_score is not None and ce < min_ce_score):
            continue
        if bayesian_avg_map is not None:
            bay = bayesian_avg_map.get(gid, 2.5) / 5.0
            score = (1.0 - rating_weight) * ce + rating_weight * bay
        else:
            score = ce
        scored.append((gid, score))

    scored.sort(key=lambda pair: -pair[1])
    return scored, n_pairs


def final_page(
    scored: List[Tuple[str, float]],
    game_info_map: Dict[str, dict],
    hard_filters: Optional[dict],
    use_diversity: bool,
) -> List[str]:
    """
    The list the app would display, in order.

    Mirrors app.py: ask `select_results` for the whole pool and paginate, rather
    than asking for 25 — the author-cap backfill behaves differently at the two
    top_k values, and production takes the former path.
    """
    if not scored:
        return []
    results = select_results(
        scored, hard_filters, game_info_map, len(scored),
        use_diversity=use_diversity, target_genres=set(), target_systems=set(),
    )
    return [gid for gid, _ in results]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(pred: Sequence[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for gid in pred[:k] if gid in relevant) / len(relevant)


def ndcg_at_k(pred: Sequence[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, gid in enumerate(pred[:k]) if gid in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(pred: Sequence[str], relevant: set) -> float:
    for rank, gid in enumerate(pred, start=1):
        if gid in relevant:
            return 1.0 / rank
    return 0.0


def overlap_at_k(pred: Sequence[str], reference: Sequence[str], k: int) -> float:
    """Fraction of the reference's top-k that the variant also shows in its top-k."""
    ref = set(reference[:k])
    if not ref:
        return 1.0
    return len(ref & set(pred[:k])) / len(ref)


def rank_displacement(pred: Sequence[str], reference: Sequence[str], k: int) -> float:
    """
    Mean |rank change| for reference top-k items the variant still shows.

    Items the variant dropped entirely are excluded — `overlap_at_k` already
    counts those, and scoring them as a fixed penalty would conflate two
    different failures.
    """
    pos = {gid: i for i, gid in enumerate(pred)}
    deltas = [abs(pos[gid] - i) for i, gid in enumerate(reference[:k]) if gid in pos]
    return float(np.mean(deltas)) if deltas else 0.0


def paired_bootstrap(
    variant: Sequence[float],
    reference: Sequence[float],
    n_boot: int = 2000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Mean paired delta (variant − reference) and its 95% CI, resampling users."""
    v = np.asarray(variant, dtype=float)
    r = np.asarray(reference, dtype=float)
    diff = v - r
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_shape(
    shape: str,
    n_systems: int,
    n_tags: int,
    args,
    cfg,
    artefacts,
) -> dict:
    (retriever, reranker, query_encoder, game_docs, doc_map, profile_map, name_map,
     bayesian_avg_map, reviews_df, playedgames_df, game_query_text_map,
     game_info_map, *_rest) = artefacts

    retr_cfg = cfg["retrieval"]
    min_retrieval_score = retr_cfg.get("min_retrieval_score", 0.25)
    min_ce_score = retr_cfg.get("min_rerank_score", 0.10)
    rating_weight = retr_cfg.get("rating_weight", 0.5)
    use_diversity = retr_cfg.get("use_diversity", True)

    interactions = pd.read_parquet(Path(cfg["paths"]["data_dir"]) / "interactions.parquet")
    test_pos = interactions[(interactions["split"] == "test") & (interactions["label"] == 1)]
    ground_truth: Dict[str, set] = test_pos.groupby("userid")["gameid"].apply(set).to_dict()

    queries = build_vibe_queries(profile_map, ground_truth, n_systems, n_tags,
                                 args.users, seed=args.seed)
    logger.info("[%s] %d queries (%d systems, %d tags each)",
                shape, len(queries), n_systems, n_tags)

    variants = [("post", cap) for cap in CAPS] + [("pre", cap) for cap in CAPS if cap]
    per_query: Dict[Tuple[str, Optional[int]], List[dict]] = {v: [] for v in variants}
    raw_sizes, kept_sizes = [], []

    t_start = time.perf_counter()
    for n, (uid, query_text) in enumerate(queries, start=1):
        raw, tag_kept, ce_scores = score_query(
            query_text, query_encoder, retriever, reranker,
            doc_map, game_info_map, min_retrieval_score,
        )
        if not raw:
            continue
        raw_sizes.append(len(raw))
        kept_sizes.append(len(tag_kept))
        relevant = ground_truth[uid]

        # Reference is the uncapped post-order ranking: today's behaviour.
        pages: Dict[Tuple[str, Optional[int]], List[str]] = {}
        filtered_pages: Dict[Tuple[str, Optional[int]], Dict[str, List[str]]] = {}

        for order, cap in variants:
            scored, n_pairs = rank_for_cap(
                raw, tag_kept, ce_scores, cap, order,
                bayesian_avg_map, rating_weight, min_ce_score,
            )
            page = final_page(scored, game_info_map, None, use_diversity)
            pages[(order, cap)] = page
            filtered_pages[(order, cap)] = {
                name: final_page(scored, game_info_map, flt, use_diversity)
                for name, flt in FILTERS.items()
            }
            per_query[(order, cap)].append({
                "uid": uid,
                "n_pairs": n_pairs,
                "depth": len(page),
                "recall@10": recall_at_k(page, relevant, 10),
                "recall@25": recall_at_k(page, relevant, 25),
                "ndcg@10": ndcg_at_k(page, relevant, 10),
                "ndcg@25": ndcg_at_k(page, relevant, 25),
                "mrr": reciprocal_rank(page[:TOP_K], relevant),
            })

        ref_page = pages[("post", None)]
        ref_filtered = filtered_pages[("post", None)]
        for order, cap in variants:
            rec = per_query[(order, cap)][-1]
            page = pages[(order, cap)]
            rec["overlap@10"] = overlap_at_k(page, ref_page, 10)
            rec["overlap@25"] = overlap_at_k(page, ref_page, TOP_K)
            rec["top1_same"] = float(bool(page) and bool(ref_page) and page[0] == ref_page[0])
            rec["displacement@25"] = rank_displacement(page, ref_page, TOP_K)
            for name in FILTERS:
                fp, rp = filtered_pages[(order, cap)][name], ref_filtered[name]
                rec[f"foverlap@25::{name}"] = overlap_at_k(fp, rp, TOP_K)
                rec[f"fdepth::{name}"] = len(fp)

        if n % 25 == 0:
            rate = n / (time.perf_counter() - t_start)
            logger.info("[%s]   %d/%d queries (%.1f q/s)", shape, n, len(queries), rate)

    return {
        "shape": shape,
        "n_systems": n_systems,
        "n_tags": n_tags,
        "n_queries": len(raw_sizes),
        "raw_pool": _describe(raw_sizes),
        "tag_kept_pool": _describe(kept_sizes),
        "per_query": {f"{order}:{cap}": rows for (order, cap), rows in per_query.items()},
    }


def _describe(values: Sequence[float]) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {}
    return {
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--users", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0,
                        help="torch threads; 0 leaves the default")
    parser.add_argument("--out", default="outputs/vibe_depth.json")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    logger.info("torch threads: %d", torch.get_num_threads())

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    artefacts = recommender.load_artefacts(cfg)

    results = {}
    for shape, (n_systems, n_tags) in SHAPES.items():
        results[shape] = run_shape(shape, n_systems, n_tags, args, cfg, artefacts)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
