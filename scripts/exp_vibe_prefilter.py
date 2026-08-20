#!/usr/bin/env python
"""
Experiment: cheaper ways to shrink the vibe pool than capping it.

A cap truncates a cosine ranking that correlates only weakly with what the
reranker wants (Spearman rho ~ 0.22), so it buys latency by discarding
candidates nearly at random with respect to the final order. Two alternatives
prune on signals the query is *made of*, which is exactly why the existing tag
pre-filter costs nothing:

  1. a higher cosine floor  — `min_retrieval_score`, currently 0.25
  2. a stricter pre-filter  — currently "shares >= 1 tag"; also require a system
                              match, and/or >= 2 tags when the query supplies
                              several

Method mirrors `exp_vibe_depth.py` so the numbers sit beside the cap table: same
held-out users, same menu-style shapes, same metrics. A cross-encoder score
depends only on its own (query, document) pair, so every variant is derived from
one full scoring per query rather than rescored — the comparisons are exact.

Reported per variant:
  n_pairs      what the cross-encoder would actually run: the latency lever
  recall/ndcg  quality against the user's held-out positives
  overlap@25   fidelity against what production serves today

Usage
-----
    uv run scripts/exp_vibe_prefilter.py --users 150 --out outputs/vibe_prefilter.json
"""

import argparse
import importlib.util
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

from src.data.columns import split_clean                      # noqa: E402
from src.data.preprocessor import parse_profile_text          # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s")
logger = logging.getLogger("exp_vibe_prefilter")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_HERE = Path(__file__).resolve().parent
ED = _load("exp_vibe_depth", _HERE / "exp_vibe_depth.py")
REC = _load("recommender", _HERE / "06_run_recommender.py")

# The floor everything is scored at. Higher floors are derived by filtering, so
# this has to be the lowest under test — and it is what production uses.
BASE_FLOOR = 0.25
THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.45)
# Read from config rather than pinned: the reference every variant is judged
# against has to be what production actually serves, or a later run silently
# compares against a cap that was retired.
PRODUCTION_CAP = None  # set from config in main()


def _policies():
    """
    Pre-filter policies: (game systems, game tags, query systems, query tags) -> keep.

    `tags>=1` is what ships. The others tighten it in the two directions that
    prune on something the query actually contains, rather than on cosine rank.
    """
    def t1(gs, gt, qs, qt):
        return not qt or bool(gt & qt)

    def t1_s1(gs, gt, qs, qt):
        return t1(gs, gt, qs, qt) and (not qs or bool(gs & qs))

    def t2_many(gs, gt, qs, qt):
        return not qt or len(gt & qt) >= (2 if len(qt) >= 3 else 1)

    def t2_many_s1(gs, gt, qs, qt):
        return t2_many(gs, gt, qs, qt) and (not qs or bool(gs & qs))

    def t2_all(gs, gt, qs, qt):
        return not qt or len(gt & qt) >= min(2, len(qt))

    return {
        "tags>=1 (today)":       t1,
        "tags>=1 & sys>=1":      t1_s1,
        "tags>=2 if 3+ given":   t2_many,
        "tags>=2 if 3+ & sys>=1": t2_many_s1,
        "tags>=2 always":        t2_all,
    }


def _pool(raw, meta, qs, qt, threshold, keep, cap: Optional[int]) -> List[str]:
    """
    The ids a variant hands the cross-encoder, with the shipped fallbacks.

    An empty pool is not a valid outcome: `filter_by_tag_overlap` returns the
    pool untouched rather than showing nothing, and a variant allowed to empty it
    would post excellent latency by rendering blank pages. The ladder here
    relaxes in the same order the app does.
    """
    above = [(gid, cos) for gid, cos in raw if cos >= threshold]
    if not above:
        above = list(raw[:1])
    kept = [gid for gid, _ in above if keep(meta[gid][0], meta[gid][1], qs, qt)]
    if not kept:
        kept = [gid for gid, _ in above if not qt or (meta[gid][1] & qt)]
    if not kept:
        kept = [gid for gid, _ in above]
    return kept[:cap] if cap else kept


def run_shape(shape, n_systems, n_tags, args, cfg, artefacts) -> dict:
    (retriever, reranker, query_encoder, game_docs, doc_map, profile_map, name_map,
     bayesian_avg_map, reviews_df, playedgames_df, game_query_text_map,
     game_info_map, *_rest) = artefacts

    retr = cfg["retrieval"]
    min_ce_score = retr.get("min_rerank_score", 0.10)
    rating_weight = retr.get("rating_weight", 0.5)
    use_diversity = retr.get("use_diversity", True)

    interactions = pd.read_parquet(Path(cfg["paths"]["data_dir"]) / "interactions.parquet")
    test_pos = interactions[(interactions["split"] == "test") & (interactions["label"] == 1)]
    ground_truth = test_pos.groupby("userid")["gameid"].apply(set).to_dict()

    queries = ED.build_vibe_queries(profile_map, ground_truth, n_systems, n_tags,
                                    args.users, seed=args.seed)
    logger.info("[%s] %d queries (%d systems, %d tags each)",
                shape, len(queries), n_systems, n_tags)

    policies = _policies()
    variants = [(t, p) for t in THRESHOLDS for p in policies]
    per_query: Dict[Tuple[float, str], List[dict]] = {v: [] for v in variants}

    t0 = time.perf_counter()
    for n, (uid, query_text) in enumerate(queries, start=1):
        raw, _tag_kept, ce = ED.score_query(
            query_text, query_encoder, retriever, reranker,
            doc_map, game_info_map, BASE_FLOOR,
        )
        if not raw:
            continue
        qs_list, qt_list = parse_profile_text(query_text)
        qs = {v.strip().lower() for v in qs_list if v.strip()}
        qt = {v.strip().lower() for v in qt_list if v.strip()}
        meta = {gid: (split_clean(game_info_map.get(gid, {}), "system"),
                      split_clean(game_info_map.get(gid, {}), "tags"))
                for gid, _ in raw}

        # What production serves today, the reference every variant is judged against.
        base_pool = _pool(raw, meta, qs, qt, BASE_FLOOR,
                          policies["tags>=1 (today)"], PRODUCTION_CAP)
        base_scored, _ = ED.rank_for_cap(
            [(g, 0.0) for g in base_pool], base_pool, ce, None, "post",
            bayesian_avg_map, rating_weight, min_ce_score)
        base_page = ED.final_page(base_scored, game_info_map, None, use_diversity)

        relevant = ground_truth.get(uid, set())
        for threshold, pname in variants:
            pool = _pool(raw, meta, qs, qt, threshold, policies[pname], PRODUCTION_CAP)
            scored, n_pairs = ED.rank_for_cap(
                [(g, 0.0) for g in pool], pool, ce, None, "post",
                bayesian_avg_map, rating_weight, min_ce_score)
            page = ED.final_page(scored, game_info_map, None, use_diversity)
            per_query[(threshold, pname)].append({
                "uid": uid,
                "n_pairs": n_pairs,
                "recall@10": ED.recall_at_k(page, relevant, 10),
                "recall@25": ED.recall_at_k(page, relevant, 25),
                "ndcg@10": ED.ndcg_at_k(page, relevant, 10),
                "overlap@25": ED.overlap_at_k(page, base_page, 25),
                "top1_same": float(bool(page and base_page and page[0] == base_page[0])),
                "identical": float(page[:25] == base_page[:25]),
            })
        if n % 25 == 0:
            logger.info("[%s] %d/%d queries (%.1fs)", shape, n, len(queries),
                        time.perf_counter() - t0)

    out = {"shape": shape, "n_systems": n_systems, "n_tags": n_tags,
           "n_queries": len(queries), "variants": {}}
    for (threshold, pname), rows in per_query.items():
        if not rows:
            continue
        out["variants"][f"{threshold}|{pname}"] = {
            "threshold": threshold, "policy": pname,
            **{m: float(np.mean([r[m] for r in rows]))
               for m in ("n_pairs", "recall@10", "recall@25", "ndcg@10",
                         "overlap@25", "top1_same", "identical")},
            "n_pairs_p90": float(np.percentile([r["n_pairs"] for r in rows], 90)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--users", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/vibe_prefilter.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    global PRODUCTION_CAP
    PRODUCTION_CAP = cfg["retrieval"].get("rerank_pool_cap") or None
    artefacts = REC.load_artefacts(cfg)
    results = {s: run_shape(s, ns, nt, args, cfg, artefacts)
               for s, (ns, nt) in ED.SHAPES.items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"))
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
