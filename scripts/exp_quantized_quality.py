#!/usr/bin/env python
"""
Experiment: what does int8 dynamic quantization cost the vibe ranking?

Splits the question in two, because the halves do not travel together:

  speed    a property of the host's quantized backend. fbgemm (x86) and qnnpack
           (ARM) differ by more than the effect being measured, so a laptop
           number is worthless — use `measure_latency.py --quantize` on the
           target machine instead. Nothing here times anything.

  quality  a property of the weights, and portable enough to measure anywhere.
           That is what this script does.

Both models score the *same* candidate pools, so every comparison is paired at
the query level: any difference is the quantization and nothing else.

Usage
-----
    uv run scripts/exp_quantized_quality.py --users 150
"""

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.env import configure_logging
configure_logging()

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("recommender", _HERE / "06_run_recommender.py")
recommender = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recommender)

_spec2 = importlib.util.spec_from_file_location("exp", _HERE / "exp_vibe_depth.py")
exp = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(exp)

from src.pipeline.quantize import quantize_cross_encoder

logger = logging.getLogger("exp_quantized_quality")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--users", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shape", default="narrow", choices=list(exp.SHAPES))
    parser.add_argument("--cap", type=int, default=0,
                        help="Pool cap to evaluate under; 0 = full pool")
    parser.add_argument("--out", default="outputs/quantized_quality.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    (retriever, reranker, query_encoder, game_docs, doc_map, profile_map, name_map,
     bayesian_avg_map, reviews_df, playedgames_df, game_query_text_map,
     game_info_map, *_rest) = recommender.load_artefacts(cfg)

    # Quantization is CPU-only; comparing against an MPS/CUDA fp32 baseline would
    # fold a device difference into the measurement.
    reranker.model.model.to("cpu")
    query_encoder.to("cpu")

    retr = cfg["retrieval"]
    min_retrieval_score = retr.get("min_retrieval_score", 0.25)
    min_ce = retr.get("min_rerank_score", 0.10)
    rating_w = retr.get("rating_weight", 0.5)
    use_div = retr.get("use_diversity", True)
    cap = args.cap or None

    interactions = pd.read_parquet(Path(cfg["paths"]["data_dir"]) / "interactions.parquet")
    test_pos = interactions[(interactions["split"] == "test") & (interactions["label"] == 1)]
    ground_truth: Dict[str, set] = test_pos.groupby("userid")["gameid"].apply(set).to_dict()

    n_systems, n_tags = exp.SHAPES[args.shape]
    queries = exp.build_vibe_queries(profile_map, ground_truth, n_systems, n_tags,
                                     args.users, seed=args.seed)
    logger.info("%d queries, shape=%s, cap=%s", len(queries), args.shape, cap or "full")

    # fp32 first: quantization is destructive, so the baseline has to be taken
    # before the model is converted, and both passes reuse the same pools.
    logger.info("Scoring fp32 baseline …")
    pools = []
    for n, (uid, query_text) in enumerate(queries, start=1):
        raw, tag_kept, ce = exp.score_query(
            query_text, query_encoder, retriever, reranker,
            doc_map, game_info_map, min_retrieval_score)
        if raw:
            pools.append({"uid": uid, "query": query_text, "raw": raw,
                          "tag_kept": tag_kept, "ce_fp32": ce})
        if n % 25 == 0:
            logger.info("  fp32 %d/%d", n, len(queries))

    logger.info("Quantizing …")
    if not quantize_cross_encoder(reranker.model):
        logger.error("No quantized backend on this host — cannot run")
        sys.exit(1)

    logger.info("Scoring int8 …")
    for n, pool in enumerate(pools, start=1):
        pairs = [(pool["query"], doc_map.get(gid, "")) for gid, _ in pool["raw"]]
        logits = reranker.model.predict(pairs, show_progress_bar=False)
        probs = torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).numpy()
        pool["ce_int8"] = {gid: float(p) for (gid, _), p in zip(pool["raw"], probs)}
        if n % 25 == 0:
            logger.info("  int8 %d/%d", n, len(pools))

    rows: List[dict] = []
    score_deltas: List[float] = []
    for pool in pools:
        relevant = ground_truth[pool["uid"]]
        pages = {}
        for tag, key in (("fp32", "ce_fp32"), ("int8", "ce_int8")):
            scored, _ = exp.rank_for_cap(
                pool["raw"], pool["tag_kept"], pool[key], cap, "post",
                bayesian_avg_map, rating_w, min_ce)
            pages[tag] = exp.final_page(scored, game_info_map, None, use_div)

        common = set(pool["ce_fp32"]) & set(pool["ce_int8"])
        score_deltas.extend(abs(pool["ce_fp32"][g] - pool["ce_int8"][g]) for g in common)

        row = {"uid": pool["uid"]}
        for tag in ("fp32", "int8"):
            page = pages[tag]
            row[f"{tag}_recall@10"] = exp.recall_at_k(page, relevant, 10)
            row[f"{tag}_recall@25"] = exp.recall_at_k(page, relevant, 25)
            row[f"{tag}_ndcg@10"] = exp.ndcg_at_k(page, relevant, 10)
            row[f"{tag}_ndcg@25"] = exp.ndcg_at_k(page, relevant, 25)
            row[f"{tag}_mrr"] = exp.reciprocal_rank(page[:exp.TOP_K], relevant)
        row["overlap@10"] = exp.overlap_at_k(pages["int8"], pages["fp32"], 10)
        row["overlap@25"] = exp.overlap_at_k(pages["int8"], pages["fp32"], exp.TOP_K)
        row["top1_same"] = float(bool(pages["fp32"]) and bool(pages["int8"])
                                 and pages["fp32"][0] == pages["int8"][0])
        row["displacement@25"] = exp.rank_displacement(pages["int8"], pages["fp32"], exp.TOP_K)
        rows.append(row)

    df = pd.DataFrame(rows)
    deltas = np.asarray(score_deltas)

    print("\n" + "=" * 82)
    print(f"  int8 dynamic quantization — quality  (n = {len(df)} queries, "
          f"shape={args.shape}, cap={cap or 'full'})")
    print(f"  engine: {torch.backends.quantized.engine}")
    print("=" * 82)

    print("\n  RELEVANCE SCORES  (per candidate, absolute difference)")
    print(f"    mean |Δ| {deltas.mean():.4f}   p99 |Δ| {np.percentile(deltas, 99):.4f}   "
          f"max |Δ| {deltas.max():.4f}   over {len(deltas):,} pairs")

    print("\n  QUALITY vs held-out positives")
    print(f"  {'metric':<12} {'fp32':>9} {'int8':>9}   {'Δ [95% CI]':>28}")
    print("  " + "-" * 62)
    for metric in ("recall@10", "recall@25", "ndcg@10", "ndcg@25", "mrr"):
        a, b = df[f"int8_{metric}"], df[f"fp32_{metric}"]
        d, lo, hi = exp.paired_bootstrap(a, b)
        flag = "" if lo <= 0 <= hi else "  *"
        print(f"  {metric:<12} {b.mean():>9.4f} {a.mean():>9.4f}   "
              f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]{flag}")
    print("  * = interval excludes zero")

    print("\n  AGREEMENT with the fp32 page")
    print(f"    overlap@10 {df['overlap@10'].mean():.3f}   "
          f"overlap@25 {df['overlap@25'].mean():.3f}   "
          f"top-1 same {df['top1_same'].mean():.3f}")
    print(f"    mean |Δrank| {df['displacement@25'].mean():.2f}   "
          f"identical pages {(df['overlap@25'] >= 0.999).mean() * 100:.1f}%")
    print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"shape": args.shape, "cap": cap,
                   "engine": torch.backends.quantized.engine,
                   "score_delta_mean": float(deltas.mean()),
                   "score_delta_max": float(deltas.max()),
                   "rows": rows}, f)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
