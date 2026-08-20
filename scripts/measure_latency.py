#!/usr/bin/env python
"""
Measure how long a live `vibe` query takes, stage by stage.

Written to answer the deployment question `measure_memory.py` does not: `vibe`
is the only mode scored live, and on a small VM it is the whole of the request
latency. The number that governs it is cross-encoder throughput in pairs per
second, which varies by more than an order of magnitude between a laptop and a
burstable cloud vCPU — so this has to be run on the target machine to mean
anything.

Reports throughput separately from end-to-end query time, because the two answer
different questions: throughput sizes the candidate pool you can afford, while
end-to-end tells you what a visitor actually waits.

Usage
-----
    uv run scripts/measure_latency.py                  # ~2 min
    uv run scripts/measure_latency.py --json
    uv run scripts/measure_latency.py --pool-caps 200,500,1000
"""

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.env import configure_logging
configure_logging(level=logging.WARNING)

_spec = importlib.util.spec_from_file_location(
    "recommender", Path(__file__).resolve().parent / "06_run_recommender.py"
)
recommender = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recommender)

from src.data.preprocessor import format_profile_text, parse_profile_text
from src.pipeline.ranker import select_results
from src.pipeline.retriever import filter_by_tag_overlap

# Representative of what the pickers offer: a small pool, a typical one, and one
# of the large ones a single broad system pick produces.
QUERIES = [
    (["choicescript"], ["romance"]),
    (["inform"], ["puzzle", "mystery"]),
    (["twine"], ["horror", "romance"]),
    (["twine"], ["fantasy"]),
    (["twine", "inform"], ["horror", "surreal", "fantasy"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure live vibe query latency")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    parser.add_argument("--pool-caps", default="",
                        help="Comma-separated caps to project, e.g. 200,500,1000")
    parser.add_argument("--device", default="",
                        help="Force a torch device. The deployment has no GPU, so "
                             "'cpu' is what reproduces it from a laptop that would "
                             "otherwise pick MPS or CUDA")
    parser.add_argument("--threads", type=int, default=0,
                        help="torch threads; 0 leaves the default (one per vCPU)")
    parser.add_argument("--no-cap", action="store_true",
                        help="Ignore retrieval.rerank_pool_cap and score the whole "
                             "pool, to measure what the cap is buying")
    parser.add_argument("--quantize", action="store_true",
                        help="Dynamically quantize the cross-encoder to int8 before "
                             "measuring. Whether this is faster depends on the host's "
                             "quantized backend, which is the point of measuring it here")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    (retriever, reranker, query_encoder, game_docs, doc_map, profile_map, name_map,
     bayesian_avg_map, reviews_df, playedgames_df, game_query_text_map,
     game_info_map, *_rest) = recommender.load_artefacts(cfg)

    if args.device:
        reranker.model.model.to(args.device)
        query_encoder.to(args.device)

    if args.quantize:
        from src.pipeline.quantize import quantize_cross_encoder
        quantize_cross_encoder(reranker.model)

    # Report what the model *is*, not what was asked for. `load_artefacts`
    # already applies `model.quantize_reranker` from config, so a run without
    # --quantize can still be measuring a quantized model — and reporting the
    # flag instead of the fact makes an int8 run look like an fp32 one.
    import torch.ao.nn.quantized.dynamic as nnqd
    n_int8 = sum(1 for m in reranker.model[0].auto_model.modules()
                 if isinstance(m, nnqd.Linear))
    quantized = n_int8 > 0

    retr = cfg["retrieval"]
    min_score = retr.get("min_retrieval_score", 0.25)
    min_ce = retr.get("min_rerank_score", 0.10)
    rating_w = retr.get("rating_weight", 0.5)
    pool_cap = 0 if args.no_cap else retr.get("rerank_pool_cap", 0)

    report: dict = {
        "threads": torch.get_num_threads(),
        "device": str(reranker.model.model.device),
        "quantized": quantized,
        "quantized_engine": torch.backends.quantized.engine if quantized else None,
        "pool_cap": pool_cap,
        "queries": [],
    }

    # Warm up: the first predict pays for lazy kernel setup, which would
    # otherwise land entirely on the first query and overstate it.
    warm = [(format_profile_text(["twine"], ["horror"]), doc) for doc in list(doc_map.values())[:32]]
    reranker.model.predict(warm, show_progress_bar=False)

    for systems, tags in QUERIES:
        query_text = format_profile_text(systems, tags)

        t0 = time.perf_counter()
        emb = query_encoder.encode([query_text], normalize_embeddings=True,
                                   show_progress_bar=False)[0]
        t_encode = time.perf_counter() - t0

        t0 = time.perf_counter()
        candidates = retriever.index.search(emb, min_score=min_score)
        t_search = time.perf_counter() - t0

        raw_n = len(candidates)
        _, query_tags = parse_profile_text(query_text)
        if retr.get("prefilter_by_tag", True) and query_tags:
            candidates = filter_by_tag_overlap(
                candidates, game_info_map, set(query_tags),
                min_matches=retr.get("prefilter_tag_matches", 1),
                min_matches_from=retr.get("prefilter_tag_matches_from"),
            )

        # Same order as both front-ends: cap after the tag pre-filter, so the cap
        # spends its budget on candidates that survived it.
        if pool_cap and len(candidates) > pool_cap:
            candidates = candidates[:pool_cap]

        t0 = time.perf_counter()
        scored, relevance = reranker.rerank(
            query_text=query_text,
            candidates=candidates,
            game_doc_lookup=doc_map,
            top_k=len(candidates),
            bayesian_avg_map=bayesian_avg_map,
            rating_weight=rating_w,
            min_ce_score=min_ce,
        )
        t_rerank = time.perf_counter() - t0

        t0 = time.perf_counter()
        results = select_results(scored, None, game_info_map, len(scored),
                                 use_diversity=retr.get("use_diversity", True),
                                 target_genres=set(), target_systems=set())
        t_select = time.perf_counter() - t0

        total = t_encode + t_search + t_rerank + t_select
        report["queries"].append({
            "query": query_text,
            "raw_pool": raw_n,
            "scored_pairs": len(candidates),
            "results": len(results),
            "encode_s": t_encode,
            "search_s": t_search,
            "rerank_s": t_rerank,
            "select_s": t_select,
            "total_s": total,
            "pairs_per_s": len(candidates) / t_rerank if t_rerank > 0 else 0.0,
        })

    rates = [q["pairs_per_s"] for q in report["queries"]]
    report["pairs_per_s_median"] = float(np.median(rates))

    caps = [int(c) for c in args.pool_caps.split(",") if c.strip()]
    if caps:
        report["projected"] = {
            str(cap): cap / report["pairs_per_s_median"] for cap in caps
        }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("\n" + "=" * 78)
    print(f"  Live vibe latency — {report['threads']} torch threads, device {report['device']}")
    if quantized:
        print(f"  cross-encoder: dynamic int8 ({report['quantized_engine']})")
    print(f"  pool cap: {pool_cap or 'none — scoring the whole pool'}")
    print("=" * 78)
    print(f"  {'query':<44} {'pairs':>6} {'rerank':>8} {'total':>8}")
    print("  " + "-" * 70)
    for q in report["queries"]:
        print(f"  {q['query'][:44]:<44} {q['scored_pairs']:>6} "
              f"{q['rerank_s']:>7.2f}s {q['total_s']:>7.2f}s")
    print("  " + "-" * 70)
    print(f"  cross-encoder throughput: {report['pairs_per_s_median']:.0f} pairs/s (median)")
    print("  everything outside the reranker — encoding, FAISS, filtering, diversity —")
    slack = sum(q["total_s"] - q["rerank_s"] for q in report["queries"]) / len(report["queries"])
    print(f"  costs {slack * 1000:.0f} ms per query combined.")

    if caps:
        print("\n  projected rerank time at a capped pool:")
        for cap, seconds in report["projected"].items():
            print(f"    top {int(cap):>5,} candidates   {seconds:>6.2f}s")
    print()


if __name__ == "__main__":
    main()
