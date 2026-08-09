#!/usr/bin/env python
"""
Check that exp_vibe_depth's derived cap variants match a genuinely capped run.

The experiment scores each query's whole pool once and subsets it per cap, on the
grounds that a cross-encoder score depends only on its own (query, document)
pair. If that is wrong every number in the sweep is wrong, so it is checked
rather than assumed: this runs the real pipeline with a real cap and compares
the resulting page against the derived one, ID for ID.
"""

import importlib.util
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.env import configure_logging
import logging
configure_logging(level=logging.WARNING)

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("recommender", _HERE / "06_run_recommender.py")
recommender = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recommender)

_spec2 = importlib.util.spec_from_file_location("exp", _HERE / "exp_vibe_depth.py")
exp = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(exp)

from src.data.preprocessor import format_profile_text, parse_profile_text
from src.pipeline.retriever import filter_by_tag_overlap

cfg = yaml.safe_load(open("config.yaml"))
(retriever, reranker, query_encoder, game_docs, doc_map, profile_map, name_map,
 bayesian_avg_map, reviews_df, playedgames_df, game_query_text_map,
 game_info_map, *_rest) = recommender.load_artefacts(cfg)

retr = cfg["retrieval"]
min_score = retr.get("min_retrieval_score", 0.25)
min_ce = retr.get("min_rerank_score", 0.10)
rating_w = retr.get("rating_weight", 0.5)
use_div = retr.get("use_diversity", True)

QUERIES = [
    (["twine"], ["horror", "romance"]),
    (["inform"], ["puzzle", "mystery", "fantasy"]),
    ([], ["surreal", "dream"]),
]
CAPS = [50, 200, 500]

ok = True
for systems, tags in QUERIES:
    query_text = format_profile_text(systems, tags)

    # Derived: one full scoring, subset per cap (what the experiment does).
    raw, tag_kept, ce_scores = exp.score_query(
        query_text, query_encoder, retriever, reranker,
        doc_map, game_info_map, min_score,
    )

    for cap in CAPS:
        derived_scored, _ = exp.rank_for_cap(
            raw, tag_kept, ce_scores, cap, "post",
            bayesian_avg_map, rating_w, min_ce,
        )
        derived = exp.final_page(derived_scored, game_info_map, None, use_div)

        # Actual: retrieve, filter, truncate, then rerank only the truncated pool.
        emb = query_encoder.encode([query_text], normalize_embeddings=True,
                                   show_progress_bar=False)[0]
        candidates = retriever.index.search(emb, min_score=min_score)
        _, qtags = parse_profile_text(query_text)
        if qtags:
            candidates = filter_by_tag_overlap(candidates, game_info_map, set(qtags))
        candidates = candidates[:cap]
        actual_scored, _ = reranker.rerank(
            query_text=query_text, candidates=candidates, game_doc_lookup=doc_map,
            top_k=len(candidates), bayesian_avg_map=bayesian_avg_map,
            rating_weight=rating_w, min_ce_score=min_ce,
        )
        actual = exp.final_page(actual_scored, game_info_map, None, use_div)

        same = derived[:25] == actual[:25]
        ok &= same
        print(f"  cap {cap:>4}  {query_text[:42]:<44} "
              f"derived={len(derived):>4} actual={len(actual):>4}  "
              f"top-25 identical: {same}")

print("\nPASS" if ok else "\nFAIL — derived variants do not match a real capped run")
sys.exit(0 if ok else 1)
