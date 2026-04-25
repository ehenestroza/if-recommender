#!/usr/bin/env python
"""
Step 5 – Run the full retrieval + reranking pipeline.

Two modes
---------
interactive  (default) – enter free-text queries or user IDs at the prompt
evaluate               – run offline evaluation on the test split
                         and print Recall@K / NDCG@K / MRR

Usage
-----
    # Interactive demo
    python scripts/05_run_pipeline.py

    # Offline evaluation on test split
    python scripts/05_run_pipeline.py --mode evaluate

    # Query by user ID
    python scripts/05_run_pipeline.py --mode interactive --query-type userid
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentence_transformers import SentenceTransformer, CrossEncoder

from src.index.faiss_index import GameIndex
from src.pipeline.retriever import Retriever
from src.pipeline.ranker import Reranker, evaluate_retrieval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_results(results: List[tuple], game_meta: pd.DataFrame, query: str) -> None:
    """Display top-K results in a readable table."""
    meta = game_meta.set_index("gameid")

    if HAS_RICH:
        table = Table(title=f"Results for: [bold]{query}[/bold]", show_lines=True)
        table.add_column("#",      style="dim", width=4)
        table.add_column("Score",  style="cyan", width=8)
        table.add_column("Title",  style="bold white", min_width=30)
        table.add_column("Author", style="yellow")
        table.add_column("Genre",  style="green")
        for rank, (gid, score) in enumerate(results, start=1):
            row = meta.loc[gid] if gid in meta.index else {}
            table.add_row(
                str(rank),
                f"{score:.4f}",
                str(row.get("title", gid)),
                str(row.get("author", "")),
                str(row.get("genre", "")),
            )
        console.print(table)
    else:
        print(f"\nResults for: {query}")
        print("-" * 80)
        for rank, (gid, score) in enumerate(results, start=1):
            row = meta.loc[gid] if gid in meta.index else {}
            print(f"  {rank:2d}. [{score:.4f}] {row.get('title', gid)} "
                  f"({row.get('author','')}) — {row.get('genre','')}")
        print()


# ---------------------------------------------------------------------------
# Load artefacts
# ---------------------------------------------------------------------------

def load_artefacts(cfg: dict):
    data_dir  = Path(cfg["paths"]["data_dir"])
    model_dir = Path(cfg["paths"]["model_dir"]) / "two_tower"
    index_dir = Path(cfg["paths"]["index_dir"])

    logger.info("Loading model from %s …", model_dir)
    bi_encoder = SentenceTransformer(str(model_dir))
    bi_encoder.max_seq_length = cfg["model"]["max_seq_length"]

    logger.info("Loading FAISS index from %s …", index_dir)
    index = GameIndex.load(index_dir)

    logger.info("Loading game embeddings …")
    embs = np.load(index_dir / "game_embs.npy")
    with open(index_dir / "gameid_to_idx.pkl", "rb") as f:
        gameid_to_idx: Dict[str, int] = pickle.load(f)
    game_embeddings = {gid: embs[idx] for gid, idx in gameid_to_idx.items()}

    logger.info("Loading game docs and user profiles …")
    game_docs     = pd.read_parquet(data_dir / "game_docs.parquet")
    user_profiles = pd.read_parquet(data_dir / "user_profiles.parquet")

    doc_map     = dict(zip(game_docs["gameid"],    game_docs["doc_text"]))
    profile_map = dict(zip(user_profiles["userid"], user_profiles["profile_text"]))

    # Wrap in our Retriever
    retriever = Retriever(
        model=None,  # we'll call bi_encoder directly below (simpler for demo)
        index=index,
        user_profiles=profile_map,
        game_embeddings=game_embeddings,
    )
    # Patch the raw SentenceTransformer encode into the retriever's model object
    retriever.bi_encoder = bi_encoder

    # Cross-encoder reranker
    logger.info("Loading cross-encoder: %s …", cfg["model"]["reranker_model"])
    reranker = Reranker(model_name=cfg["model"]["reranker_model"])

    return retriever, reranker, bi_encoder, game_docs, doc_map, profile_map


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(
    retriever, reranker, bi_encoder,
    game_docs, doc_map, profile_map,
    cfg, query_type: str = "text",
) -> None:
    retr_cfg   = cfg["retrieval"]
    top_k_ret  = retr_cfg["top_k_retrieve"]
    top_k_rank = retr_cfg["top_k_rerank"]

    print("\n" + "=" * 60)
    print("  IFDB Two-Tower Retrieval Demo")
    print("=" * 60)
    print(f"  Query type  : {query_type}")
    print(f"  Retrieve top: {top_k_ret}  Rerank top: {top_k_rank}")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        # Encode query
        if query_type == "userid":
            profile = profile_map.get(query)
            if not profile:
                print(f"  ⚠ No profile found for user '{query}'")
                continue
            query_text = profile
        else:
            query_text = query

        emb = bi_encoder.encode([query_text], normalize_embeddings=True)[0]

        # Retrieve
        candidates = retriever.index.search(emb, top_k=top_k_ret)

        # Rerank
        results = reranker.rerank(
            query_text=query_text,
            candidates=candidates,
            game_doc_lookup=doc_map,
            top_k=top_k_rank,
        )

        print_results(results, game_docs, query)


# ---------------------------------------------------------------------------
# Evaluation mode
# ---------------------------------------------------------------------------

def run_evaluation(
    retriever, bi_encoder,
    game_docs, doc_map, profile_map,
    cfg,
) -> None:
    data_dir   = Path(cfg["paths"]["data_dir"])
    retr_cfg   = cfg["retrieval"]
    top_k_ret  = retr_cfg["top_k_retrieve"]

    logger.info("Loading test interactions …")
    interactions = pd.read_parquet(data_dir / "interactions.parquet")
    test_pos = interactions[
        (interactions["split"] == "test") & (interactions["label"] == 1)
    ]

    # Ground truth: {userid → {gameid, …}}
    ground_truth: Dict[str, set] = (
        test_pos.groupby("userid")["gameid"].apply(set).to_dict()
    )

    # Build predictions for each test user
    logger.info("Running retrieval for %d test users …", len(ground_truth))
    predictions: Dict[str, List[str]] = {}
    for uid in ground_truth:
        profile = profile_map.get(uid)
        if not profile:
            continue
        emb = bi_encoder.encode([profile], normalize_embeddings=True)[0]
        candidates = retriever.index.search(emb, top_k=top_k_ret)
        predictions[uid] = [gid for gid, _ in candidates]

    # Metrics
    results = evaluate_retrieval(
        predictions=predictions,
        ground_truth=ground_truth,
        ks=(1, 5, 10, 20, 50),
    )

    print("\n" + "=" * 50)
    print("  Evaluation results (test split)")
    print("=" * 50)
    for metric, value in sorted(results.items()):
        print(f"  {metric:<15}  {value:.4f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--mode",       default="interactive",
                        choices=["interactive", "evaluate"])
    parser.add_argument("--query-type", default="text",
                        choices=["text", "userid", "game_ids"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    retriever, reranker, bi_encoder, game_docs, doc_map, profile_map = \
        load_artefacts(cfg)

    if args.mode == "interactive":
        run_interactive(
            retriever, reranker, bi_encoder,
            game_docs, doc_map, profile_map,
            cfg, query_type=args.query_type,
        )
    else:
        run_evaluation(retriever, bi_encoder, game_docs, doc_map, profile_map, cfg)


if __name__ == "__main__":
    main()
