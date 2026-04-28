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
from typing import Dict, List

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

def _game_row_cells(gid: str, score: str, meta: pd.DataFrame) -> tuple:
    """Return display cells for one game row (shared between tables)."""
    # Convert Series → plain dict so subsequent .get() calls return scalars.
    row: dict = meta.loc[gid].to_dict() if gid in meta.index else {}
    try:
        rating_str = f"{float(row['bayesian_avg']):.1f} ({int(row['review_count'])})"
    except (KeyError, TypeError, ValueError):
        rating_str = ""
    return (
        score,
        str(row.get("title", gid)),
        str(row.get("author", "")),
        str(row.get("genre", "")),
        str(row.get("system", "")),
        str(row.get("tags", "")),
        rating_str,
    )


def _add_game_columns(table) -> None:
    """Add standard game columns to a Rich table."""
    table.add_column("Score",  style="cyan",       width=8)
    table.add_column("Title",  style="bold white",  min_width=30)
    table.add_column("Author", style="yellow")
    table.add_column("Genre",  style="green")
    table.add_column("System", style="blue")
    table.add_column("Tags",   style="violet")
    table.add_column("Rating", style="magenta",    width=12)


def print_results(results: List[tuple], game_meta: pd.DataFrame, query: str) -> None:
    """Display top-K results in a readable table."""
    meta = game_meta.set_index("gameid")

    if HAS_RICH:
        table = Table(title=f"Results for: [bold]{query}[/bold]", show_lines=True)
        table.add_column("#", style="dim", width=4)
        _add_game_columns(table)
        for rank, (gid, score) in enumerate(results, start=1):
            table.add_row(str(rank), *_game_row_cells(gid, f"{score:.4f}", meta))
        console.print(table)
    else:
        print(f"\nResults for: {query}")
        print("-" * 80)
        for rank, (gid, score) in enumerate(results, start=1):
            row: dict = meta.loc[gid].to_dict() if gid in meta.index else {}
            try:
                rating_str = f"  ★{float(row['bayesian_avg']):.1f}"
            except (KeyError, TypeError, ValueError):
                rating_str = ""
            print(f"  {rank:2d}. [{score:.4f}] {row.get('title', gid)} "
                  f"({row.get('author','')}) — {row.get('genre','')}{rating_str}")
        print()


def print_game_summary(gid: str, game_meta: pd.DataFrame, label: str = "Game") -> None:
    """Print a single game's metadata in the standard table format."""
    meta = game_meta.set_index("gameid")
    if HAS_RICH:
        table = Table(title=f"[bold]{label}[/bold]", show_lines=True)
        _add_game_columns(table)
        table.add_row(*_game_row_cells(gid, "–", meta))
        console.print(table)
    else:
        row: dict = meta.loc[gid].to_dict() if gid in meta.index else {}
        try:
            rating_str = f"  ★{float(row['bayesian_avg']):.1f}"
        except (KeyError, TypeError, ValueError):
            rating_str = ""
        print(f"\n{label}: {row.get('title', gid)} "
              f"({row.get('author','')}) — {row.get('genre','')}{rating_str}")
        print()


def print_user_profile(userid: str, profile_text: str) -> None:
    """Print a user's taste profile in a readable format."""
    # profile_text: "A player who enjoys: Genre: X; System: Y; Tags: a, b, c"
    if HAS_RICH:
        from rich.panel import Panel
        from rich.text import Text
        lines = Text()
        lines.append(f"User: {userid}\n", style="bold cyan")
        prefix = "A player who enjoys: "
        body = profile_text[len(prefix):] if profile_text.startswith(prefix) else profile_text
        for part in body.split("; "):
            if ": " in part:
                key, val = part.split(": ", 1)
                lines.append(f"  {key}: ", style="bold yellow")
                lines.append(f"{val}\n")
            else:
                lines.append(f"  {part}\n")
        console.print(Panel(lines, title="User Profile", border_style="cyan"))
    else:
        print(f"\nUser profile — {userid}")
        print("-" * 60)
        prefix = "A player who enjoys: "
        body = profile_text[len(prefix):] if profile_text.startswith(prefix) else profile_text
        for part in body.split("; "):
            print(f"  {part}")
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

    doc_map     = dict(zip(game_docs["gameid"],     game_docs["doc_text"]))
    profile_map = dict(zip(user_profiles["userid"], user_profiles["profile_text"]))

    retriever = Retriever(
        model=None,
        index=index,
        user_profiles=profile_map,
        game_embeddings=game_embeddings,
    )
    retriever.bi_encoder = bi_encoder

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

    game_meta = game_docs.set_index("gameid")

    while True:
        try:
            query = input("Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        if query_type == "userid":
            emb = retriever._encode_userid(query)
            if emb is None:
                print(f"  No profile found for user '{query}'")
                continue
            profile_text = profile_map.get(query, "")
            print_user_profile(query, profile_text)
            query_text = profile_text

        elif query_type == "game_id":
            if query not in game_meta.index:
                print(f"  Game ID '{query}' not found in index")
                continue
            print_game_summary(query, game_docs, label="Input game")
            emb = retriever._encode_game_ids([query])
            if emb is None:
                print(f"  Could not encode game '{query}'")
                continue
            query_text = doc_map.get(query, query)

        else:  # text
            emb = bi_encoder.encode([query], normalize_embeddings=True)[0]
            query_text = query

        candidates = retriever.index.search(emb, top_k=top_k_ret)

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
    retriever,
    game_docs, doc_map,
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

    ground_truth: Dict[str, set] = (
        test_pos.groupby("userid")["gameid"].apply(set).to_dict()
    )

    logger.info("Running retrieval for %d test users …", len(ground_truth))
    predictions: Dict[str, List[str]] = {}
    n_skipped = 0
    for uid in ground_truth:
        emb = retriever._encode_userid(uid)
        if emb is None:
            n_skipped += 1
            continue
        candidates = retriever.index.search(emb, top_k=top_k_ret)
        predictions[uid] = [gid for gid, _ in candidates]

    if n_skipped:
        logger.warning("%d test users had no profile and were skipped", n_skipped)

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
                        choices=["text", "userid", "game_id"])
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
        run_evaluation(retriever, game_docs, doc_map, cfg)


if __name__ == "__main__":
    main()
