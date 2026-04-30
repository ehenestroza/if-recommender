#!/usr/bin/env python
"""
Step 6 – Run the full retrieval + reranking pipeline.

Two modes
---------
interactive  (default) – enter free-text queries or user IDs at the prompt
evaluate               – run offline evaluation on the test split
                         and print Recall@K / NDCG@K / MRR

Usage
-----
    # Interactive demo
    python scripts/06_run_pipeline.py

    # Offline evaluation on test split
    python scripts/06_run_pipeline.py --mode evaluate

    # Query by user ID
    python scripts/06_run_pipeline.py --mode interactive --query-type userid
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

from src.utils.env import configure_logging
configure_logging()

from sentence_transformers import SentenceTransformer, CrossEncoder

from src.index.faiss_index import GameIndex
from src.pipeline.retriever import Retriever
from src.pipeline.ranker import Reranker, evaluate_retrieval, diversify_results
from src.pipeline.retriever import apply_hard_filters, cap_candidates_by_author

logger = logging.getLogger(__name__)

try:
    from rich.console import Console
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
        str(row.get("year", "")),
        str(row.get("system", "")),
        str(row.get("tags", "")),
        rating_str,
    )


def _add_game_columns(table) -> None:
    """Add standard game columns to a Rich table."""
    table.add_column("Score",  style="cyan",       width=8)
    table.add_column("Title",  style="bold white",  min_width=30)
    table.add_column("Author", style="yellow")
    table.add_column("Year",   style="dim",         width=6)
    table.add_column("System", style="blue")
    table.add_column("Tags",   style="violet")
    table.add_column("Rating", style="magenta",    width=12)


def print_results(results: List[tuple], game_meta: pd.DataFrame, query: str) -> None:
    """Display top-K results in a readable table."""
    meta = game_meta.set_index("gameid")

    if HAS_RICH:
        from rich.table import Table
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
            year = row.get("year", "")
            year_str = f" {year}" if year else ""
            print(f"  {rank:2d}. [{score:.4f}] {row.get('title', gid)} "
                  f"({row.get('author','')}){year_str} — {row.get('system','')}{rating_str}")
        print()


def print_game_summary(gid: str, game_meta: pd.DataFrame, label: str = "Game") -> None:
    """Print a single game's metadata in the standard table format."""
    meta = game_meta.set_index("gameid")
    if HAS_RICH:
        from rich.table import Table
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
        year = row.get("year", "")
        year_str = f" {year}" if year else ""
        print(f"\n{label}: {row.get('title', gid)} "
              f"({row.get('author','')}){year_str} — {row.get('system','')}{rating_str}")
        print()


def print_user_profile(userid: str, profile_text: str, user_name: str = "") -> None:
    """Print a user's taste profile in a readable format."""
    display_id = f"{user_name} ({userid})" if user_name else userid
    if HAS_RICH:
        from rich.panel import Panel
        from rich.text import Text
        lines = Text()
        lines.append(f"User: {display_id}\n", style="bold cyan")
        for part in profile_text.split(". "):
            if ": " in part:
                key, val = part.split(": ", 1)
                lines.append(f"  {key}: ", style="bold yellow")
                lines.append(f"{val}\n")
            else:
                lines.append(f"  {part}\n")
        console.print(Panel(lines, title="User Profile", border_style="cyan"))
    else:
        print(f"\nUser profile — {display_id}")
        print("-" * 60)
        for part in profile_text.split(". "):
            print(f"  {part}")
        print()


def print_rated_games(
    rated: List[tuple],
    game_docs: "pd.DataFrame",
) -> None:
    """Print a table of games the user has rated highly (rating, title, author, …)."""
    if not rated:
        return
    meta = game_docs.set_index("gameid")
    if HAS_RICH:
        from rich.table import Table
        table = Table(title="Games you've rated highly", show_lines=True)
        table.add_column("Your ★", style="magenta", width=7)
        table.add_column("Title",  style="bold white", min_width=30)
        table.add_column("Author", style="yellow")
        table.add_column("Year",   style="dim",        width=6)
        table.add_column("System", style="blue")
        table.add_column("Tags",   style="violet")
        for gid, rating in rated:
            row: dict = meta.loc[gid].to_dict() if gid in meta.index else {}
            table.add_row(
                f"★{rating}/5",
                str(row.get("title", gid)),
                str(row.get("author", "")),
                str(row.get("year", "")),
                str(row.get("system", "")),
                str(row.get("tags", "")),
            )
        console.print(table)
    else:
        print("\nGames rated highly:")
        print("-" * 60)
        for gid, rating in rated:
            row = meta.loc[gid].to_dict() if gid in meta.index else {}
            print(f"  ★{rating}/5 — {row.get('title', gid)} ({row.get('author', '')})")
        print()


# ---------------------------------------------------------------------------
# Diversity helpers
# ---------------------------------------------------------------------------

def _parse_profile_targets(text: str):
    """Return (genres: set, systems: set) parsed from a profile/query_text string.

    Genre has been folded into tags, so genres is always empty; only systems are
    extracted here. The return shape is kept for compatibility with diversify_results.
    """
    systems: set = set()
    for part in text.split(". "):
        part = part.strip()
        if part.startswith("Systems:"):
            systems = {s.strip() for s in part[8:].split(",") if s.strip()}
    return set(), systems


def _parse_filters(raw: str) -> dict:
    """Parse a semicolon-separated 'key:value' filter string.

    Recognised keys: year (or year_range), author, system (or sys), tag (or tags).
    Example: 'year:2010-2020; system:inform; tags:fantasy, horror'
    Returns a dict suitable for **kwargs to apply_hard_filters().
    """
    filters: dict = {}
    if not raw.strip():
        return filters
    for part in raw.split(";"):
        part = part.strip()
        if ":" not in part:
            continue
        key, _, val = part.partition(":")
        key = key.strip().lower()
        val = val.strip()
        if not val:
            continue
        if key in ("year", "year_range"):
            filters["year_range"] = val
        elif key == "author":
            filters["author"] = val
        elif key in ("system", "sys"):
            filters["system"] = val
        elif key in ("tag", "tags"):
            filters["tags"] = val
        elif key in ("min_rating", "rating"):
            try:
                filters["min_rating"] = float(val)
            except ValueError:
                pass
        elif key in ("min_count", "min_rating_count", "count"):
            try:
                filters["min_rating_count"] = int(val)
            except ValueError:
                pass
    return filters


# ---------------------------------------------------------------------------
# Load artefacts
# ---------------------------------------------------------------------------

def load_artefacts(cfg: dict):
    data_dir  = Path(cfg["paths"]["data_dir"])
    base_dir  = Path(cfg["paths"]["model_dir"])
    index_dir = Path(cfg["paths"]["index_dir"])
    retr_cfg  = cfg.get("retrieval", {})

    # Prefer asymmetric query_encoder; fall back to legacy two_tower
    if (base_dir / "query_encoder").exists():
        encoder_dir = base_dir / "query_encoder"
    else:
        encoder_dir = base_dir / "two_tower"
    logger.info("Loading query encoder from %s …", encoder_dir)
    query_encoder = SentenceTransformer(str(encoder_dir))
    query_encoder.max_seq_length = cfg["model"]["max_seq_length"]

    logger.info("Loading FAISS index from %s …", index_dir)
    index = GameIndex.load(index_dir)

    logger.info("Loading game embeddings …")
    embs = np.load(index_dir / "game_embs.npy")
    with open(index_dir / "gameid_to_idx.pkl", "rb") as f:
        gameid_to_idx: Dict[str, int] = pickle.load(f)
    game_embeddings = {gid: embs[idx] for gid, idx in gameid_to_idx.items()}

    # Query-space game embeddings for game_id mode (encoded by query encoder, no desc)
    game_query_embeddings: Dict[str, np.ndarray] = {}
    query_embs_path = index_dir / "game_query_embs.npy"
    if query_embs_path.exists():
        q_embs = np.load(query_embs_path)
        game_query_embeddings = {gid: q_embs[idx] for gid, idx in gameid_to_idx.items()}
        logger.info("Loaded game_query_embs.npy (%d entries)", len(game_query_embeddings))

    logger.info("Loading game docs and user profiles …")
    game_docs     = pd.read_parquet(data_dir / "game_docs_retrieval.parquet")
    user_profiles = pd.read_parquet(data_dir / "user_profiles_retrieval.parquet")

    doc_map     = dict(zip(game_docs["gameid"],     game_docs["doc_text"]))
    profile_map = dict(zip(user_profiles["userid"], user_profiles["profile_text"]))
    name_map: Dict[str, str] = (
        dict(zip(user_profiles["userid"], user_profiles["name"]))
        if "name" in user_profiles.columns else {}
    )
    game_query_text_map: Dict[str, str] = (
        dict(zip(game_docs["gameid"], game_docs["query_text"]))
        if "query_text" in game_docs.columns else doc_map
    )
    # author, system, year, tags, rating fields — used by hard filtering and diversity/dedup
    info_cols = ["author", "system", "year", "tags", "bayesian_avg", "review_count"]
    available = [c for c in info_cols if c in game_docs.columns]
    game_info_map: Dict[str, dict] = (
        game_docs.set_index("gameid")[available].to_dict("index")
    )

    retriever = Retriever(
        model=None,
        index=index,
        user_profiles=profile_map,
        game_embeddings=game_embeddings,
        game_query_embeddings=game_query_embeddings,
    )
    retriever.bi_encoder = query_encoder

    # Prefer fine-tuned reranker; fall back to base cross-encoder model
    reranker_dir = base_dir / "reranker"
    if reranker_dir.exists() and any(reranker_dir.iterdir()):
        reranker_model = str(reranker_dir)
        logger.info("Loading fine-tuned reranker from %s …", reranker_dir)
    else:
        reranker_model = cfg["model"]["reranker_model"]
        logger.info("Loading cross-encoder: %s …", reranker_model)
    reranker = Reranker(model_name=reranker_model)

    # Optional bayesian_avg blending in reranker — only for games that have reviews
    bayesian_avg_map = None
    if retr_cfg.get("use_rating_reranking", False):
        reviewed = game_docs[game_docs["review_count"] > 0]
        bayesian_avg_map = dict(zip(reviewed["gameid"], reviewed["bayesian_avg"]))
        logger.info(
            "Rating-weighted reranking enabled (weight=%.2f, %d reviewed games)",
            retr_cfg.get("rating_weight", 0.5), len(bayesian_avg_map),
        )

    # Reviews + playedgames for seen-game filtering and rated-game display
    reviews_df = None
    reviews_path = data_dir / "reviews.parquet"
    if reviews_path.exists():
        reviews_df = pd.read_parquet(reviews_path, columns=["userid", "gameid", "rating"])
        logger.info("Loaded reviews (%d rows) for seen-game filtering", len(reviews_df))

    playedgames_df = None
    played_path = data_dir / "playedgames.parquet"
    if played_path.exists():
        playedgames_df = pd.read_parquet(played_path)
        logger.info("Loaded playedgames (%d rows)", len(playedgames_df))

    return (
        retriever, reranker, query_encoder,
        game_docs, doc_map, profile_map, name_map,
        bayesian_avg_map,
        reviews_df, playedgames_df,
        game_query_text_map,
        game_info_map,
    )


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(
    retriever, reranker, query_encoder,
    game_docs, doc_map, profile_map, name_map,
    cfg, query_type: str = "text",
    bayesian_avg_map=None,
    reviews_df=None,
    playedgames_df=None,
    game_query_text_map=None,
    game_info_map=None,
) -> None:
    retr_cfg        = cfg["retrieval"]
    min_score       = retr_cfg.get("min_retrieval_score", 0.25)
    min_rerank_score = retr_cfg.get("min_rerank_score", 0.25)
    top_k_ret       = retr_cfg["top_k_retrieve"]
    top_k_rank      = retr_cfg["top_k_rerank"]
    rating_w        = retr_cfg.get("rating_weight", 0.5)
    use_diversity   = retr_cfg.get("use_diversity", False)

    if game_query_text_map is None:
        game_query_text_map = doc_map

    print("\n" + "=" * 60)
    print("  IFDB Two-Tower Retrieval Demo")
    print("=" * 60)
    print(f"  Query type       : {query_type}")
    print(f"  Score threshold  : {min_score}  Reranker input: {top_k_ret}  Output: {top_k_rank}")
    print("  Type 'quit' to exit.")
    print("  Filters (optional): year:2010-2020; author:emily short; system:inform; tags:fantasy, horror; rating:3.5; count:10\n")

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

        try:
            filters_raw = input("Filters > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        hard_filters = _parse_filters(filters_raw)

        seen_games: set = set()
        target_genres: set = set()
        target_systems: set = set()

        if query_type == "userid":
            emb = retriever._encode_userid(query)
            if emb is None:
                print(f"  No profile found for user '{query}'")
                continue
            profile_text = profile_map.get(query, "")
            user_name = name_map.get(query, "") if name_map else ""
            print_user_profile(query, profile_text, user_name=user_name)

            # Build seen-games set: all reviewed + all played games for this user
            if reviews_df is not None and "userid" in reviews_df.columns:
                user_revs = reviews_df[reviews_df["userid"] == query]
                seen_games = set(user_revs["gameid"])

                # Show up to 10 games rated >= 4 (actual rating, not Bayesian avg)
                high_rated = (
                    user_revs[user_revs["rating"] >= 4]
                    .sort_values("rating", ascending=False)
                    .head(10)
                )
                if len(high_rated) > 0:
                    rated_pairs = list(zip(
                        high_rated["gameid"],
                        high_rated["rating"].astype(int),
                    ))
                    print_rated_games(rated_pairs, game_docs)

            if playedgames_df is not None and "userid" in playedgames_df.columns:
                seen_games |= set(playedgames_df[playedgames_df["userid"] == query]["gameid"])

            query_text = profile_text
            target_genres, target_systems = _parse_profile_targets(profile_text)

        elif query_type == "game_id":
            if query not in game_meta.index:
                print(f"  Game ID '{query}' not found in index")
                continue
            print_game_summary(query, game_docs, label="Input game")
            emb = retriever._encode_game_ids([query])
            if emb is None:
                print(f"  Could not encode game '{query}'")
                continue
            query_text = game_query_text_map.get(query, doc_map.get(query, query))
            target_genres, target_systems = _parse_profile_targets(query_text)
            target_systems = set(list(target_systems)[:1])

        else:  # text — no structured targets, diversity is a no-op
            emb = query_encoder.encode([query], normalize_embeddings=True)[0]
            query_text = query

        # Step 1: retrieve all candidates above the score threshold
        candidates = retriever.index.search(emb, min_score=min_score)

        # Remove seen/query games
        if query_type == "userid" and seen_games:
            candidates = [(gid, s) for gid, s in candidates if gid not in seen_games]
        elif query_type == "game_id":
            candidates = [(gid, s) for gid, s in candidates if gid != query]

        # Step 2: apply hard filters
        if hard_filters and game_info_map is not None:
            candidates = apply_hard_filters(candidates, game_info_map, **hard_filters)
            logger.info("After hard filters: %d candidates", len(candidates))

        # Cap to 2 games per author (preserving retrieval order) before truncating
        if game_info_map is not None:
            candidates = cap_candidates_by_author(candidates, game_info_map)

        # Step 3: rerank up to top_k_retrieve filtered candidates
        rerank_candidates = candidates[:top_k_ret]
        rerank_top_k = len(rerank_candidates) if (use_diversity and (target_genres or target_systems)) else top_k_rank
        all_scored = reranker.rerank(
            query_text=query_text,
            candidates=rerank_candidates,
            game_doc_lookup=doc_map,
            top_k=rerank_top_k,
            bayesian_avg_map=bayesian_avg_map,
            rating_weight=rating_w,
            min_ce_score=min_rerank_score,
        )

        if use_diversity and (target_genres or target_systems) and game_info_map is not None:
            results = diversify_results(all_scored, game_info_map, target_genres, target_systems, top_k_rank)
        else:
            results = all_scored[:top_k_rank]

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

    (
        retriever, reranker, query_encoder,
        game_docs, doc_map, profile_map, name_map,
        bayesian_avg_map,
        reviews_df, playedgames_df,
        game_query_text_map,
        game_info_map,
    ) = load_artefacts(cfg)

    if args.mode == "interactive":
        run_interactive(
            retriever, reranker, query_encoder,
            game_docs, doc_map, profile_map, name_map,
            cfg, query_type=args.query_type,
            bayesian_avg_map=bayesian_avg_map,
            reviews_df=reviews_df,
            playedgames_df=playedgames_df,
            game_query_text_map=game_query_text_map,
            game_info_map=game_info_map,
        )
    else:
        run_evaluation(retriever, game_docs, doc_map, cfg)


if __name__ == "__main__":
    main()
