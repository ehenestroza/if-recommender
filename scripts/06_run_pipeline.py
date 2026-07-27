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
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.env import configure_logging
configure_logging()

from sentence_transformers import SentenceTransformer, CrossEncoder

from src.index.faiss_index import GameIndex
from src.pipeline.retriever import Retriever
from src.data.preprocessor import parse_profile_text
from src.pipeline.ranker import Reranker, evaluate_retrieval, diversify_results
from src.pipeline.retriever import apply_hard_filters, filter_by_tag_overlap

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

# Shown instead of a score when nobody has rated a game.
NO_RATING = "—"


def _rating_cell(row: dict) -> str:
    """
    Rating column text: "3.8 (2)", or an em dash when the game has no ratings.

    Shows the raw community average. `bayesian_avg` is a ranking signal — it
    shrinks toward a 3.5 prior — so displaying it would report a score nobody
    actually gave, and for unrated games would be the prior alone.
    """
    try:
        count = int(row["review_count"])
        if count == 0:
            return NO_RATING
        value = float(row["avg_rating"])
        return "" if pd.isna(value) else f"{value:.1f} ({count})"
    except (KeyError, TypeError, ValueError):
        return ""


def _rating_suffix(row: dict) -> str:
    """Same distinction as `_rating_cell`, for the plain-text (no-Rich) output."""
    cell = _rating_cell(row)
    if not cell:
        return ""
    return "  (unrated)" if cell == NO_RATING else f"  ★{cell.split(' ')[0]}"


def _game_row_cells(
    gid: str, score: str, meta: pd.DataFrame, relevance: Optional[dict] = None
) -> tuple:
    """Return display cells for one game row (shared between tables)."""
    # Convert Series → plain dict so subsequent .get() calls return scalars.
    row: dict = meta.loc[gid].to_dict() if gid in meta.index else {}
    rel = relevance.get(gid) if relevance else None
    return (
        score,
        str(row.get("title", gid)),
        str(row.get("author", "")),
        str(row.get("year", "")),
        str(row.get("system", "")),
        str(row.get("tags", "")),
        "–" if rel is None else f"{rel:.2f}",
        _rating_cell(row),
    )


def _add_game_columns(table) -> None:
    """Add standard game columns to a Rich table."""
    table.add_column("Score",  style="cyan",       width=8)
    table.add_column("Title",  style="bold white",  min_width=30)
    table.add_column("Author", style="yellow")
    table.add_column("Year",   style="dim",         width=6)
    table.add_column("System", style="blue")
    table.add_column("Tags",   style="violet")
    # The two components of Score, shown so their trade-off is visible.
    table.add_column("Relev.", style="green",      width=6)
    table.add_column("Rating", style="magenta",    width=10)


def print_results(
    results: List[tuple],
    game_meta: pd.DataFrame,
    query: str,
    relevance: Optional[dict] = None,
) -> None:
    """Display top-K results in a readable table."""
    meta = game_meta.set_index("gameid")

    if HAS_RICH:
        from rich.table import Table
        table = Table(title=f"Results for: [bold]{query}[/bold]", show_lines=True)
        table.add_column("#", style="dim", width=4)
        _add_game_columns(table)
        for rank, (gid, score) in enumerate(results, start=1):
            table.add_row(str(rank), *_game_row_cells(gid, f"{score:.4f}", meta, relevance))
        console.print(table)
    else:
        print(f"\nResults for: {query}")
        print("-" * 80)
        for rank, (gid, score) in enumerate(results, start=1):
            row: dict = meta.loc[gid].to_dict() if gid in meta.index else {}
            rating_str = _rating_suffix(row)
            year = row.get("year", "")
            year_str = f" {year}" if year else ""
            rel = relevance.get(gid) if relevance else None
            rel_str = f" rel {rel:.2f}" if rel is not None else ""
            print(f"  {rank:2d}. [{score:.4f}]{rel_str} {row.get('title', gid)} "
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
        rating_str = _rating_suffix(row)
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


def load_precomputed(path: Path, key_col: str) -> dict:
    """
    Load a precomputed ranking file into {key: (gameids, scores, relevances)}.

    Returns {} when the file is absent, so the pipeline falls back to scoring
    live. Arrays rather than tuples keep 1.6M rows to a manageable footprint.
    """
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    table = {
        key: (grp["gameid"].tolist(), grp["score"].to_numpy(), grp["relevance"].to_numpy())
        for key, grp in frame.groupby(key_col, sort=False)
    }
    logger.info("Loaded %s — %d keys", path.name, len(table))
    return table


FILTER_HELP = """  Refine these results with filters, e.g.  year:2020-2026; rating:3.5; count:2
  Keys: year, author, system, tags, rating, count   (each entry replaces the last)
  'clear' show all again  ·  'back' new query  ·  'quit' exit"""


def show_results(
    scored: List[tuple],
    hard_filters: dict,
    *,
    game_docs: pd.DataFrame,
    game_info_map: Optional[dict],
    top_k: int,
    use_diversity: bool,
    target_genres: set,
    target_systems: set,
    label: str,
    relevance: Optional[dict] = None,
) -> None:
    """
    Filter, diversify, and print a slice of an already-scored candidate list.

    `scored` is the whole reranked pool for one query, so filtering here strictly
    narrows what the user is already looking at rather than changing which
    candidates ever reached the reranker.
    """
    results = scored
    if hard_filters and game_info_map is not None:
        results = apply_hard_filters(results, game_info_map, **hard_filters)
        logger.info("Filters kept %d of %d scored candidates", len(results), len(scored))

    if game_info_map is not None:
        # diversify_results caps repeat authors even with no coverage targets,
        # which is why no separate author-cap pass is needed before reranking.
        genres = target_genres if use_diversity else set()
        systems = target_systems if use_diversity else set()
        results = diversify_results(results, game_info_map, genres, systems, top_k)
    else:
        results = results[:top_k]

    if not results:
        print("  Nothing matches those filters. Try relaxing them, or 'clear'.\n")
        return
    print_results(results, game_docs, label, relevance=relevance)


# ---------------------------------------------------------------------------
# Load artefacts
# ---------------------------------------------------------------------------

def load_artefacts(cfg: dict):
    data_dir  = Path(cfg["paths"]["data_dir"])
    base_dir  = Path(cfg["paths"]["model_dir"])
    index_dir = Path(cfg["paths"]["index_dir"])
    retr_cfg  = cfg.get("retrieval", {})

    encoder_dir = base_dir / "query_encoder"
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
    # Fields used by hard filtering and diversity/dedup. The `_clean` variants are
    # what those consumers read; the originals are listed as a fallback for
    # game_docs files written before the clean/original split. Display reads
    # game_docs directly, so it always shows the IFDB originals.
    info_cols = ["author", "author_clean", "system", "system_clean",
                 "tags", "tags_clean", "genre", "year",
                 "avg_rating", "bayesian_avg", "review_count"]
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

    # Precomputed rankings for the enumerable query modes; absent files simply
    # mean those modes fall back to scoring live.
    precomputed_user = load_precomputed(data_dir / "precomputed_userid.parquet", "userid")
    precomputed_game = load_precomputed(data_dir / "precomputed_gameid.parquet", "seed_gameid")

    return (
        retriever, reranker, query_encoder,
        game_docs, doc_map, profile_map, name_map,
        bayesian_avg_map,
        reviews_df, playedgames_df,
        game_query_text_map,
        game_info_map,
        precomputed_user, precomputed_game,
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
    precomputed_user=None,
    precomputed_game=None,
) -> None:
    precomputed_user = precomputed_user or {}
    precomputed_game = precomputed_game or {}
    retr_cfg        = cfg["retrieval"]
    min_score       = retr_cfg.get("min_retrieval_score", 0.25)
    min_rerank_score = retr_cfg.get("min_rerank_score", 0.25)
    top_k_rank      = retr_cfg["top_k_rerank"]
    pool_cap        = retr_cfg.get("rerank_pool_cap", 0)
    prefilter_tags  = retr_cfg.get("prefilter_by_tag", True)
    rating_w        = retr_cfg.get("rating_weight", 0.5)
    use_diversity   = retr_cfg.get("use_diversity", False)

    if game_query_text_map is None:
        game_query_text_map = doc_map

    print("\n" + "=" * 60)
    print("  IFDB Retrieval Demo")
    print("=" * 60)
    print(f"  Query type       : {query_type}")
    print(f"  Score threshold  : {min_score}   Output: {top_k_rank}")
    print("  Enter a query to see results, then refine them with filters.")
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

        # userid and game_id draw from a fixed key set, so their rankings are
        # precomputed offline (scripts/07_precompute.py) and served as a lookup.
        cached = None
        if query_type == "userid":
            cached = precomputed_user.get(query)
        elif query_type == "game_id":
            cached = precomputed_game.get(query)

        if cached is not None:
            gids, scores, rels = cached
            scored = list(zip(gids, scores))
            relevance = dict(zip(gids, rels))
            logger.info("Precomputed ranking (%d entries) — no scoring needed", len(gids))
        else:
            # Retrieve every candidate above the cosine threshold, then score the
            # whole pool with the cross-encoder exactly once. Cosine rank and
            # cross-encoder rank correlate only weakly, so truncating before
            # reranking would hide most of what the reranker would have chosen —
            # and would make a filter change which candidates get scored at all.
            candidates = retriever.index.search(emb, min_score=min_score)

            # Remove seen/query games
            if query_type == "userid" and seen_games:
                candidates = [(gid, s) for gid, s in candidates if gid not in seen_games]
            elif query_type == "game_id":
                candidates = [(gid, s) for gid, s in candidates if gid != query]

            if not candidates:
                print("  No candidates above the retrieval threshold.\n")
                continue

            # Drop candidates sharing no tag with the query. Measured free, and
            # far better targeted than truncating by cosine rank.
            if prefilter_tags and game_info_map is not None:
                _, query_tags = parse_profile_text(query_text)
                if query_tags:
                    before = len(candidates)
                    candidates = filter_by_tag_overlap(
                        candidates, game_info_map, set(query_tags)
                    )
                    if len(candidates) < before:
                        logger.info("Tag pre-filter: %d → %d candidates", before, len(candidates))

            # Bound the live cross-encoder cost. Quality plateaus well before this
            # depth, so the cap trades unused depth for predictable latency.
            if pool_cap and len(candidates) > pool_cap:
                logger.info("Capping pool %d → %d for scoring", len(candidates), pool_cap)
                candidates = candidates[:pool_cap]

            logger.info("Scoring %d candidates with the reranker …", len(candidates))
            scored, relevance = reranker.rerank(
                query_text=query_text,
                candidates=candidates,
                game_doc_lookup=doc_map,
                top_k=len(candidates),
                bayesian_avg_map=bayesian_avg_map,
                rating_weight=rating_w,
                min_ce_score=min_rerank_score,
            )

        render = lambda flt: show_results(
            scored, flt,
            game_docs=game_docs, game_info_map=game_info_map, top_k=top_k_rank,
            use_diversity=use_diversity, target_genres=target_genres,
            target_systems=target_systems, label=query, relevance=relevance,
        )
        render({})

        # Refine loop: filters are applied to the cached ranking above, so
        # they narrow these results instead of triggering a fresh search.
        print(FILTER_HELP)
        while True:
            try:
                raw = input("Filter > ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            command = raw.lower()
            if command in ("back", "b"):
                break
            if command in ("quit", "exit", "q"):
                return
            if command in ("help", "?"):
                print(FILTER_HELP)
                continue
            if command in ("", "clear", "reset"):
                render({})
                continue
            hard_filters = _parse_filters(raw)
            if not hard_filters:
                print("  Could not read that. Filters look like 'rating:3.5; year:2020-2026'.")
                continue
            render(hard_filters)


# ---------------------------------------------------------------------------
# Evaluation mode
# ---------------------------------------------------------------------------

def run_evaluation(
    retriever,
    game_docs, doc_map,
    cfg,
    reranker=None,
    profile_map=None,
    bayesian_avg_map=None,
    rerank: bool = False,
) -> None:
    data_dir   = Path(cfg["paths"]["data_dir"])
    retr_cfg   = cfg["retrieval"]
    top_k_ret  = retr_cfg["top_k_retrieve"]
    min_score  = retr_cfg.get("min_retrieval_score", 0.25)
    profile_map = profile_map or {}

    logger.info("Loading test interactions …")
    interactions = pd.read_parquet(data_dir / "interactions.parquet")
    test_pos = interactions[
        (interactions["split"] == "test") & (interactions["label"] == 1)
    ]

    ground_truth: Dict[str, set] = (
        test_pos.groupby("userid")["gameid"].apply(set).to_dict()
    )

    mode = "retrieval + reranking" if rerank else "raw retrieval"
    logger.info("Running %s for %d test users …", mode, len(ground_truth))
    predictions: Dict[str, List[str]] = {}
    n_skipped = 0
    for n, uid in enumerate(ground_truth, start=1):
        emb = retriever._encode_userid(uid)
        if emb is None:
            n_skipped += 1
            continue
        if rerank:
            # Mirror the interactive path: score every candidate above the
            # cosine threshold, so the measurement reflects what users see.
            candidates = retriever.index.search(emb, min_score=min_score)
            scored, _ = reranker.rerank(
                query_text=profile_map.get(uid, ""),
                candidates=candidates,
                game_doc_lookup=doc_map,
                top_k=len(candidates),
                bayesian_avg_map=bayesian_avg_map,
                rating_weight=retr_cfg.get("rating_weight", 0.5),
                min_ce_score=retr_cfg.get("min_rerank_score", 0.25),
            )
            predictions[uid] = [gid for gid, _ in scored]
        else:
            candidates = retriever.index.search(emb, top_k=top_k_ret)
            predictions[uid] = [gid for gid, _ in candidates]
        if rerank and n % 100 == 0:
            logger.info("  %d/%d users scored", n, len(ground_truth))

    if n_skipped:
        logger.warning("%d test users had no profile and were skipped", n_skipped)

    results = evaluate_retrieval(
        predictions=predictions,
        ground_truth=ground_truth,
        ks=(1, 5, 10, 20, 50),
    )

    print("\n" + "=" * 50)
    print(f"  Evaluation results (test split, {mode})")
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
    parser.add_argument("--rerank", action="store_true",
                        help="Evaluate mode: score candidates with the reranker, "
                             "as the interactive pipeline does (slower)")
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
        precomputed_user, precomputed_game,
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
            precomputed_user=precomputed_user,
            precomputed_game=precomputed_game,
        )
    else:
        run_evaluation(
            retriever, game_docs, doc_map, cfg,
            reranker=reranker, profile_map=profile_map,
            bayesian_avg_map=bayesian_avg_map, rerank=args.rerank,
        )


if __name__ == "__main__":
    main()
