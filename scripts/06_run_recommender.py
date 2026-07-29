#!/usr/bin/env python
"""
Step 6 – Run the full retrieval + reranking pipeline.

The terminal equivalent of app.py, with the same four search modes:

    game      pick a game      → games with a similar feel
    author    pick an author   → games in their spirit, excluding their own
    reviewer  pick a reviewer  → what suits their taste, from their history
    vibe      pick systems/tags → games matching that combination

Every mode picks from a type-ahead list, so nothing here needs an ID pasted out
of an IFDB URL. Results are then narrowed with the same substring filters the web
app applies.

Navigation is uniform at every prompt: 'back' (or Ctrl-C) goes up one level,
'quit' leaves from any depth.

Usage
-----
    # Interactive
    python scripts/06_run_recommender.py

    # Offline evaluation on the test split
    python scripts/06_run_recommender.py --mode evaluate
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
from src.data.preprocessor import (
    SYSTEM_GENRE_SEPARATORS, TAG_SEPARATORS,
    build_author_profiles, build_display_map, clean_frequencies, format_display,
    format_profile_text, author_game_map, parse_profile_text, profile_display,
)
from src.data.pickers import (
    author_choices, game_choices, reviewer_choices, vocab_choices,
)
from src.utils.prompts import (
    HAS_PROMPT_TOOLKIT, Cancelled, Quit, choose, pick_many, pick_one, read_line,
)
from src.pipeline.ranker import Reranker, evaluate_retrieval, select_results
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
    """Return display cells for one game row, in `_add_game_columns` order."""
    # Convert Series → plain dict so subsequent .get() calls return scalars.
    row: dict = meta.loc[gid].to_dict() if gid in meta.index else {}
    rel = relevance.get(gid) if relevance else None
    return (
        score,
        "–" if rel is None else f"{rel:.2f}",
        _rating_cell(row),
        str(row.get("title", gid)),
        str(row.get("author", "")),
        str(row.get("year", "")),
        str(row.get("system_display", row.get("system", ""))),
        str(row.get("genre_display", row.get("genre", ""))),
        # *_display columns are added at load time: deduplicated, canonically cased,
        # ordered by how many games use each tag.
        str(row.get("tags_display", row.get("tags", ""))),
    )


def _add_game_columns(table) -> None:
    """
    Add the game columns, in the web app's order.

    Score first, then the two numbers it is made of, then the game itself — so
    the trade-off between relevance and rating sits next to the score it explains
    rather than at the far end of a wide table.
    """
    table.add_column("score",     style="cyan",       width=6)
    table.add_column("relevance", style="green",      width=9)
    table.add_column("rating",    style="magenta",    width=10)
    table.add_column("title",     style="bold white", min_width=24)
    table.add_column("author",    style="yellow")
    table.add_column("year",      style="dim",        width=6)
    table.add_column("system",    style="blue")
    table.add_column("genre",     style="cyan")
    table.add_column("tags",      style="violet")


def print_results(
    results: List[tuple],
    game_meta: pd.DataFrame,
    relevance: Optional[dict] = None,
) -> None:
    """
    Display top-K results in a readable table.

    Untitled: the Query panel printed just above already names what was searched
    from, and repeating it here read as noise — worst in vibe mode, where the
    only available label was the raw profile text.
    """
    meta = game_meta.set_index("gameid")

    if HAS_RICH:
        from rich.table import Table
        table = Table(show_lines=True)
        table.add_column("#", style="dim", width=4)
        _add_game_columns(table)
        for rank, (gid, score) in enumerate(results, start=1):
            table.add_row(str(rank), *_game_row_cells(gid, f"{score:.2f}", meta, relevance))
        console.print(table)
    else:
        print()
        print("-" * 80)
        for rank, (gid, score) in enumerate(results, start=1):
            row: dict = meta.loc[gid].to_dict() if gid in meta.index else {}
            rating_str = _rating_suffix(row)
            year = row.get("year", "")
            year_str = f" {year}" if year else ""
            rel = relevance.get(gid) if relevance else None
            rel_str = f" rel {rel:.2f}" if rel is not None else ""
            print(f"  {rank:2d}. [{score:.2f}]{rel_str} {row.get('title', gid)} "
                  f"({row.get('author','')}){year_str} — {row.get('system','')}{rating_str}")
        print()


def print_query_panel(kind: str, subject: str, profile: str) -> None:
    """
    Print what is being searched from, in one shape for all four modes.

    A line naming the pick, then the profile it resolved to, rendered the app's
    way — "inform // parser, puzzles", no field labels. Game mode used to print
    the seed game's full metadata row instead, which put the query in a different
    place and format depending on how you got there, and showed columns (genre,
    rating) that say nothing about what the search will actually match on.
    """
    heading = f"{kind}: {subject}" if subject else kind
    if HAS_RICH:
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        # A grid rather than one wrapped Text: profiles are long enough to wrap,
        # and in a plain Text the continuation returns to column zero, level with
        # the heading, which reads as a second field rather than more of the same.
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=1)
        grid.add_column(style="yellow", overflow="fold")
        grid.add_row("", profile)
        console.print(Panel(Group(Text(heading, style="bold cyan"), grid),
                            title="query", border_style="cyan"))
    else:
        print(f"\n{heading}")
        print("-" * 60)
        print(f"  {profile}")
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
        # Multi-word values work unquoted ("tags:slice of life"), but people
        # reasonably try quoting them, so accept and discard the quotes.
        val = val.strip().strip('"').strip("'").strip()
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
        elif key == "genre":
            filters["genre"] = val
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
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        # A precompute in progress, or a half-written file from a failed run.
        # Falling back to live scoring is always correct, just slower.
        logger.warning("Ignoring unreadable %s (%s) — scoring live instead",
                       path.name, type(exc).__name__)
        return {}
    table = {
        key: (grp["gameid"].tolist(), grp["score"].to_numpy(), grp["relevance"].to_numpy())
        for key, grp in frame.groupby(key_col, sort=False)
    }
    logger.info("Loaded %s — %d keys", path.name, len(table))
    return table


FILTER_HELP = """  refine these results with filters, e.g.  year:2020-2026; rating:3.5; count:2
  keys: year, author, system, tags, genre, rating, count   (each entry replaces the last)
  values match what the results show, e.g.  tags:IFComp 2025  ·  system:Inform 7
  'clear' show all again  ·  'back' pick again  ·  'mode' change mode  ·  'quit' exit"""


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
    relevance: Optional[dict] = None,
) -> None:
    """
    Filter, diversify, and print a slice of an already-scored candidate list.

    `scored` is the whole reranked pool for one query, so filtering here strictly
    narrows what the user is already looking at rather than changing which
    candidates ever reached the reranker.
    """
    results = select_results(
        scored, hard_filters, game_info_map, top_k,
        use_diversity=use_diversity,
        target_genres=target_genres, target_systems=target_systems,
    )
    if hard_filters:
        logger.info("Filters kept %d of %d scored candidates", len(results), len(scored))

    if not results:
        print("  nothing matches those filters. try relaxing them, or 'clear'.\n")
        return
    print_results(results, game_docs, relevance=relevance)


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

    # Precompute the display form of each free-text field once, rather than
    # reformatting on every render. System and genre also split on "/", which
    # IFDB uses inconsistently ("Drama / Political").
    for column, separators in (
        ("tags", TAG_SEPARATORS),
        ("system", SYSTEM_GENRE_SEPARATORS),
        ("genre", SYSTEM_GENRE_SEPARATORS),
    ):
        if column not in game_docs.columns:
            continue
        display_map = build_display_map(game_docs, column, separators)
        game_docs[f"{column}_display"] = game_docs[column].apply(
            lambda raw, m=display_map, s=separators: format_display(raw, m, s)
        )
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

    # Author profiles: same shape as user profiles, aggregated over the games an
    # author wrote. Derived from game_docs, so no separate artefact to keep in sync.
    author_profiles = build_author_profiles(game_docs)
    author_profile_map = dict(zip(author_profiles["authorid"], author_profiles["profile_text"]))
    author_name_map = dict(zip(author_profiles["authorid"], author_profiles["name"]))
    author_games = author_game_map(game_docs)
    logger.info("Built %d author profiles", len(author_profile_map))

    # Precomputed rankings for the enumerable query modes; absent files simply
    # mean those modes fall back to scoring live.
    precomputed_user = load_precomputed(data_dir / "precomputed_userid.parquet", "userid")
    precomputed_game = load_precomputed(data_dir / "precomputed_gameid.parquet", "seed_gameid")
    precomputed_author = load_precomputed(data_dir / "precomputed_authorid.parquet", "authorid")

    return (
        retriever, reranker, query_encoder,
        game_docs, doc_map, profile_map, name_map,
        bayesian_avg_map,
        reviews_df, playedgames_df,
        game_query_text_map,
        game_info_map,
        precomputed_user, precomputed_game, precomputed_author,
        author_profile_map, author_name_map, author_games,
    )


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

SEARCH_MODES = [
    ("game      ·  games with a similar feel", "game"),
    ("author    ·  games in an author's spirit, excluding their own", "author"),
    ("reviewer  ·  what suits a reviewer's taste, from their history", "reviewer"),
    ("vibe      ·  games matching the systems and tags you pick", "vibe"),
]

# Offered as completions on the filter line, so the syntax is discoverable
# without reading the help text.
FILTER_WORDS = ["year:", "author:", "system:", "tags:", "genre:", "rating:", "count:",
                "clear", "back", "mode", "quit", "help"]


def _picker(cache: dict, name: str, build):
    """Build a choice list on first use.

    Each list costs a pass over 10k games, and a session only ever touches the
    one or two its mode needs.
    """
    if name not in cache:
        cache[name] = build()
    return cache[name]


def _profile_for(picks: dict, game_docs, query_text: str, corpus_order: bool) -> str:
    """Render a profile the app's way; corpus order for game/vibe (see profile_display)."""
    if not corpus_order:
        return profile_display(query_text)
    freq = _picker(picks, "freq", lambda: (
        clean_frequencies(game_docs, "system_clean"),
        clean_frequencies(game_docs, "tags_clean"),
    ))
    return profile_display(query_text, *freq)


def _prompt_query(
    search: str, picks: dict, *,
    retriever, query_encoder, game_docs, doc_map, profile_map, name_map,
    game_query_text_map, reviews_df, playedgames_df,
    author_profile_map, author_name_map, author_games,
):
    """
    Ask this mode for its input and return everything the scoring path needs.

    Returns (key, query_text, emb, seen_games) or None if the mode could
    not build a query. Raises Cancelled when the user backs out.
    """
    if search == "game":
        choices = _picker(picks, "game", lambda: game_choices(game_docs))
        gid = pick_one("game", choices)
        label = next(lbl for lbl, v in choices if v == gid)
        query_text = game_query_text_map.get(gid, doc_map.get(gid, gid))
        print_query_panel("game", label, _profile_for(picks, game_docs, query_text, True))
        emb = retriever._encode_game_ids([gid])
        if emb is None:
            print(f"  could not encode game '{gid}'")
            return None
        # Its own entry would otherwise top the ranking.
        return gid, query_text, emb, {gid}

    if search == "author":
        choices = _picker(picks, "author", lambda: author_choices(pd.DataFrame({
            "authorid": list(author_profile_map),
            "name": [author_name_map.get(a, a) for a in author_profile_map],
            "game_count": [len(author_games.get(a, [])) for a in author_profile_map],
        })))
        key = pick_one("author", choices)
        query_text = author_profile_map[key]
        name = author_name_map.get(key, key)
        n_games = len(author_games.get(key, []))
        print_query_panel("author", f"{name} — {n_games} game{'s' if n_games != 1 else ''}",
                          _profile_for(picks, game_docs, query_text, False))
        emb = query_encoder.encode([query_text], normalize_embeddings=True)[0]
        # Suppress the author's own catalogue, as game mode suppresses its seed.
        return key, query_text, emb, set(author_games.get(key, []))

    if search == "reviewer":
        choices = _picker(picks, "reviewer",
                          lambda: reviewer_choices(profile_map, name_map, reviews_df))
        uid = pick_one("reviewer", choices)
        emb = retriever._encode_userid(uid)
        if emb is None:
            print(f"  no profile found for reviewer '{uid}'")
            return None
        query_text = profile_map.get(uid, "")
        name = name_map.get(uid, "") if name_map else ""
        print_query_panel("reviewer", name or uid, _profile_for(picks, game_docs, query_text, False))

        # Everything they have already reviewed or played, so recommendations are
        # things they have not seen.
        seen: set = set()
        if reviews_df is not None and "userid" in reviews_df.columns:
            seen = set(reviews_df[reviews_df["userid"] == uid]["gameid"])
        if playedgames_df is not None and "userid" in playedgames_df.columns:
            seen |= set(playedgames_df[playedgames_df["userid"] == uid]["gameid"])
        return uid, query_text, emb, seen

    # vibe — built from the trained vocabulary, so every pick is a token the
    # encoders actually saw.
    systems_c, tags_c = _picker(picks, "vocab", lambda: vocab_choices(game_docs))
    print("  pick systems, then tags. blank line moves on.")
    systems = pick_many("system", systems_c)
    tags = pick_many("tag", tags_c)
    if not systems and not tags:
        print("  nothing picked.")
        return None
    query_text = format_profile_text(systems, tags)
    print_query_panel("vibe", "", _profile_for(picks, game_docs, query_text, True))
    emb = query_encoder.encode([query_text], normalize_embeddings=True)[0]
    return None, query_text, emb, set()


def run_interactive(
    retriever, reranker, query_encoder,
    game_docs, doc_map, profile_map, name_map,
    cfg,
    bayesian_avg_map=None,
    reviews_df=None,
    playedgames_df=None,
    game_query_text_map=None,
    game_info_map=None,
    precomputed_user=None,
    precomputed_game=None,
    precomputed_author=None,
    author_profile_map=None,
    author_name_map=None,
    author_games=None,
) -> None:
    precomputed_user = precomputed_user or {}
    precomputed_game = precomputed_game or {}
    precomputed_author = precomputed_author or {}
    author_profile_map = author_profile_map or {}
    author_name_map = author_name_map or {}
    author_games = author_games or {}
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
    print("  IFDB recs")
    print("=" * 60)
    print(f"  score threshold  : {min_score}   output: {top_k_rank}")
    print("  pick something to search from, then refine the results with filters.")
    print("  'back' goes up a level anywhere  ·  'quit' exits.")
    if not HAS_PROMPT_TOOLKIT:
        print("  (install prompt_toolkit for type-ahead pickers)")
    print()

    picks: dict = {}
    mode = None

    while True:
        if mode is None:
            print("  search by:")
            try:
                mode = choose("mode", SEARCH_MODES)
            except (Cancelled, Quit):
                return   # nothing sits above the menu, so backing out of it exits
            print()

        try:
            built = _prompt_query(
                mode, picks,
                retriever=retriever, query_encoder=query_encoder,
                game_docs=game_docs, doc_map=doc_map,
                profile_map=profile_map, name_map=name_map,
                game_query_text_map=game_query_text_map,
                reviews_df=reviews_df, playedgames_df=playedgames_df,
                author_profile_map=author_profile_map,
                author_name_map=author_name_map, author_games=author_games,
            )
        except Cancelled:
            mode = None      # out of the picker, back to the mode menu
            continue
        except Quit:
            return

        if built is None:
            continue
        key, query_text, emb, seen_games = built

        # Diversity targets come from the query itself, so results spread across
        # the systems the query is actually made of.
        target_genres, target_systems = _parse_profile_targets(query_text)
        if mode == "game":
            target_systems = set(list(target_systems)[:1])

        # game, author and reviewer draw from fixed key sets, so their rankings
        # are precomputed offline (scripts/07_precompute.py) and served as a
        # lookup. Only vibe is scored live.
        cached = None
        if key is not None:
            cached = {"reviewer": precomputed_user,
                      "game": precomputed_game,
                      "author": precomputed_author}[mode].get(key)

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
            if seen_games:
                candidates = [(gid, s) for gid, s in candidates if gid not in seen_games]

            if not candidates:
                print("  no candidates above the retrieval threshold.\n")
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
            target_systems=target_systems, relevance=relevance,
        )
        render({})

        # Refine loop: filters are applied to the cached ranking above, so
        # they narrow these results instead of triggering a fresh search.
        print(FILTER_HELP)
        to_menu = False
        while True:
            try:
                raw = read_line("filter", FILTER_WORDS)
            except Cancelled:
                break    # same as 'back': up one level, to the picker
            command = raw.strip().lower()
            if command in ("back", "b"):
                break
            if command in ("mode", "m"):
                to_menu = True       # shortcut past the picker
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
                print("  could not read that. filters look like 'rating:3.5; year:2020-2026'.")
                continue
            render(hard_filters)

        if to_menu:
            mode = None


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
        precomputed_user, precomputed_game, precomputed_author,
        author_profile_map, author_name_map, author_games,
    ) = load_artefacts(cfg)

    if args.mode == "interactive":
        run_interactive(
            retriever, reranker, query_encoder,
            game_docs, doc_map, profile_map, name_map,
            cfg,
            bayesian_avg_map=bayesian_avg_map,
            reviews_df=reviews_df,
            playedgames_df=playedgames_df,
            game_query_text_map=game_query_text_map,
            game_info_map=game_info_map,
            precomputed_user=precomputed_user,
            precomputed_game=precomputed_game,
            precomputed_author=precomputed_author,
            author_profile_map=author_profile_map,
            author_name_map=author_name_map,
            author_games=author_games,
        )
    else:
        run_evaluation(
            retriever, game_docs, doc_map, cfg,
            reranker=reranker, profile_map=profile_map,
            bayesian_avg_map=bayesian_avg_map, rerank=args.rerank,
        )


if __name__ == "__main__":
    main()
