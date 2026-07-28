"""
Gradio front-end for the IFDB recommender.

Four ways in, all resolving to the same profile-format query the encoders were
trained on:

    game    – "more like this"             (precomputed)
    author  – "more in this spirit"        (precomputed)
    user    – from a reviewer's history    (precomputed)
    browse  – pick systems and tags        (scored live)

Run locally with `python app.py`; Hugging Face Spaces picks this file up by name.
"""

import importlib.util
import logging
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd
import yaml

from src.data.preprocessor import (
    AUTHOR_SEPARATORS,
    SYSTEM_GENRE_SEPARATORS,
    build_author_profiles,
    build_display_map,
    format_profile_text,
    profile_vocabulary,
)
from src.pipeline.ranker import select_results

logger = logging.getLogger(__name__)

IFDB_GAME_URL = "https://ifdb.org/viewgame?id={gameid}"
MODES = ["game", "author", "user", "browse"]

# Catch-all credits and single-system studios. Both sit at the top of a
# frequency-ordered list, and both give poor recommendations — an aggregate of
# 61 unrelated games, or a catalogue so system-specific that almost nothing
# outside it matches. A bad result from the most obvious first click is worse
# than the entry being absent.
EXCLUDED_AUTHORS = {
    "anonymous",
    "anonymous (first row software publishing inc.)",
    "failbetter games",
}

RESULT_COLUMNS = ["#", "score", "relevance", "rating", "title",
                  "author", "year", "system", "genre", "tags"]
# Relevance needs room for the word itself, and genre truncates badly below
# ~12%. The narrow numeric columns carry a little slack so their headers still
# fit on one line once a sort arrow appears beside them.
COLUMN_WIDTHS = ["4%", "6%", "8%", "8%", "13%", "9%", "6%", "7%", "12%", "27%"]
PAGE_SIZES = [10, 25, 50]
ANY_LABEL = "any"
# gr.Dropdown has no placeholder; `info` is the idiomatic way to show greyed-out
# guidance, and leaving value=None stops it preselecting the first entry — which
# reads as "this is a fixed menu" rather than "type here to search".
# ~85 KB retained per entry (measured over the live objects, not RSS), so
# 2,048 is roughly 171 MB against a 1.5 GB baseline in a 16 GB box. Raising it
# also reduces allocator churn, since every hit is a scoring that never runs.
BROWSE_CACHE_SIZE = 2_048
SEARCH_HINT = "start typing to search"
PICK_HINT = "choose one or more"
RATING_CHOICES = [round(0.5 * i, 1) for i in range(10)]      # 0.0 … 4.5
RATING_COUNT_CHOICES = [0, 1, 2, 5, 10, 25, 50]

# Longest values a cell may show before trailing off. Tags and multi-author
# credits are unbounded in the source data and will otherwise blow up row height.
CLIP = {"author": 34, "tags": 170, "genre": 46, "system": 20}

# NOTE: keep this free of /* */ comments — Gradio drops the remainder of the
# sheet when it encounters one, silently discarding later rules.
# The #results rules size the table to its content: no inner scrollbar, and no
# reserved empty space before the pager. Row height varies with how far the tags
# column wraps, so a pixel estimate is always wrong in one direction.
CSS = """
:root, .gradio-container { font-family: "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace !important; }
.gradio-container { max-width: 100% !important; padding: 1.2em 1.6em !important; }
h1 { font-weight: 600 !important; letter-spacing: -0.01em; margin-bottom: 0.5em !important; }
.gr-button { border-radius: 2px !important; }
input, textarea, select { border-radius: 2px !important; }
table { font-size: 0.82em !important; table-layout: fixed !important; }
table td { vertical-align: top !important; }
#results table td, #results table th { padding: 0.35em 0.5em !important; }
#pager { align-items: center; }
.block-header { padding: 0.75em 0 0.15em 0.9em !important; opacity: 0.8; }
#results .table-wrap { max-height: none !important; height: auto !important; overflow-y: visible !important; }
#results div[class*="table"] { max-height: none !important; height: auto !important; overflow-y: visible !important; }
footer { display: none !important; }
"""


def _load_pipeline_module():
    """Import 06_run_pipeline.py by path — its name is not a valid identifier."""
    path = Path(__file__).resolve().parent / "scripts" / "06_run_pipeline.py"
    spec = importlib.util.spec_from_file_location("pipeline_mod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = _load_pipeline_module()
CFG = yaml.safe_load(open("config.yaml"))
RETR = CFG["retrieval"]

(
    RETRIEVER, RERANKER, QUERY_ENCODER,
    GAME_DOCS, DOC_MAP, PROFILE_MAP, USER_NAME_MAP,
    BAYESIAN_AVG_MAP, REVIEWS_DF, PLAYEDGAMES_DF,
    GAME_QUERY_TEXT_MAP, GAME_INFO_MAP,
    PRE_USER, PRE_GAME, PRE_AUTHOR,
    AUTHOR_PROFILE_MAP, AUTHOR_NAME_MAP, AUTHOR_GAMES,
) = pipeline.load_artefacts(CFG)

META = GAME_DOCS.set_index("gameid")
AUTHOR_PROFILES = build_author_profiles(GAME_DOCS)
SYSTEM_CASING = build_display_map(GAME_DOCS, "system", SYSTEM_GENRE_SEPARATORS)
GENRE_CASING = build_display_map(GAME_DOCS, "genre", SYSTEM_GENRE_SEPARATORS)
TAG_CASING = build_display_map(GAME_DOCS, "tags")
AUTHOR_CASING = build_display_map(GAME_DOCS, "author", AUTHOR_SEPARATORS)


# ---------------------------------------------------------------------------
# Picker choices — labels users recognise, IDs behind them
# ---------------------------------------------------------------------------

def _game_choices():
    """
    Games as "Title — Author (Year)".

    119 titles are shared by 2-5 different games, so a bare title would make all
    but one of each unreachable.
    """
    frame = GAME_DOCS.sort_values("review_count", ascending=False)
    return [
        (f"{r.title} — {r.author}" + (f" ({r.year})" if str(r.year).strip() else ""), r.gameid)
        for r in frame.itertuples(index=False)
    ]


def _author_choices():
    frame = AUTHOR_PROFILES[~AUTHOR_PROFILES["authorid"].isin(EXCLUDED_AUTHORS)]
    frame = frame.sort_values("game_count", ascending=False)
    return [
        (f"{r.name}  ·  {r.game_count} game{'s' if r.game_count != 1 else ''}", r.authorid)
        for r in frame.itertuples(index=False)
    ]


def _user_choices():
    """Users by review count, so recognisable reviewers surface first."""
    counts = REVIEWS_DF.groupby("userid").size() if REVIEWS_DF is not None else pd.Series(dtype=int)
    rows = [(USER_NAME_MAP.get(uid, uid), uid, int(counts.get(uid, 0))) for uid in PROFILE_MAP]
    rows.sort(key=lambda r: -r[2])
    return [(f"{name}  ·  {n} reviews", uid) for name, uid, n in rows]


def _vocab_choices():
    """
    Systems and tags for *building a query*, canonically cased, clean values
    underneath.

    Drawn from `tags_clean`, so competition tags (XYZZY, IFComp, Spring Thing)
    are absent by design — preprocessing strips them, the encoders never saw
    them, and offering one here would send the model a token it cannot
    represent. Use the tags *filter* to narrow results by those instead.
    """
    systems, tags = profile_vocabulary(GAME_DOCS, n_systems=10_000, n_tags=10_000)
    return (
        [(SYSTEM_CASING.get(s, (s, 0))[0], s) for s in systems],
        [(TAG_CASING.get(t, (t, 0))[0], t) for t in tags],
    )


def _filter_choices():
    """
    Values for the filter dropdowns, most common first, canonically cased.

    Everything is offered, not a popular subset: filters match the original IFDB
    values, and a truncated list silently hides real tags — "XYZZY Best Game"
    (28 games) sat below a top-400 cutoff even though people search for exactly
    that. These are typeaheads, so length costs nothing but payload.
    """
    genres = [g for g, _ in sorted(GENRE_CASING.items(), key=lambda kv: -kv[1][1])]
    systems = [s for s, _ in sorted(SYSTEM_CASING.items(), key=lambda kv: -kv[1][1])]
    authors = [a for a, _ in sorted(AUTHOR_CASING.items(), key=lambda kv: -kv[1][1])]
    tags = [t for t, _ in sorted(TAG_CASING.items(), key=lambda kv: -kv[1][1])]
    present = {int(y) for y in GAME_DOCS["year"] if str(y).strip().isdigit()}
    # A continuous span, not just the years that happen to occur: the data has no
    # 1968, 1969, or 1971-1976, and a dropdown that skips them looks broken.
    years = list(range(max(present), min(present) - 1, -1))
    return (
        # An explicit "any" entry, because a single-select dropdown otherwise
        # defaults to its first choice and cannot be cleared.
        [(ANY_LABEL, "")] + [(GENRE_CASING[g][0], GENRE_CASING[g][0]) for g in genres],
        [(ANY_LABEL, "")] + [(SYSTEM_CASING[s][0], SYSTEM_CASING[s][0]) for s in systems],
        [(ANY_LABEL, "")] + [(AUTHOR_CASING[a][0], AUTHOR_CASING[a][0]) for a in authors],
        [TAG_CASING[t][0] for t in tags],
        years,
    )


GAME_CHOICES = _game_choices()
AUTHOR_CHOICES = _author_choices()
USER_CHOICES = _user_choices()
SYSTEM_CHOICES, TAG_CHOICES = _vocab_choices()
(GENRE_FILTER_CHOICES, SYSTEM_FILTER_CHOICES, AUTHOR_FILTER_CHOICES,
 TAG_FILTER_CHOICES, YEAR_CHOICES) = _filter_choices()
YEAR_MAX, YEAR_MIN = YEAR_CHOICES[0], YEAR_CHOICES[-1]


# ---------------------------------------------------------------------------
# Ranking and rendering
# ---------------------------------------------------------------------------

def _as_int(value, default):
    """
    Coerce a dropdown value to int.

    Numeric dropdown values can arrive as strings depending on how the browser
    serialises them, and an uncoerced string makes list slicing raise — which
    leaves the previous table on screen and looks like the control did nothing.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clip(value, limit):
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip(" ,") + "…"


def _rank(query_text, emb, exclude, cached):
    """Serve a precomputed ranking, or score the pool live."""
    if cached is not None:
        gids, scores, rels = cached
        return list(zip(gids, scores)), dict(zip(gids, rels))

    candidates = RETRIEVER.index.search(emb, min_score=RETR.get("min_retrieval_score", 0.25))
    candidates = [(g, s) for g, s in candidates if g not in exclude]
    if not candidates:
        return [], {}

    if RETR.get("prefilter_by_tag", True):
        from src.data.preprocessor import parse_profile_text
        from src.pipeline.retriever import filter_by_tag_overlap
        _, query_tags = parse_profile_text(query_text)
        if query_tags:
            candidates = filter_by_tag_overlap(candidates, GAME_INFO_MAP, set(query_tags))

    cap = RETR.get("rerank_pool_cap", 0)
    if cap and len(candidates) > cap:
        candidates = candidates[:cap]

    return RERANKER.rerank(
        query_text=query_text,
        candidates=candidates,
        game_doc_lookup=DOC_MAP,
        top_k=len(candidates),
        bayesian_avg_map=BAYESIAN_AVG_MAP,
        rating_weight=RETR.get("rating_weight", 0.5),
        min_ce_score=RETR.get("min_rerank_score", 0.25),
    )


@lru_cache(maxsize=BROWSE_CACHE_SIZE)
def _score_browse(systems, tags):
    """
    Score a browse query, shared across every visitor.

    Session state already avoids rescoring while one person narrows filters, but
    it is per-session — ten people picking the same tags each pay full price, and
    with inference serialised that duplicated work extends the queue for
    everyone. Browse is the only mode that scores live, so this is the whole of
    the live cost.

    Deliberately no TTL: a scored pool depends only on the query, the models, and
    the index, all fixed for the life of the process. Regenerating artefacts
    needs a restart to take effect anyway, so entries cannot go stale — an
    expiry would only add moving parts. Bounding the size is enough.

    Arguments are tuples so they hash; the returned lists are never mutated
    downstream (`select_results` builds new ones), so sharing them is safe.
    """
    query_text = format_profile_text(list(systems), list(tags))
    emb = QUERY_ENCODER.encode([query_text], normalize_embeddings=True)[0]
    return _rank(query_text, emb, set(), None)


def _page_table(results, relevance, page, per_page):
    start = page * per_page
    rows = []
    for rank, (gid, score) in enumerate(results[start : start + per_page], start=start + 1):
        row = META.loc[gid].to_dict() if gid in META.index else {}
        rel = relevance.get(gid)
        rows.append([
            rank,
            f"{score:.4f}",
            "–" if rel is None else f"{rel:.2f}",
            pipeline._rating_cell(row),
            f"[{row.get('title', gid)}]({IFDB_GAME_URL.format(gameid=gid)})",
            _clip(row.get("author", ""), CLIP["author"]),
            str(row.get("year", "")),
            _clip(row.get("system_display", row.get("system", "")), CLIP["system"]),
            _clip(row.get("genre_display", row.get("genre", "")), CLIP["genre"]),
            _clip(row.get("tags_display", row.get("tags", "")), CLIP["tags"]),
        ])
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _pager_text(state):
    total = len(state["results"])
    if not total:
        return ""
    pages = max(1, -(-total // state["per_page"]))
    return f"page {state['page'] + 1} of {pages}  ·  {total} results"


def _build_filters(genre, system, author, tags, rating, count, year_from, year_to):
    """Turn the filter dropdowns into apply_hard_filters kwargs."""
    filters = {}
    if genre:
        filters["genre"] = genre
    if system:
        filters["system"] = system
    if author:
        filters["author"] = author
    if tags:
        filters["tags"] = ", ".join(tags)
    if rating:
        filters["min_rating"] = float(rating)
    if count:
        filters["min_rating_count"] = int(count)
    # Only constrain years if the user actually narrowed the span. The dropdowns
    # default to the full range for legibility, but applying it would silently
    # drop the 184 games with no recorded year — apply_hard_filters excludes a
    # game it cannot date.
    low = int(year_from) if year_from else YEAR_MIN
    high = int(year_to) if year_to else YEAR_MAX
    if low > YEAR_MIN or high < YEAR_MAX:
        filters["year_range"] = f"{low}-{high}"
    return filters


def recommend(state, mode, game, author, user, systems, tags,
              f_genre, f_system, f_author, f_tags, f_rating, f_count,
              f_year_from, f_year_to, per_page):
    """Resolve the chosen mode to a query, rank, and render the first page."""
    blank = pd.DataFrame(columns=RESULT_COLUMNS)
    empty_state = {"results": [], "scored": [], "relevance": {},
                   "query_key": None, "page": 0, "per_page": per_page}
    per_page = _as_int(per_page, 25)
    hard_filters = _build_filters(f_genre, f_system, f_author, f_tags,
                                  f_rating, f_count, f_year_from, f_year_to)
    exclude, cached, emb, query_text = set(), None, None, ""

    if mode == "game":
        if not game:
            return blank, "Pick a game to get recommendations like it.", empty_state, ""
        query_text = GAME_QUERY_TEXT_MAP.get(game, DOC_MAP.get(game, ""))
        cached, exclude = PRE_GAME.get(game), {game}
        emb = None if cached is not None else RETRIEVER._encode_game_ids([game])
        note = f"Games like **{META.loc[game, 'title']}**"

    elif mode == "author":
        if not author:
            return blank, "Pick an author.", empty_state, ""
        query_text = AUTHOR_PROFILE_MAP.get(author, "")
        cached = PRE_AUTHOR.get(author)
        exclude = set(AUTHOR_GAMES.get(author, []))
        emb = None if cached is not None else QUERY_ENCODER.encode([query_text], normalize_embeddings=True)[0]
        note = f"In the spirit of **{AUTHOR_NAME_MAP.get(author, author)}** — their own games excluded"

    elif mode == "user":
        if not user:
            return blank, "Pick a reviewer.", empty_state, ""
        query_text = PROFILE_MAP.get(user, "")
        cached = PRE_USER.get(user)
        if REVIEWS_DF is not None:
            exclude = set(REVIEWS_DF[REVIEWS_DF["userid"] == user]["gameid"])
        if PLAYEDGAMES_DF is not None:
            exclude |= set(PLAYEDGAMES_DF[PLAYEDGAMES_DF["userid"] == user]["gameid"])
        emb = None if cached is not None else RETRIEVER._encode_userid(user)
        note = f"For **{USER_NAME_MAP.get(user, user)}** — games they've rated or played excluded"

    else:  # browse
        if not systems and not tags:
            return blank, "Pick at least one system or tag.", empty_state, ""
        query_text = format_profile_text(list(systems or []), list(tags or []))
        emb = QUERY_ENCODER.encode([query_text], normalize_embeddings=True)[0]
        note = f"`{query_text}`"

    if not query_text:
        return blank, "No profile available for that selection.", empty_state, ""

    # Scoring is the expensive half and depends only on the query, never on the
    # filters. Reuse it while the user narrows results, and drop it as soon as
    # they change what they are searching for.
    query_key = (mode, game, author, user, tuple(systems or []), tuple(tags or []))
    reused = bool(state) and state.get("query_key") == query_key and state.get("scored")
    if reused:
        scored, relevance = state["scored"], state["relevance"]

    elif mode == "browse":
        scored, relevance = _score_browse(tuple(systems or []), tuple(tags or []))
    else:
        scored, relevance = _rank(query_text, emb, exclude, cached)
    if not scored:
        return blank, "Nothing above the retrieval threshold for that query.", empty_state, ""

    targets = pipeline._parse_profile_targets(query_text) if mode != "browse" else (set(), set())
    # Ask for every result the pool can yield, then paginate locally.
    results = select_results(
        scored, hard_filters, GAME_INFO_MAP, len(scored),
        use_diversity=RETR.get("use_diversity", True),
        target_genres=targets[0], target_systems=targets[1],
    )
    if not results:
        return blank, f"{note}\n\nNo results match those filters — try relaxing them.", empty_state, ""

    state = {"results": results, "scored": scored, "relevance": relevance,
             "query_key": query_key, "page": 0, "per_page": per_page}
    if reused:
        note += "  ·  reused scores"
    return _page_table(results, relevance, 0, per_page), note, state, _pager_text(state)


def turn_page(state, step):
    if not state or not state["results"]:
        return pd.DataFrame(columns=RESULT_COLUMNS), state, ""
    per_page = _as_int(state["per_page"], 25)
    pages = max(1, -(-len(state["results"]) // per_page))
    state = {**state, "per_page": per_page, "page": min(max(state["page"] + step, 0), pages - 1)}
    table = _page_table(state["results"], state["relevance"], state["page"], per_page)
    return table, state, _pager_text(state)


def resize_page(state, per_page):
    per_page = _as_int(per_page, 25)
    if not state or not state["results"]:
        return pd.DataFrame(columns=RESULT_COLUMNS), {**(state or {}), "per_page": per_page}, ""
    state = {**state, "per_page": per_page, "page": 0}
    table = _page_table(state["results"], state["relevance"], 0, per_page)
    return table, state, _pager_text(state)


def _visibility(mode):
    return [gr.update(visible=(mode == m)) for m in ("game", "author", "user", "browse", "browse")]


def build_ui():
    with gr.Blocks(title="IFDB Recommender") as demo:
        gr.Markdown("# IFDB Recommender")
        state = gr.State({"results": [], "scored": [], "relevance": {},
                          "query_key": None, "page": 0, "per_page": 25})

        with gr.Group():
            gr.Markdown("search type", elem_classes="block-header")
            mode = gr.Radio(MODES, value="game", show_label=False, interactive=True)
            game = gr.Dropdown(GAME_CHOICES, value=None, label="game", info=SEARCH_HINT,
                               filterable=True, visible=True)
            author = gr.Dropdown(AUTHOR_CHOICES, value=None, label="author", info=SEARCH_HINT,
                                 filterable=True, visible=False)
            user = gr.Dropdown(USER_CHOICES, value=None, label="reviewer", info=SEARCH_HINT,
                               filterable=True, visible=False)
            systems = gr.Dropdown(SYSTEM_CHOICES, value=[], label="systems", info=PICK_HINT,
                                  multiselect=True, visible=False)
            tags = gr.Dropdown(TAG_CHOICES, value=[], label="tags", info=PICK_HINT,
                               multiselect=True, visible=False)

        with gr.Group():
            gr.Markdown("result filters", elem_classes="block-header")
            with gr.Row():
                # Every filter defaults to a no-op, so an untouched block filters
                # nothing and each control can be returned to that state.
                # allow_custom_value lets someone type a fragment and press
                # return — "xyzzy" or "inform" then matches every value
                # containing it, since apply_hard_filters compares by substring.
                # Without it the dropdowns would only ever match one exact value.
                f_genre = gr.Dropdown(GENRE_FILTER_CHOICES, value="", label="genre",
                                      filterable=True, allow_custom_value=True)
                f_system = gr.Dropdown(SYSTEM_FILTER_CHOICES, value="", label="system",
                                       filterable=True, allow_custom_value=True)
                f_author = gr.Dropdown(AUTHOR_FILTER_CHOICES, value="", label="author",
                                       filterable=True, allow_custom_value=True)
                f_tags = gr.Dropdown(TAG_FILTER_CHOICES, value=[], label="tags",
                                     multiselect=True, filterable=True, allow_custom_value=True)
            with gr.Row():
                f_rating = gr.Dropdown(RATING_CHOICES, value=0, label="rating ≥")
                f_count = gr.Dropdown(RATING_COUNT_CHOICES, value=0, label="rating count ≥")
                f_year_from = gr.Dropdown(YEAR_CHOICES, value=YEAR_MIN, label="year ≥")
                f_year_to = gr.Dropdown(YEAR_CHOICES, value=YEAR_MAX, label="year ≤")

        with gr.Row():
            per_page = gr.Dropdown(PAGE_SIZES, value=25, label="results per page", scale=1)
            go = gr.Button("recommend", variant="primary", scale=3)

        note = gr.Markdown()
        table = gr.Dataframe(
            headers=RESULT_COLUMNS,
            datatype=["number", "str", "str", "str", "markdown", "str", "str", "str", "str", "str"],
            column_widths=COLUMN_WIDTHS,
            wrap=True,
            interactive=False,
            # Effectively uncapped; the CSS above sizes it to content.
            max_height=100_000,
            elem_id="results",
        )
        with gr.Row(elem_id="pager"):
            prev = gr.Button("◀ prev", scale=1)
            pager = gr.Markdown()
            nxt = gr.Button("next ▶", scale=1)

        mode.change(_visibility, mode, [game, author, user, systems, tags])
        inputs = [state, mode, game, author, user, systems, tags,
                  f_genre, f_system, f_author, f_tags, f_rating, f_count,
                  f_year_from, f_year_to, per_page]
        # One at a time: two CPUs shared between concurrent requests makes
        # everyone slow, whereas a queue makes the wait visible.
        go.click(recommend, inputs, [table, note, state, pager], concurrency_limit=1)
        prev.click(lambda s: turn_page(s, -1), state, [table, state, pager])
        nxt.click(lambda s: turn_page(s, +1), state, [table, state, pager])
        per_page.change(resize_page, [state, per_page], [table, state, pager])
    return demo


if __name__ == "__main__":
    # Gradio 6 takes theme and css at launch(), not on the Blocks constructor —
    # passing them to Blocks is accepted with a warning and then ignored.
    build_ui().queue().launch(theme=gr.themes.Monochrome(), css=CSS)
