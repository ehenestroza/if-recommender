"""
Gradio front-end for the IFDB recommender.

Four ways in, all resolving to the same profile-format query the encoders were
trained on:

    game    – "more like this"             (precomputed)
    author  – "more in this spirit"        (precomputed)
    reviewer – from a reviewer's history   (precomputed)
    vibe    – pick systems and tags        (scored live)

Run locally with `python app.py`; Hugging Face Spaces picks this file up by name.
"""

import importlib.util
import logging
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd
from collections import Counter
from html import escape
import yaml

from src.data.preprocessor import (
    AUTHOR_SEPARATORS,
    parse_profile_text,
    SYSTEM_GENRE_SEPARATORS,
    build_author_profiles,
    build_display_map,
    clean_frequencies,
    format_profile_text,
    profile_display,
    profile_vocabulary,
)
from src.data.pickers import (
    EXCLUDED_AUTHORS, author_choices, game_choices, reviewer_choices, vocab_choices,
)
from src.pipeline.ranker import select_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

IFDB_GAME_URL = "https://ifdb.org/viewgame?id={gameid}"
# "vibe" rather than "browse": the systems and tags are turned into a profile
# and matched semantically, not used as exact filters, and the name should not
# promise the latter.
MODES = ["game", "author", "reviewer", "vibe"]

RESULT_COLUMNS = ["#", "score", "relevance", "rating", "title",
                  "author", "year", "system", "genre", "tags"]
# Relevance needs room for the word itself, and genre truncates badly below
# ~12%. The narrow numeric columns carry a little slack so their headers still
# fit on one line once a sort arrow appears beside them.
COLUMN_WIDTHS = ["4%", "6%", "8%", "8%", "13%", "9%", "6%", "9%", "12%", "25%"]
PAGE_SIZES = [10, 25, 50]
SCROLL_TO_SUMMARY = ("() => { const el = document.getElementById('summary');"
                     " if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'}); }")
ANY_LABEL = "any"
# gr.Dropdown has no placeholder; `info` is the idiomatic way to show greyed-out
# guidance, and leaving value=None stops it preselecting the first entry — which
# reads as "this is a fixed menu" rather than "type here to search".
# ~85 KB retained per entry (measured over the live objects, not RSS), so
# 2,048 is roughly 171 MB against a 1.5 GB baseline in a 16 GB box. Raising it
# also reduces allocator churn, since every hit is a scoring that never runs.
BROWSE_CACHE_SIZE = 2_048
SEARCH_HINT = "start typing to search"
# The big pickers ship every option to the browser, so the first open costs a
# moment of client-side rendering. That is browser work, not server work — it
# does not compete with scoring.
BIG_HINT = "start typing to search · {n}K options, first open takes a second"
PICK_HINT = "choose one or more"
# The four text-valued filters all accept a typed fragment and match by
# substring, so they carry the same hint — documenting it on only one would
# imply the others behave differently.
FREE_TEXT_HINT = "type text fragments and press return · case insensitive"
RATING_CHOICES = [round(0.5 * i, 1) for i in range(10)]      # 0.0 … 4.5
RATING_COUNT_CHOICES = [0, 1, 2, 5, 10, 25, 50]

# Longest values a cell may show before trailing off. Tags and multi-author
# credits are unbounded in the source data and will otherwise blow up row height.
CLIP = {"author": 34, "tags": 170, "genre": 46, "system": 26}

# NOTE: keep this free of /* */ comments — Gradio drops the remainder of the
# sheet when it encounters one, silently discarding later rules.
#
# Do not try to remove the results table's inner scrollbar. The component
# virtualises rows against its scroll container, so both a huge max_height and a
# CSS height:auto override stop most rows from rendering at all.
CSS = """
:root, .gradio-container { font-family: "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace !important; }
.gradio-container { max-width: 100% !important; padding: 1.2em 1.6em !important; }
h1 { font-weight: 600 !important; letter-spacing: -0.01em; margin-bottom: 0.5em !important; }
.gr-button { border-radius: 2px !important; }
input, textarea, select { border-radius: 2px !important; }
#results .results-table { width: 100%; border-collapse: collapse; table-layout: fixed;
  font-size: 1em; margin-top: 0.4em; }
#results .results-table th, #results .results-table td { text-indent: 0 !important; }
#results .results-table th { text-align: left; font-weight: 600; opacity: 0.75;
  padding: 0.45em 0.5em; border-bottom: 1px solid rgba(128,128,128,0.45); }
#results .results-table td { vertical-align: top; padding: 0.4em 0.5em;
  border-bottom: 1px solid rgba(128,128,128,0.18); word-wrap: break-word; }
#results .results-table tr:hover td { background: rgba(128,128,128,0.09); }
#results .results-table a { text-decoration: none; font-weight: 600; display: inline !important;
  text-indent: 0 !important; padding: 0 !important; margin: 0 !important; border: none !important; }
#results .results-table a::before, #results .results-table a::after { content: none !important; display: none !important; }
#results .results-table a:hover { text-decoration: underline; }
#pager { align-items: center; }
#filters-head { align-items: center !important; gap: 0.6em !important; flex-wrap: nowrap !important;
  padding: 0 !important; margin: 0 !important; }
#filters-head > * { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
#clear-filters { background: none !important; border: none !important; box-shadow: none !important;
  text-decoration: underline; opacity: 0.55; padding: 0 !important; min-width: 0 !important;
  font-size: 0.85em !important; position: relative; top: 0.18em; }
#clear-filters:hover { opacity: 1; }
/* Darker than the block body so a header does not read as an editable field. */
.block-header { padding: 0.35em 0 0.35em 0.7em !important; margin: 0 !important; opacity: 0.9;
  background: rgba(0,0,0,0.16) !important; border: none !important; border-radius: 0 !important;
  letter-spacing: 0.02em; }
#filters-head { background: rgba(0,0,0,0.16) !important; }
#filters-head .block-header { background: none !important; }
/* Padding in rem, not em: em would be relative to this element's own font
   size, so changing the text size would silently shift the indent too. */
/* Page background, not the group's fill, so the line does not read as an input.
   Uses the theme variable so it stays correct in dark mode too. */
.filter-hint { padding: 0.5em 0 0.5em 0.62rem !important; margin: 0 !important; opacity: 0.5;
  background: var(--body-background-fill, #fff) !important; }
/* Size the text only, never the wrapper too — em on both compounds. */
.filter-hint p, .filter-hint span { font-size: 0.92em !important; line-height: 1.3 !important; }
footer { display: none !important; }
"""


def _load_pipeline_module():
    """Import 06_run_recommender.py by path — its name is not a valid identifier."""
    path = Path(__file__).resolve().parent / "scripts" / "06_run_recommender.py"
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

SYSTEM_FREQ = clean_frequencies(GAME_DOCS, "system_clean")
TAG_FREQ = clean_frequencies(GAME_DOCS, "tags_clean")


# ---------------------------------------------------------------------------
# Picker choices — labels users recognise, IDs behind them
# ---------------------------------------------------------------------------

def _filter_choices():
    """
    Values for the filter dropdowns, most common first, canonically cased.

    Everything is offered, not a popular subset: filters match the original IFDB
    values, and a truncated list silently hides real ones — "XYZZY Best Game"
    (28 games) sat below a top-400 cutoff even though people search for exactly
    that. These are typeaheads, so length costs nothing but payload.
    """
    genres = [g for g, _ in sorted(GENRE_CASING.items(), key=lambda kv: -kv[1][1])]
    systems = [s for s, _ in sorted(SYSTEM_CASING.items(), key=lambda kv: -kv[1][1])]
    authors = [a for a, _ in sorted(AUTHOR_CASING.items(), key=lambda kv: -kv[1][1])]
    tags = [t for t, _ in sorted(TAG_CASING.items(), key=lambda kv: -kv[1][1])]
    present = {int(y) for y in GAME_DOCS["year"] if str(y).strip().isdigit()}
    # A continuous span, not just the years that happen to occur: the data has no
    # 1968, 1969 or 1971-1976, and a dropdown that skips them looks broken.
    years = list(range(max(present), min(present) - 1, -1))
    # No "any" sentinel: these are multiselect, so empty already means any — and
    # single-select could not accept a typed fragment, which is why they are
    # multiselect at all.
    return (
        [(GENRE_CASING[g][0], GENRE_CASING[g][0]) for g in genres],
        [(SYSTEM_CASING[s][0], SYSTEM_CASING[s][0]) for s in systems],
        [(AUTHOR_CASING[a][0], AUTHOR_CASING[a][0]) for a in authors],
        [TAG_CASING[t][0] for t in tags],
        years,
    )


GAME_CHOICES = game_choices(GAME_DOCS)
AUTHOR_CHOICES = author_choices(AUTHOR_PROFILES)
USER_CHOICES = reviewer_choices(PROFILE_MAP, USER_NAME_MAP, REVIEWS_DF)
SYSTEM_CHOICES, TAG_CHOICES = vocab_choices(GAME_DOCS)
(GENRE_FILTER_CHOICES, SYSTEM_FILTER_CHOICES, AUTHOR_FILTER_CHOICES,
 TAG_FILTER_CHOICES, YEAR_CHOICES) = _filter_choices()
YEAR_MAX, YEAR_MIN = YEAR_CHOICES[0], YEAR_CHOICES[-1]
# The value each filter holds when it is filtering nothing. Defined once so the
# initial state and the reset button cannot drift apart.
FILTER_DEFAULTS = ([], [], [], [], 0, 0, YEAR_MIN, YEAR_MAX)


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

    n_retrieved = len(candidates)
    if RETR.get("prefilter_by_tag", True):
        from src.pipeline.retriever import filter_by_tag_overlap
        _, query_tags = parse_profile_text(query_text)
        if query_tags:
            candidates = filter_by_tag_overlap(candidates, GAME_INFO_MAP, set(query_tags))
    n_filtered = len(candidates)

    cap = RETR.get("rerank_pool_cap", 0)
    if cap and len(candidates) > cap:
        candidates = candidates[:cap]

    # One line per live scoring, because pairs scored is what live latency is
    # made of — roughly 0.8 s plus a pair count over the host's throughput. It
    # also makes a misconfigured deployment visible from the journal: a cap that
    # is silently zero looks exactly like a cap that is working until you notice
    # the scored count running past it.
    logger.info("Live scoring: retrieved %d → tag-filtered %d → scoring %d (cap %s)",
                n_retrieved, n_filtered, len(candidates), cap or "none")

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


def _table_update(frame):
    """
    Render a page of results as a plain HTML table.

    Deliberately not `gr.Dataframe`: that component virtualises rows against its
    own scroll container, so it always keeps an inner scrollbar — and any attempt
    to remove one (a huge max_height, or a CSS height override) stops most rows
    from rendering at all. Plain HTML grows with the page, so the mouse wheel
    scrolls the page rather than a frame inside it.

    The cost is losing the built-in column sorting.
    """
    if frame.empty:
        return ""
    cols = "".join(f'<col style="width:{w}">' for w in COLUMN_WIDTHS)
    head = "".join(f"<th>{escape(c)}</th>" for c in RESULT_COLUMNS)
    body = []
    for row in frame.itertuples(index=False):
        cells = []
        for name, value in zip(RESULT_COLUMNS, row):
            # `title` arrives pre-built as a link; everything else is escaped text.
            cells.append(f"<td>{value if name == 'title' else escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<table class="results-table"><colgroup>{cols}</colgroup>'
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>")


def _page_table(results, relevance, page, per_page):
    start = page * per_page
    rows = []
    for rank, (gid, score) in enumerate(results[start : start + per_page], start=start + 1):
        row = META.loc[gid].to_dict() if gid in META.index else {}
        rel = relevance.get(gid)
        rows.append([
            rank,
            f"{score:.2f}",
            "–" if rel is None else f"{rel:.2f}",
            pipeline._rating_cell(row),
            (f'<a href="{IFDB_GAME_URL.format(gameid=gid)}" target="_blank" '
             f'rel="noopener">{escape(str(row.get("title", gid)))}</a>'),
            _clip(row.get("author", ""), CLIP["author"]),
            str(row.get("year", "")),
            _clip(row.get("system_display", row.get("system", "")), CLIP["system"]),
            _clip(row.get("genre_display", row.get("genre", "")), CLIP["genre"]),
            _clip(row.get("tags_display", row.get("tags", "")), CLIP["tags"]),
        ])
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _profile_display(query_text, corpus_order=False):
    """Shared renderer; corpus order for game/vibe, stored order otherwise."""
    freq = (SYSTEM_FREQ, TAG_FREQ) if corpus_order else (None, None)
    return profile_display(query_text, *freq)


def _summary(headline, query_text, n_results, page, per_page, corpus_order=False):
    """
    The line above the results: what was asked for, the profile it resolved to,
    and how much came back.

    The profile is shown for every mode, not just `vibe`, because it is the
    actual query in all of them. Phrasing it as "games like X: <profile>" keeps
    it reading as a description of what is being looked for rather than a set of
    constraints the results all satisfy.
    """
    first = page * per_page + 1
    last = min(n_results, (page + 1) * per_page)
    profile = _profile_display(query_text, corpus_order)
    line = f"{headline}: `{profile}`" if profile else headline
    return f"{line}\n\n**{n_results} results** · showing {first}–{last}"


def _summary_for(state):
    """Rebuild the summary from state, so paging updates the shown range."""
    return _summary(state.get("headline", ""), state.get("query_text", ""),
                    len(state["results"]), state["page"], _as_int(state["per_page"], 25),
                    state.get("corpus_order", False))


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
            return _table_update(blank), "Pick a game to get recommendations like it.", empty_state, ""
        query_text = GAME_QUERY_TEXT_MAP.get(game, DOC_MAP.get(game, ""))
        cached, exclude = PRE_GAME.get(game), {game}
        emb = None if cached is not None else RETRIEVER._encode_game_ids([game])
        note = f"games like **{META.loc[game, 'title']}**"

    elif mode == "author":
        if not author:
            return _table_update(blank), "Pick an author.", empty_state, ""
        query_text = AUTHOR_PROFILE_MAP.get(author, "")
        cached = PRE_AUTHOR.get(author)
        exclude = set(AUTHOR_GAMES.get(author, []))
        emb = None if cached is not None else QUERY_ENCODER.encode([query_text], normalize_embeddings=True)[0]
        note = f"in the spirit of **{AUTHOR_NAME_MAP.get(author, author)}** (their own games excluded)"

    elif mode == "reviewer":
        if not user:
            return _table_update(blank), "Pick a reviewer.", empty_state, ""
        query_text = PROFILE_MAP.get(user, "")
        cached = PRE_USER.get(user)
        if REVIEWS_DF is not None:
            exclude = set(REVIEWS_DF[REVIEWS_DF["userid"] == user]["gameid"])
        if PLAYEDGAMES_DF is not None:
            exclude |= set(PLAYEDGAMES_DF[PLAYEDGAMES_DF["userid"] == user]["gameid"])
        emb = None if cached is not None else RETRIEVER._encode_userid(user)
        note = f"for **{USER_NAME_MAP.get(user, user)}** (games they've rated or played excluded)"

    else:  # vibe
        if not systems and not tags:
            return _table_update(blank), "Pick at least one system or tag.", empty_state, ""
        query_text = format_profile_text(list(systems or []), list(tags or []))
        emb = QUERY_ENCODER.encode([query_text], normalize_embeddings=True)[0]
        note = "games matching this vibe"

    if not query_text:
        return _table_update(blank), "No profile available for that selection.", empty_state, ""

    # Scoring is the expensive half and depends only on the query, never on the
    # filters. Reuse it while the user narrows results, and drop it as soon as
    # they change what they are searching for.
    query_key = (mode, game, author, user, tuple(systems or []), tuple(tags or []))
    reused = bool(state) and state.get("query_key") == query_key and state.get("scored")
    logger.info("query mode=%s systems=%s tags=%s | previous_key=%s | reused=%s",
                mode, systems, tags, (state or {}).get("query_key"), bool(reused))
    if reused:
        scored, relevance = state["scored"], state["relevance"]

    elif mode == "vibe":
        scored, relevance = _score_browse(tuple(systems or []), tuple(tags or []))
    else:
        scored, relevance = _rank(query_text, emb, exclude, cached)
    if not scored:
        return _table_update(blank), "Nothing above the retrieval threshold for that query.", empty_state, ""

    targets = pipeline._parse_profile_targets(query_text) if mode != "vibe" else (set(), set())
    # Ask for every result the pool can yield, then paginate locally.
    results = select_results(
        scored, hard_filters, GAME_INFO_MAP, len(scored),
        use_diversity=RETR.get("use_diversity", True),
        target_genres=targets[0], target_systems=targets[1],
    )
    if not results:
        return _table_update(blank), f"{note}\n\nNo results match those filters — try relaxing them.", empty_state, ""

    state = {"results": results, "scored": scored, "relevance": relevance,
             "query_key": query_key, "page": 0, "per_page": per_page,
             "headline": note, "query_text": query_text,
             "corpus_order": mode in ("game", "vibe")}
    summary = _summary(note, query_text, len(results), 0, per_page,
                       corpus_order=mode in ("game", "vibe"))
    return _table_update(_page_table(results, relevance, 0, per_page)), summary, state, _pager_text(state)


def turn_page(state, step):
    if not state or not state["results"]:
        return _table_update(pd.DataFrame(columns=RESULT_COLUMNS)), "", state, ""
    per_page = _as_int(state["per_page"], 25)
    pages = max(1, -(-len(state["results"]) // per_page))
    state = {**state, "per_page": per_page, "page": min(max(state["page"] + step, 0), pages - 1)}
    table = _page_table(state["results"], state["relevance"], state["page"], per_page)
    return _table_update(table), _summary_for(state), state, _pager_text(state)


def resize_page(state, per_page):
    per_page = _as_int(per_page, 25)
    if not state or not state["results"]:
        return _table_update(pd.DataFrame(columns=RESULT_COLUMNS)), "", {**(state or {}), "per_page": per_page}, ""
    state = {**state, "per_page": per_page, "page": 0}
    table = _page_table(state["results"], state["relevance"], 0, per_page)
    return _table_update(table), _summary_for(state), state, _pager_text(state)


def _visibility(mode):
    return [gr.update(visible=(mode == m)) for m in ("game", "author", "reviewer", "vibe", "vibe")]


def build_ui():
    with gr.Blocks(title="IFDB recs") as demo:
        gr.Markdown("# IFDB recs")
        state = gr.State({"results": [], "scored": [], "relevance": {},
                          "query_key": None, "page": 0, "per_page": 25})

        with gr.Group():
            gr.Markdown("search mode", elem_classes="block-header")
            mode = gr.Radio(MODES, value="game", show_label=False, interactive=True)
            game = gr.Dropdown(GAME_CHOICES, value=None, label="game",
                               info=BIG_HINT.format(n=round(len(GAME_CHOICES) / 1000)),
                               filterable=True, visible=True)
            author = gr.Dropdown(AUTHOR_CHOICES, value=None, label="author",
                                 info=BIG_HINT.format(n=round(len(AUTHOR_CHOICES) / 1000)),
                                 filterable=True, visible=False)
            user = gr.Dropdown(USER_CHOICES, value=None, label="reviewer",
                               info=BIG_HINT.format(n=round(len(USER_CHOICES) / 1000)),
                               filterable=True, visible=False)
            systems = gr.Dropdown(SYSTEM_CHOICES, value=[], label="systems", info=PICK_HINT,
                                  multiselect=True, visible=False)
            tags = gr.Dropdown(TAG_CHOICES, value=[], label="tags", info=PICK_HINT,
                               multiselect=True, visible=False)

        with gr.Group():
            with gr.Row(elem_id="filters-head"):
                gr.Markdown("result filters", elem_classes="block-header")
                clear_filters = gr.Button("( clear )", size="sm", elem_id="clear-filters")
            gr.Markdown(FREE_TEXT_HINT, elem_classes="filter-hint")
            with gr.Row():
                # Every filter defaults to a no-op, so an untouched block filters
                # nothing and each control can be returned to that state.
                # allow_custom_value lets someone type a fragment and press
                # return — "xyzzy" or "inform" then matches every value
                # containing it, since apply_hard_filters compares by substring.
                # Without it the dropdowns would only ever match one exact value.
                f_genre = gr.Dropdown(GENRE_FILTER_CHOICES, value=FILTER_DEFAULTS[0], label="genre",
                                      multiselect=True,
                                      filterable=True, allow_custom_value=True)
                f_system = gr.Dropdown(SYSTEM_FILTER_CHOICES, value=FILTER_DEFAULTS[1], label="system",
                                       multiselect=True,
                                       filterable=True, allow_custom_value=True)
                f_author = gr.Dropdown(AUTHOR_FILTER_CHOICES, value=FILTER_DEFAULTS[2], label="author",
                                       multiselect=True,
                                       filterable=True, allow_custom_value=True)
                f_tags = gr.Dropdown(TAG_FILTER_CHOICES, value=FILTER_DEFAULTS[3], label="tags",
                                     multiselect=True,
                                     filterable=True, allow_custom_value=True)
            with gr.Row():
                f_rating = gr.Dropdown(RATING_CHOICES, value=FILTER_DEFAULTS[4], label="rating ≥")
                f_count = gr.Dropdown(RATING_COUNT_CHOICES, value=FILTER_DEFAULTS[5], label="rating count ≥")
                f_year_from = gr.Dropdown(YEAR_CHOICES, value=FILTER_DEFAULTS[6], label="year ≥")
                f_year_to = gr.Dropdown(YEAR_CHOICES, value=FILTER_DEFAULTS[7], label="year ≤")

        with gr.Row():
            per_page = gr.Dropdown(PAGE_SIZES, value=25, label="results per page", scale=1)
            go = gr.Button("recommend", variant="primary", scale=3)

        note = gr.Markdown(elem_id="summary")
        table = gr.HTML(
            elem_id="results",
        )
        with gr.Row(elem_id="pager"):
            prev = gr.Button("◀ prev", scale=1)
            pager = gr.Markdown()
            nxt = gr.Button("next ▶", scale=1)

        filter_controls = [f_genre, f_system, f_author, f_tags,
                           f_rating, f_count, f_year_from, f_year_to]
        # Resets the controls only; the results on screen stay until the user
        # asks for them again, so nothing changes under them unexpectedly.
        clear_filters.click(lambda: FILTER_DEFAULTS, None, filter_controls)

        mode.change(_visibility, mode, [game, author, user, systems, tags])
        inputs = [state, mode, game, author, user, systems, tags,
                  f_genre, f_system, f_author, f_tags, f_rating, f_count,
                  f_year_from, f_year_to, per_page]
        # One at a time: two CPUs shared between concurrent requests makes
        # everyone slow, whereas a queue makes the wait visible.
        go.click(recommend, inputs, [table, note, state, pager], concurrency_limit=1)
        # Paging lands the reader back at the summary, not stranded at the
        # bottom of the previous page.
        prev.click(lambda s: turn_page(s, -1), state, [table, note, state, pager]).then(
            None, None, None, js=SCROLL_TO_SUMMARY)
        nxt.click(lambda s: turn_page(s, +1), state, [table, note, state, pager]).then(
            None, None, None, js=SCROLL_TO_SUMMARY)
        per_page.change(resize_page, [state, per_page], [table, note, state, pager])
    return demo


if __name__ == "__main__":
    # Gradio 6 takes theme and css at launch(), not on the Blocks constructor —
    # passing them to Blocks is accepted with a warning and then ignored.
    build_ui().queue().launch(theme=gr.themes.Monochrome(), css=CSS)
