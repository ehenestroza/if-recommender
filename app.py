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
from datetime import date
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
from src.pipeline.ranker import order_by_relevance, select_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

IFDB_GAME_URL = "https://ifdb.org/viewgame?id={gameid}"
# "vibe" rather than "browse": the systems and tags are turned into a profile
# and matched semantically, not used as exact filters, and the name should not
# promise the latter.
MODES = ["game", "author", "reviewer", "vibe"]

# Field order within a card. Also the DataFrame's column order, so the two
# cannot drift. There is no column-width table any more: results render as
# cards at every width, so nothing is apportioning a fixed share of a row.
#
# Ordered the way a scanning eye needs it, and grouped by kind: identity
# (title, author, year), then signals (relevance, then rating), then
# classification (system, genre), then the two prose blocks last. Prose reads
# slowest, so putting it above the cheap identifying facts stalls the eye on
# every card; keeping description and tags adjacent also gives the fixed-height
# card a consistent rhythm — six short rows, then two text blocks.
#
# The blended score is deliberately absent. It answered "match plus quality" in
# one number that nobody could decompose, and ordering by it pushed already
# well-loved games up — the opposite of what a discovery tool is for. Relevance
# alone is what a reader can act on, and rating is right there beside it for
# anyone who wants to weigh it themselves.
RESULT_COLUMNS = ["#", "title", "author", "year", "relevance", "rating",
                  "system", "genre", "description", "tags"]
# Even sizes only, and no 25. Results are two-up above 1024px, so an odd page
# size leaves a lone card in the last row with an empty slot beside it — which
# reads as "that is all there is" even when more pages follow.
PAGE_SIZES = [10, 20, 50]
DEFAULT_PAGE_SIZE = 20
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
# The big pickers ship every option to the browser, so the first open costs a
# moment of client-side rendering. That is browser work, not server work — it
# does not compete with scoring.
#
# "type to search" rather than "start typing to search": the hint sits under a
# label on a phone, where the saved characters are the difference between one
# line and two.
BIG_HINT = "type to search · {n}K options, first open takes a second"
PICK_HINT = "choose one or more"
# The four text-valued filters all accept a typed fragment and match by
# substring, so they carry the same hint — documenting it on only one would
# imply the others behave differently.
FREE_TEXT_HINT = "type text fragments and press return · case insensitive"
RATING_CHOICES = [round(0.5 * i, 1) for i in range(10)]      # 0.0 … 4.5
RATING_COUNT_CHOICES = [0, 1, 2, 5, 10, 25, 50]

REPO_URL = "https://github.com/ehenestroza/if-recommender"
LICENSE_URL = f"{REPO_URL}/blob/main/LICENSE"
IFDB_URL = "https://ifdb.org"
IFARCHIVE_URL = "https://ifarchive.org/indexes/if-archive/info/ifdb/"
DATA_LICENSE_URL = "https://creativecommons.org/licenses/by/3.0/us/"

# The IFDB dump this was built from. IFDB publishes to the IF Archive roughly
# quarterly, so refreshing the data means re-running the pipeline and moving
# this date. Kept as an ISO string and formatted for display, so updating it is
# one unambiguous edit and no reader has to guess whether 6/1 is June or
# January. `data/manifest.json` deliberately carries no timestamp — it records
# what the data *is*, not when it was taken — so this cannot be derived.
DATA_THROUGH = "2026-06-01"


def _link(href: str, text: str) -> str:
    return f'<a href="{href}" target="_blank" rel="noopener">{text}</a>'


def _data_through(iso: str) -> str:
    """ISO date to "1 June 2026". Day is unpadded; %-d is not portable."""
    day = date.fromisoformat(iso)
    return f"{day.day} {day.strftime('%B')} {day.year}"


# Two lines: who made it, and where the data came from. The second is not
# decoration — IFDB content is CC BY 3.0 US, which permits these extracts and
# the models trained on them precisely on condition that IFDB is credited, so
# the attribution belongs somewhere a reader actually sees it rather than only
# in the repository. Rendered with sanitize_html=False so target/rel survive;
# safe because every part is a literal constant.
FOOTER_HTML = (
    f'by Enrique · {_link(REPO_URL, "source on GitHub")}'
    f' · {_link(LICENSE_URL, "MIT licence")}<br>'
    f'game data from {_link(IFDB_URL, "IFDB")}'
    f', via the {_link(IFARCHIVE_URL, "IF Archive")}'
    f' · through {_data_through(DATA_THROUGH)}'
    f' · used under {_link(DATA_LICENSE_URL, "CC BY 3.0 US")}'
)

# NOTE: keep this free of /* */ comments — Gradio drops the remainder of the
# sheet when it encounters one, silently discarding later rules.
#
# Do not try to remove the results table's inner scrollbar. The component
# virtualises rows against its scroll container, so both a huge max_height and a
# CSS height:auto override stop most rows from rendering at all.
#
# Results are cards at every width — there is no tabular mode. Ten columns only
# ever fitted a maximised laptop window, and every width below that spent its
# budget wrapping each value onto three lines. A card gives every field a whole
# line and lets the page grow downwards instead, which is the axis a browser has
# to spare.
#
# Width buys *columns of cards* rather than narrower cells: one up to 1024px,
# two above. The grid lives on `tbody`, so ordering is the browser's default
# row-major flow — result 1 top-left, result 2 to its right — and rank stays
# readable left-to-right the way a numbered list should.
#
# Two columns is the ceiling because the page itself is capped at 1280px. A
# third column would not widen anything: it would spend that same 1280px on
# narrower cards (596px → 387px, about phone width), so maximising the window
# would make each result *smaller*. The cap and the column count have to be
# decided together — raise `max-width` and a third column becomes worth it
# again.
#
# Card height is bounded by clamping fields rather than by fixing a height:
# every value is one line with an ellipsis, except tags, which get five. The
# clamping is CSS (`text-overflow` and `-webkit-line-clamp`) rather than
# truncating the string in Python, because a card is 366px on a phone and 596px
# on a desktop — a character budget that fits one is wrong for the other, and
# would trail off mid-line while empty space sat to the right.
#
# That needs each value wrapped in a `span.v`: the value would otherwise be a
# bare text node, which is an anonymous grid item and cannot take overflow or a
# line clamp.
#
# Cards are then a fixed height rather than merely bounded: the tags value
# reserves its full five lines (`height: 7.5em`, five times the 1.5 line-height)
# whether or not it needs them, and `.v { min-height: 1.5em }` keeps a field
# with no value occupying its line. Every card therefore has the same fields on
# the same lines, so scanning across a row compares like with like — which is
# worth more than the blank space it costs on sparse games.
#
# The remaining @media block is 768px, and it is about touch rather than
# results: bigger tap targets, 16px inputs, filters two-up, tighter padding.
# Keeping it separate from the card breakpoints is what lets an intermediate
# window get desktop controls with a one- or two-column card grid.
#
# Details that are load-bearing:
#   * the 1fr track is minmax(0, 1fr) — a bare 1fr floors at max-content, so a
#     long tag list overflows the card instead of wrapping inside it
#   * every field is emitted even when empty, and reserves its line, because
#     alignment across a row depends on it
#   * 16px inputs, below which iOS Safari zooms on focus and stays zoomed
#   * `order` pulls rank and title ahead of the relevance and rating fields
#     that precede them in the DOM, so the card leads with what identifies
#     the game rather than with its numbers
#
# The filter selectors go through `.control-row > * > *` rather than the more
# obvious `.control-row > *`. Gradio wraps consecutive form components in an
# implicit `form` component, so a row of four dropdowns has exactly one child —
# sizing that child to 50% squeezes all four filters into half the row instead
# of giving each a half. The extra level is the wrapper, not decoration.
#
# min-width:0 on their descendants because Gradio's own component CSS carries
# min-widths up to 200px; two of those in one row exceed a 360px phone.
#
# Do not put a `gap` on that form when the filters wrap. A Gradio group paints
# the divider colour as its own background and lets opaque children sit on top,
# so the hairlines between filters are really 1px of group background showing
# through. Widening the gap widens those lines: a 0.5em gap rendered as 7px
# bands between the filter cells at 560px, which read as a rendering fault.
# Gradio's own 1px gap is the hairline — leave it alone and size the children
# to `calc(50% - 0.5px)` so two of them plus the gap come to exactly 100%.
CSS = """
:root, .gradio-container { font-family: "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace !important; }
.gradio-container { max-width: 1280px !important; margin: 0 auto !important;
  padding: 1.2em clamp(0.25rem, 1.5vw, 1.6em) !important; box-sizing: border-box !important;
  --block-radius: 10px; --block-border-width: 1px;
  --block-border-color: rgba(128,128,128,0.16);
  --block-shadow: none; --block-label-shadow: none;
  --border-color-primary: rgba(128,128,128,0.07);
  --border-color-secondary: rgba(128,128,128,0.07);
  --input-radius: 8px; --input-border-color: rgba(128,128,128,0.22);
  --button-large-radius: 8px; --button-small-radius: 6px;
  --button-primary-shadow: none; --button-secondary-shadow: none;
  --app-link: #2563eb; --app-link-visited: #7c3aed; }
.dark .gradio-container { --app-link: #7aa7ff; --app-link-visited: #cba6ff; }
.gradio-container .app { padding-left: clamp(0.25rem, 1.5vw, var(--size-8)) !important;
  padding-right: clamp(0.25rem, 1.5vw, var(--size-8)) !important;
  max-width: 100% !important; }
h1 { font-weight: 600 !important; letter-spacing: -0.01em; margin-bottom: 0.6em !important; }
#results { border: none !important; background: none !important; box-shadow: none !important;
  padding: 0 !important; }
#results .html-container { padding: 0 !important; }
#results .results-table { width: 100%; display: block; font-size: 1em; margin-top: 0.6em;
  border: none !important; }
#results .results-table td { text-indent: 0 !important; }
#results .results-table tbody { display: grid; gap: 0.75em; align-items: stretch;
  grid-template-columns: minmax(0, 1fr); }
#results .results-table tr { display: flex; flex-wrap: wrap; align-items: baseline;
  border: 1px solid rgba(128,128,128,0.18); border-radius: 10px;
  padding: 0.7em 0.85em; margin: 0; }
#results .results-table tr:hover { border-color: rgba(128,128,128,0.34); }
#results .results-table td { display: grid; grid-template-columns: 6rem minmax(0, 1fr);
  gap: 0.5em; flex: 1 1 100%; min-width: 0; border: none; padding: 0.18em 0;
  line-height: 1.5; overflow-wrap: break-word; }
#results .results-table td::before { content: attr(data-label); opacity: 0.45;
  font-size: 0.85em; letter-spacing: 0.04em; }
#results .results-table td .v { display: block; min-width: 0; min-height: 1.5em;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#results .results-table td[data-label="tags"] .v { white-space: normal;
  display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 5;
  line-clamp: 5; height: 7.5em; }
#results .results-table td[data-label="description"] .v { white-space: normal;
  display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2;
  line-clamp: 2; height: 3em; }
#results .results-table td[data-label="#"],
#results .results-table td[data-label="title"] {
  display: block; flex: 0 1 auto; padding-bottom: 0.3em; }
#results .results-table td[data-label="#"]::before,
#results .results-table td[data-label="title"]::before { content: none; }
#results .results-table td[data-label="#"] { order: -2; margin-right: 0.55em;
  font-weight: 700 !important; color: var(--body-text-color) !important; }
#results .results-table td[data-label="title"] { order: -1; flex: 1 1 0%; font-size: 1.05em;
  min-width: 0; overflow: hidden; }
@media (min-width: 1024px) {
  #results .results-table tbody { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.85em; }
}
#results .results-table a { text-decoration: none; font-weight: 600; display: inline !important;
  text-indent: 0 !important; padding: 0 !important; margin: 0 !important; border: none !important;
  color: var(--app-link) !important; }
#results .results-table a:visited { color: var(--app-link-visited) !important; }
#results .results-table a::before, #results .results-table a::after { content: none !important; display: none !important; }
#results .results-table a:hover { text-decoration: underline; }
#pager { align-items: center; }
#filters-head { align-items: center !important; gap: 0.6em !important; flex-wrap: nowrap !important;
  padding: 0 !important; margin: 0 !important; }
#filters-head > * { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
#reset-filters { background: none !important; border: none !important; box-shadow: none !important;
  text-decoration: underline; opacity: 0.55; padding: 0 !important; min-width: 0 !important;
  font-size: 0.9em !important; position: relative; top: 0.05em; }
#reset-filters:hover { opacity: 1; }
/* Darker than the block body so a header does not read as an editable field. */
.block-header { padding: 0.65em 0 0.6em 0.85em !important; margin: 0 !important;
  background: var(--block-background-fill) !important; border: none !important;
  border-bottom: 1px solid rgba(128,128,128,0.16) !important;
  border-radius: 10px 10px 0 0 !important; letter-spacing: 0.03em; }
.block-header p, .block-header span { font-size: 1.05em !important;
  color: var(--body-text-color) !important; font-weight: 500 !important; }
.block-header .block-header { border-bottom: none !important; padding: 0 !important;
  background: none !important; }
#filters-head { background: var(--block-background-fill) !important;
  border-radius: 10px 10px 0 0 !important;
  border-bottom: 1px solid rgba(128,128,128,0.16) !important; }
#filters-head .block-header { background: none !important; border-bottom: none !important;
  border-radius: 0 !important; }
/* Padding in rem, not em: em would be relative to this element's own font
   size, so changing the text size would silently shift the indent too. */
/* Page background, not the group's fill, so the line does not read as an input.
   Uses the theme variable so it stays correct in dark mode too. */
.filter-hint { padding: 0.55em 0 0.6em 1rem !important; margin: 0 !important;
  background: var(--block-background-fill) !important; }
.filter-hint .filter-hint { padding: 0 !important; background: none !important; }
/* Size the text only, never the wrapper too — em on both compounds. */
.filter-hint p, .filter-hint span { font-size: 0.92em !important; line-height: 1.3 !important;
  color: var(--block-info-text-color) !important; }
#summary:has(p) { position: relative; margin: 2.2em 0 0.2em !important;
  padding: 0.9em 1.05em !important; border-radius: 10px !important;
  background: linear-gradient(rgba(128,128,128,0.10), rgba(128,128,128,0.10)),
    var(--block-background-fill) !important;
  border: 1px solid rgba(128,128,128,0.14) !important; }
#summary:has(p)::before { content: ""; position: absolute; left: 0; right: 0; top: -1.15em;
  border-top: 1px solid rgba(128,128,128,0.18); }
#summary p { margin: 0.2em 0 !important; }
#summary code { background: var(--block-background-fill) !important;
  border: 1px solid rgba(128,128,128,0.18) !important; }
#result-count { margin: 1.4em 0 0 !important; }
#result-count p { margin: 0 !important; font-size: 0.92em !important; opacity: 0.6; }
#pager:not(:has(.md p)) { display: none !important; }
footer { display: none !important; }
#page-footer { margin-top: 2.2em !important; padding: 1em 0 0.4em !important;
  border-top: 1px solid rgba(128,128,128,0.14) !important; }
#page-footer p { margin: 0 !important; font-size: 0.9em !important; line-height: 1.7 !important;
  letter-spacing: 0.02em; color: var(--body-text-color-subdued) !important; }
#page-footer a { color: var(--app-link) !important; text-decoration: underline;
  text-underline-offset: 3px; }
#page-footer a:visited { color: var(--app-link-visited) !important; }
#page-footer a:hover { text-decoration-thickness: 2px; }

@media (max-width: 768px) {
  input, textarea, select { font-size: 16px !important; }
  .control-row *, #action-row * { min-width: 0 !important; }
  .control-row, .control-row > * { flex-wrap: wrap !important; }
  .control-row > * {
    display: flex !important; flex: 1 1 100% !important; max-width: 100% !important;
  }
  .control-row > * > * {
    flex: 1 1 calc(50% - 0.5px) !important;
    max-width: calc(50% - 0.5px) !important;
  }
  #action-row { flex-wrap: wrap !important; }
  #action-row > * { flex: 1 1 100% !important; max-width: 100% !important; }
  .gr-button { min-height: 44px !important; }
  #reset-filters { min-height: 0 !important; }
}

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
# What each filter holds on load, and what the reset button restores. Defined
# once so the two cannot drift apart.
#
# Rating and count are the only two that start switched on, and they are the
# counterweight to ordering by relevance alone. A page led by unrated games asks
# the reader to gamble with no information, and 2.x games occupy slots that a
# merely-decent game could have used to spark interest. 3.0 with at least one
# rating clears both without being a quality bar in any real sense — and because
# `min_rating` also drops games nobody has rated, lowering rating to 0.0 is what
# opens the genuinely unknown tail to anyone who wants it.
FILTER_DEFAULTS = ([], [], [], [], 3.0, 1, YEAR_MIN, YEAR_MAX)


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
    body = []
    for row in frame.itertuples(index=False):
        cells = []
        for name, value in zip(RESULT_COLUMNS, row):
            # `title` arrives pre-built as a link; everything else is escaped text.
            # Every field is emitted even when the game has no value for it, so
            # that the same label sits on the same line in every card. A blank
            # value beside a label is the price of that alignment; dropping the
            # row instead would shift every field below it up by a line and put
            # neighbouring cards out of step.
            inner = value if name == "title" else escape(str(value))
            # data-label carries the column name into each cell so the card
            # layout can label it, and the value is wrapped so CSS can clamp it:
            # a bare text node in a grid cell is an anonymous item and cannot be
            # given overflow or a line clamp.
            cells.append(f'<td data-label="{escape(name, quote=True)}">'
                         f'<span class="v">{inner}</span></td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="results-table"><tbody>{"".join(body)}</tbody></table>'


def _page_table(results, relevance, page, per_page):
    start = page * per_page
    rows = []
    for rank, (gid, _value) in enumerate(results[start : start + per_page], start=start + 1):
        row = META.loc[gid].to_dict() if gid in META.index else {}
        rel = relevance.get(gid)
        rows.append([
            f"{rank}.",
            (f'<a href="{IFDB_GAME_URL.format(gameid=gid)}" target="_blank" '
             f'rel="noopener">{escape(str(row.get("title", gid)))}</a>'),
            str(row.get("author", "")),
            str(row.get("year", "")),
            "–" if rel is None else f"{rel:.2f}",
            pipeline._rating_cell(row),
            str(row.get("system_display", row.get("system", ""))),
            str(row.get("genre_display", row.get("genre", ""))),
            str(row.get("description", "")),
            str(row.get("tags_display", row.get("tags", ""))),
        ])
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _profile_display(query_text, corpus_order=False):
    """Shared renderer; corpus order for game/vibe, stored order otherwise."""
    freq = (SYSTEM_FREQ, TAG_FREQ) if corpus_order else (None, None)
    return profile_display(query_text, *freq)


def _summary(headline, query_text, corpus_order=False):
    """
    What was asked for and the profile it resolved to.

    The profile is shown for every mode, not just `vibe`, because it is the
    actual query in all of them. Phrasing it as "games like X: <profile>" keeps
    it reading as a description of what is being looked for rather than a set of
    constraints the results all satisfy.

    The count of results lives in `_result_count`, not here: this describes the
    *query*, and sits in its own panel above the divider, while the count
    belongs to the *results* and sits directly on top of them.
    """
    profile = _profile_display(query_text, corpus_order)
    return f"{headline}: `{profile}`" if profile else headline


def _result_count(n_results, page, per_page):
    """The tally that sits immediately above the cards it is counting."""
    if not n_results:
        return ""
    first = page * per_page + 1
    last = min(n_results, (page + 1) * per_page)
    return f"**{n_results} results** · showing {first}–{last}"


def _count_for(state):
    """Rebuild the count from state, so paging updates the shown range."""
    return _result_count(len(state["results"]), state["page"],
                         _as_int(state["per_page"], DEFAULT_PAGE_SIZE))


def _summary_for(state):
    """Rebuild the summary from state."""
    return _summary(state.get("headline", ""), state.get("query_text", ""),
                    state.get("corpus_order", False))


def _pager_text(state):
    total = len(state["results"])
    if not total:
        return ""
    pages = max(1, -(-total // state["per_page"]))
    return f"page {state['page'] + 1} of {pages}  ·  {total} results"


def _pager_buttons(state):
    """
    Enable prev/next only where there is somewhere to go.

    Without this a click at either end still fires the handler, which re-renders
    the same page and runs the scroll-to-summary that follows it — so the button
    looks like it did something, and what it did was jump you to the top of the
    page you were already on.
    """
    total = len(state["results"]) if state else 0
    if not total:
        return gr.update(interactive=False), gr.update(interactive=False)
    per_page = _as_int(state["per_page"], DEFAULT_PAGE_SIZE)
    last = max(1, -(-total // per_page)) - 1
    page = state["page"]
    return gr.update(interactive=page > 0), gr.update(interactive=page < last)


NO_PAGES = (gr.update(interactive=False), gr.update(interactive=False))


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
    per_page = _as_int(per_page, DEFAULT_PAGE_SIZE)
    hard_filters = _build_filters(f_genre, f_system, f_author, f_tags,
                                  f_rating, f_count, f_year_from, f_year_to)
    exclude, cached, emb, query_text = set(), None, None, ""

    if mode == "game":
        if not game:
            return _table_update(blank), "Pick a game to get recommendations like it.", "", empty_state, "", *NO_PAGES
        query_text = GAME_QUERY_TEXT_MAP.get(game, DOC_MAP.get(game, ""))
        cached, exclude = PRE_GAME.get(game), {game}
        emb = None if cached is not None else RETRIEVER._encode_game_ids([game])
        note = f"games like **{META.loc[game, 'title']}**"

    elif mode == "author":
        if not author:
            return _table_update(blank), "Pick an author.", "", empty_state, "", *NO_PAGES
        query_text = AUTHOR_PROFILE_MAP.get(author, "")
        cached = PRE_AUTHOR.get(author)
        exclude = set(AUTHOR_GAMES.get(author, []))
        emb = None if cached is not None else QUERY_ENCODER.encode([query_text], normalize_embeddings=True)[0]
        note = f"in the spirit of **{AUTHOR_NAME_MAP.get(author, author)}** (excluding their own games)"

    elif mode == "reviewer":
        if not user:
            return _table_update(blank), "Pick a reviewer.", "", empty_state, "", *NO_PAGES
        query_text = PROFILE_MAP.get(user, "")
        cached = PRE_USER.get(user)
        if REVIEWS_DF is not None:
            exclude = set(REVIEWS_DF[REVIEWS_DF["userid"] == user]["gameid"])
        if PLAYEDGAMES_DF is not None:
            exclude |= set(PLAYEDGAMES_DF[PLAYEDGAMES_DF["userid"] == user]["gameid"])
        emb = None if cached is not None else RETRIEVER._encode_userid(user)
        note = f"for **{USER_NAME_MAP.get(user, user)}** (excluding games they've rated or played)"

    else:  # vibe
        if not systems and not tags:
            return _table_update(blank), "Pick at least one system or tag.", "", empty_state, "", *NO_PAGES
        query_text = format_profile_text(list(systems or []), list(tags or []))
        emb = QUERY_ENCODER.encode([query_text], normalize_embeddings=True)[0]
        note = "games matching this vibe"

    if not query_text:
        return _table_update(blank), "No profile available for that selection.", "", empty_state, "", *NO_PAGES

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
        return _table_update(blank), "Nothing above the retrieval threshold for that query.", "", empty_state, "", *NO_PAGES

    targets = pipeline._parse_profile_targets(query_text) if mode != "vibe" else (set(), set())
    # Ask for every result the pool can yield, then paginate locally.
    results = select_results(
        order_by_relevance(scored, relevance), hard_filters, GAME_INFO_MAP, len(scored),
        use_diversity=RETR.get("use_diversity", True),
        target_genres=targets[0], target_systems=targets[1],
    )
    if not results:
        return _table_update(blank), f"{note}\n\nNo results match those filters — try relaxing them.", "", empty_state, "", *NO_PAGES

    state = {"results": results, "scored": scored, "relevance": relevance,
             "query_key": query_key, "page": 0, "per_page": per_page,
             "headline": note, "query_text": query_text,
             "corpus_order": mode in ("game", "vibe")}
    summary = _summary(note, query_text, corpus_order=mode in ("game", "vibe"))
    return (_table_update(_page_table(results, relevance, 0, per_page)), summary,
            _result_count(len(results), 0, per_page), state, _pager_text(state),
            *_pager_buttons(state))


def turn_page(state, step):
    if not state or not state["results"]:
        return (_table_update(pd.DataFrame(columns=RESULT_COLUMNS)), "", "", state, "",
            *NO_PAGES)
    per_page = _as_int(state["per_page"], DEFAULT_PAGE_SIZE)
    pages = max(1, -(-len(state["results"]) // per_page))
    state = {**state, "per_page": per_page, "page": min(max(state["page"] + step, 0), pages - 1)}
    table = _page_table(state["results"], state["relevance"], state["page"], per_page)
    return (_table_update(table), _summary_for(state), _count_for(state), state,
            _pager_text(state), *_pager_buttons(state))


def resize_page(state, per_page):
    per_page = _as_int(per_page, DEFAULT_PAGE_SIZE)
    if not state or not state["results"]:
        return (_table_update(pd.DataFrame(columns=RESULT_COLUMNS)), "", "",
            {**(state or {}), "per_page": per_page}, "", *NO_PAGES)
    state = {**state, "per_page": per_page, "page": 0}
    table = _page_table(state["results"], state["relevance"], 0, per_page)
    return (_table_update(table), _summary_for(state), _count_for(state), state,
            _pager_text(state), *_pager_buttons(state))


def _visibility(mode):
    return [gr.update(visible=(mode == m)) for m in ("game", "author", "reviewer", "vibe", "vibe")]


def build_ui():
    with gr.Blocks(title="IF recommender") as demo:
        gr.Markdown("# IF recommender")
        state = gr.State({"results": [], "scored": [], "relevance": {},
                          "query_key": None, "page": 0, "per_page": DEFAULT_PAGE_SIZE})

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
                reset_filters = gr.Button("( reset )", size="sm", elem_id="reset-filters")
            gr.Markdown(FREE_TEXT_HINT, elem_classes="filter-hint")
            # elem_classes rather than Gradio's own row class: the internal names
            # are not API and have changed between majors, whereas these are ours.
            with gr.Row(elem_classes="control-row"):
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
            with gr.Row(elem_classes="control-row"):
                f_rating = gr.Dropdown(RATING_CHOICES, value=FILTER_DEFAULTS[4], label="rating ≥")
                f_count = gr.Dropdown(RATING_COUNT_CHOICES, value=FILTER_DEFAULTS[5], label="rating count ≥")
                f_year_from = gr.Dropdown(YEAR_CHOICES, value=FILTER_DEFAULTS[6], label="year ≥")
                f_year_to = gr.Dropdown(YEAR_CHOICES, value=FILTER_DEFAULTS[7], label="year ≤")

        with gr.Row(elem_id="action-row"):
            per_page = gr.Dropdown(PAGE_SIZES, value=DEFAULT_PAGE_SIZE, label="results per page", scale=1)
            go = gr.Button("recommend", variant="primary", scale=3)

        note = gr.Markdown(elem_id="summary")
        count = gr.Markdown(elem_id="result-count")
        table = gr.HTML(
            elem_id="results",
        )
        with gr.Row(elem_id="pager"):
            prev = gr.Button("◀ prev", scale=1, interactive=False)
            pager = gr.Markdown()
            nxt = gr.Button("next ▶", scale=1, interactive=False)

        # A div, not a <footer>: the CSS hides Gradio's own footer by tag name,
        # and a semantic element here would be hidden along with it.
        gr.Markdown(FOOTER_HTML, elem_id="page-footer", sanitize_html=False)

        filter_controls = [f_genre, f_system, f_author, f_tags,
                           f_rating, f_count, f_year_from, f_year_to]
        # Resets the controls only; the results on screen stay until the user
        # asks for them again, so nothing changes under them unexpectedly.
        reset_filters.click(lambda: FILTER_DEFAULTS, None, filter_controls)

        mode.change(_visibility, mode, [game, author, user, systems, tags])
        inputs = [state, mode, game, author, user, systems, tags,
                  f_genre, f_system, f_author, f_tags, f_rating, f_count,
                  f_year_from, f_year_to, per_page]
        # One at a time: two CPUs shared between concurrent requests makes
        # everyone slow, whereas a queue makes the wait visible.
        go.click(recommend, inputs, [table, note, count, state, pager, prev, nxt], concurrency_limit=1)
        # Paging lands the reader back at the summary, not stranded at the
        # bottom of the previous page.
        prev.click(lambda s: turn_page(s, -1), state, [table, note, count, state, pager, prev, nxt]).then(
            None, None, None, js=SCROLL_TO_SUMMARY)
        nxt.click(lambda s: turn_page(s, +1), state, [table, note, count, state, pager, prev, nxt]).then(
            None, None, None, js=SCROLL_TO_SUMMARY)
        per_page.change(resize_page, [state, per_page], [table, note, count, state, pager, prev, nxt])
    return demo


if __name__ == "__main__":
    # Gradio 6 takes theme and css at launch(), not on the Blocks constructor —
    # passing them to Blocks is accepted with a warning and then ignored.
    build_ui().queue().launch(theme=gr.themes.Monochrome(), css=CSS)
