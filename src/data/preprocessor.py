"""Transform raw IFDB parquet tables into training artefacts."""

import logging
import re
from html import unescape
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.columns import clean_col, clean_col_in

logger = logging.getLogger(__name__)

_COMP_RE = re.compile(
    r"(comp)|(award)|(\d\d\d\d)|(transcript)|(xyzzy)|(spring thing)|(source available)|(prize)|(showcase)|(jam)|"
    r"(walkthrough)|(cover art)|(available)|(adrift)|(inform)|(windrift)|(twine)|(interactive fiction)|(top \d\d)|"
    r"(winner)|(playoff)|(inklewriter)|(storynexus)|(choicescript)|(ink)|(student project)", re.IGNORECASE)

_MAX_TAGS = 20


def format_profile_text(systems: List[str], tags: List[str]) -> str:
    """
    Build the query string the encoders were trained on.

    Every query the model sees — user profiles, game profiles, and queries built
    from a UI's system/tag pickers — must use this exact shape, so it lives in one
    place rather than being re-spelled at each call site:

        "Systems: twine, ink. Tags: fantasy, horror"

    Values should be the normalised `_clean` forms; IFDB's own casing
    ("Inform 7", "IFComp 2019") is not what the encoders saw during training.
    """
    parts: List[str] = []
    if systems:
        parts.append(f"Systems: {', '.join(systems)}")
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
    return ". ".join(parts)


# Tags may legitimately contain a slash ("gay/queer protagonist"), so they split
# on commas only. System and genre use both, since IFDB stores values like
# "Drama / Political" and "Ink / HTML5".
TAG_SEPARATORS = re.compile(r",")
# Authors are also joined with " and ".
AUTHOR_SEPARATORS = re.compile(r"\s*/\s*|\s*,\s*|\s+and\s+")
SYSTEM_GENRE_SEPARATORS = re.compile(r"[,/]")


def build_display_map(
    game_docs: pd.DataFrame,
    column: str,
    separators: re.Pattern = TAG_SEPARATORS,
) -> Dict[str, Tuple[str, int]]:
    """
    Map each value in `column` (lowercased) to its dominant casing and game count.

        "ifcomp 2025" -> ("IFComp 2025", 84)
        "dendry"      -> ("Dendry", 41)

    IFDB fields are free text, so the same value appears in many casings and with
    inconsistent separators. Showing whichever form a given game happened to
    store makes a column look ragged; showing the community's dominant casing
    makes it look edited.
    """
    casings: Dict[str, Counter] = {}
    game_counts: Counter = Counter()
    for value in game_docs.get(column, pd.Series(dtype=str)).fillna(""):
        seen_here = set()
        for raw in separators.split(str(value)):
            item = raw.strip()
            if not item:
                continue
            key = item.lower()
            casings.setdefault(key, Counter())[item] += 1
            if key not in seen_here:          # count games, not occurrences
                game_counts[key] += 1
                seen_here.add(key)
    # Tie-break on the casing itself so the choice is stable across runs.
    return {
        key: (min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0], game_counts[key])
        for key, counter in casings.items()
    }


# IFDB descriptions are author-written HTML: about 12% carry <p>, <br>, <i> or
# <b>, and some carry entities. They are display text only — never model input —
# so nothing upstream had reason to clean them, and rendering them escaped would
# show the tags literally.
_MARKUP_BREAK = re.compile(r"<\s*(br|/p|/div|/li|/h[1-6])\s*/?>", re.I)
_MARKUP_TAG = re.compile(r"<[^>]*>")
_WHITESPACE = re.compile(r"\s+")


# IFDB lets the community tag an entry "not a game": tools, indexes, reviews of
# other games, language toys. They are legitimate IFDB records but nothing to
# recommend playing, so they are dropped everywhere at once — results, every
# picker, and the system/tag vocabularies built from the same table.
NOT_A_GAME_TAG = "not a game"


def drop_non_games(game_docs: pd.DataFrame) -> Tuple[pd.DataFrame, set]:
    """
    Split game_docs into the entries worth recommending and the ids to suppress.

    Matches a whole comma-separated tag rather than a substring, so a future tag
    like "not a game jam" would not be swept up, and reads both the original and
    the `_clean` column: the cleaned one folds in genre and caps at 20 tags, so
    a long tag list could drop the marker from one but not the other.

    Returns (kept rows, excluded gameids). The caller needs the id set because
    the FAISS index and the precomputed tables are built offline and still
    contain these games.
    """
    def tagged(value) -> bool:
        return any(part.strip().lower() == NOT_A_GAME_TAG
                   for part in str(value or "").split(","))

    flagged = (game_docs["tags"].map(tagged)
               | game_docs.get("tags_clean", game_docs["tags"]).map(tagged))
    excluded = set(game_docs.loc[flagged, "gameid"])
    return game_docs.loc[~flagged].copy(), excluded


# IFDB writes "no authoring system recorded" as the literal string "None" — 164
# games — and one entry as "N/A". Both are an absence wearing the costume of a
# value: they sat in the vibe picker and the system filter as though they were
# something to search for, and rendered on cards as "None" where every other
# missing field shows an em dash.
SYSTEM_PLACEHOLDERS = {"none", "n/a", "na", "null", "nan", "unknown"}


def strip_placeholder_systems(game_docs: pd.DataFrame) -> pd.DataFrame:
    """
    Blank system values that only say "not recorded".

    Rewrites the original and the `_clean` column together, before anything is
    derived from either: the display map and the system filter are built from
    the first, the vibe vocabulary from the second, so a value left in one would
    reappear in half the interface. Entries are dropped per comma-separated
    part, leaving any real system alongside them intact.

    Deliberately not `drop_non_games`: these are ordinary games that happen to
    have nothing recorded in one field, and dropping them would hide 164 games
    to tidy up a dropdown.
    """
    def kept(value) -> str:
        parts = [p.strip() for p in str(value or "").split(",") if p.strip()]
        return ", ".join(p for p in parts if p.lower() not in SYSTEM_PLACEHOLDERS)

    out = game_docs.copy()
    for column in ("system", clean_col("system")):
        if column in out.columns:
            out[column] = out[column].map(kept)
    return out


# IFDB's language field is free text that has collected ISO codes ("en"),
# regional variants ("en-US", "zh-Hans"), spelled-out names ("English"),
# three-letter forms ("rus"), and multi-language entries ("en, fr"). Readers want
# "English", and filters match what is displayed, so both go through this.
LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "ru": "Russian", "cs": "Czech", "zh": "Chinese",
    "ja": "Japanese", "pt": "Portuguese", "sv": "Swedish", "sk": "Slovak",
    "nl": "Dutch", "hu": "Hungarian", "pl": "Polish", "sr": "Serbian",
    "sl": "Slovenian", "ca": "Catalan", "ko": "Korean", "da": "Danish",
    "hr": "Croatian", "eo": "Esperanto", "no": "Norwegian", "uk": "Ukrainian",
    "tr": "Turkish", "el": "Greek", "ro": "Romanian", "bs": "Bosnian",
    "ms": "Malay", "bn": "Bengali", "id": "Indonesian", "ar": "Arabic",
    "iu": "Inuktitut", "fi": "Finnish", "is": "Icelandic",
    # Spelled-out and three-letter forms that appear in the dump.
    "english": "English", "danish": "Danish", "rus": "Russian",
    "cat": "Catalan", "sco": "Scots", "jbo": "Lojban",
    "tok": "Toki Pona", "toki pona": "Toki Pona",
}

# Tokens carrying no language information. "mis" and "und" are the ISO codes for
# "uncoded" and "undetermined", which say nothing a reader can use.
LANGUAGE_NOISE = {"+", "users choice", "mis", "und", "zxx", "n/a", "none", "nan"}

_PARENTHETICAL = re.compile(r"\([^)]*\)")


def clean_language(raw) -> str:
    """
    IFDB's language field to displayable names, e.g. "en, fr" -> "English, French".

    Unrecognised tokens are kept rather than dropped — a code this map has not
    seen is still information, and silently blanking it would hide the game's
    language rather than admit the map is incomplete. Short unknowns are upper-
    cased so they read as codes; longer ones are title-cased so "spanglish"
    arrives as "Spanglish".
    """
    names = []
    text = _PARENTHETICAL.sub(" ", str(raw or ""))
    for part in re.split(r"[,;/]", text):
        token = part.strip().lower()
        if not token or token in LANGUAGE_NOISE:
            continue
        base = token.split("-")[0].split("_")[0].strip()
        if base in LANGUAGE_NOISE:
            continue
        name = LANGUAGE_NAMES.get(base) or LANGUAGE_NAMES.get(token)
        if name is None:
            name = base.upper() if len(base) <= 3 else base.title()
        if name not in names:
            names.append(name)
    return ", ".join(names)


def clean_description(raw) -> str:
    """
    IFDB's HTML description to a single line of plain text.

    Block-level ends become spaces before tags are stripped, or the last word of
    a paragraph would be glued to the first word of the next. Entities are
    unescaped here because the renderer escapes on the way out — leaving them
    would double-encode and show `&amp;amp;`.
    """
    text = str(raw or "")
    if not text.strip():
        return ""
    text = _MARKUP_BREAK.sub(" ", text)
    text = _MARKUP_TAG.sub("", text)
    # Strip again after unescaping: a few descriptions are double-encoded, so
    # `&lt;p&gt;` only becomes a tag once entities are resolved. The tag pattern
    # requires a letter after `<`, so this leaves author-written `a < b` and
    # `<3` intact.
    text = _MARKUP_TAG.sub("", unescape(text))
    return _WHITESPACE.sub(" ", text).strip()


def format_display(
    raw: str,
    display_map: Dict[str, Tuple[str, int]],
    separators: re.Pattern = TAG_SEPARATORS,
) -> str:
    """
    Render one field for display: split, deduplicated, canonically cased, and
    ordered by how widely each value is used, most common first.

    Popularity ordering puts the values that situate a game — "fantasy",
    "parser" — ahead of the long tail of one-off entries, which is what a reader
    scanning a truncated column wants to see first. Output is always
    comma-separated, so slash-delimited source values are normalised too.
    """
    chosen: Dict[str, Tuple[str, int]] = {}
    for part in separators.split(str(raw or "")):
        item = part.strip()
        if not item:
            continue
        key = item.lower()
        if key not in chosen:
            chosen[key] = display_map.get(key, (item, 0))
    ordered = sorted(chosen.values(), key=lambda pair: (-pair[1], pair[0].lower()))
    return ", ".join(display for display, _ in ordered)


def author_game_map(game_docs: pd.DataFrame) -> Dict[str, List[str]]:
    """Map each individual author (lowercased) to the games they wrote."""
    games: Dict[str, List[str]] = {}
    author_col = clean_col_in(game_docs.columns, "author")
    for gid, authors in zip(game_docs["gameid"], game_docs[author_col].fillna("")):
        # author_clean is already split into individual people and rejoined.
        for name in {a.strip().lower() for a in str(authors).split(",") if a.strip()}:
            games.setdefault(name, []).append(gid)
    return games


def build_author_profiles(
    game_docs: pd.DataFrame,
    min_games: int = 1,
    n_systems: int = 3,
    n_tags: int = _MAX_TAGS,
) -> pd.DataFrame:
    """
    Build a taste profile for each author from the games they wrote.

    Structurally identical to a user profile — top systems and tags in the same
    `format_profile_text` shape — but aggregated over an author's own catalogue
    rather than the games a user rated highly. That makes "recommend me something
    like this author" just another profile query, using the same encoders.

    Single-game authors are included by default. Their profile is effectively
    that game's `query_text`, so results resemble `game_id` mode for that game —
    but a user picking an author has no idea how many games they wrote, and
    being told to switch modes and find a game ID would be a poor answer.

    Returns columns: authorid, name, game_count, profile_text.
    """
    games = author_game_map(game_docs)
    casings = build_display_map(game_docs, "author", AUTHOR_SEPARATORS)
    systems_by_game = dict(zip(game_docs["gameid"], game_docs[clean_col("system")].fillna("")))
    tags_by_game = dict(zip(game_docs["gameid"], game_docs[clean_col("tags")].fillna("")))

    rows: List[dict] = []
    for key, gids in games.items():
        if len(gids) < min_games:
            continue
        systems: Counter = Counter()
        tags: Counter = Counter()
        for gid in gids:
            systems.update(v.strip() for v in str(systems_by_game.get(gid, "")).split(",") if v.strip())
            tags.update(v.strip() for v in str(tags_by_game.get(gid, "")).split(",") if v.strip())
        profile_text = format_profile_text(
            [s for s, _ in systems.most_common(n_systems)],
            [t for t, _ in tags.most_common(n_tags)],
        )
        if not profile_text:
            continue
        rows.append({
            "authorid": key,
            "name": casings.get(key, (key, 0))[0],
            "game_count": len(gids),
            "profile_text": profile_text,
        })
    return pd.DataFrame(rows).sort_values("game_count", ascending=False).reset_index(drop=True)


def canonical_vibe(systems, tags, system_rank=None, tag_rank=None):
    """
    Put a vibe pick in a fixed order, so the same picks give the same query.

    A multiselect reports values in click order, which carries no meaning — but
    it reaches the encoder as text, so "Tags: horror, romance" and "Tags:
    romance, horror" embed differently and return pages that differ by 4-12% of
    their entries. Two people wanting the same thing get different answers, and
    a lookup table would have to store both spellings to catch either.

    Ordered by corpus frequency rather than alphabetically, because that is how
    the profiles the encoder trained on were built, and how the pickers list
    their options — so a canonical query stays in distribution instead of
    landing in an ordering the model never saw. Duplicates are dropped, and
    anything unranked sorts last, alphabetically, so the result is total.
    """
    system_rank = system_rank or {}
    tag_rank = tag_rank or {}
    far = float("inf")
    ordered_systems = sorted(dict.fromkeys(systems),
                             key=lambda v: (system_rank.get(v, far), v))
    ordered_tags = sorted(dict.fromkeys(tags),
                          key=lambda v: (tag_rank.get(v, far), v))
    return ordered_systems, ordered_tags


def parse_profile_text(text: str) -> Tuple[List[str], List[str]]:
    """Inverse of `format_profile_text` — recover (systems, tags) from a query."""
    systems: List[str] = []
    tags: List[str] = []
    for part in str(text).split(". "):
        part = part.strip()
        if part.startswith("Systems:"):
            systems = [v.strip() for v in part[len("Systems:"):].split(",") if v.strip()]
        elif part.startswith("Tags:"):
            tags = [v.strip() for v in part[len("Tags:"):].split(",") if v.strip()]
    return systems, tags


def clean_frequencies(game_docs: pd.DataFrame, column: str) -> Counter:
    """How many games carry each `_clean` value, for corpus-order display."""
    counts: Counter = Counter()
    for value in game_docs[column].fillna(""):
        counts.update({v.strip() for v in str(value).split(",") if v.strip()})
    return counts


def profile_display(
    query_text: str,
    system_freq: Optional[Counter] = None,
    tag_freq: Optional[Counter] = None,
) -> str:
    """
    Compact rendering of a profile: "twine, ink // mystery, surreal".

    The stored form ("Systems: … Tags: …") is what the encoders were trained on
    and must not change; this is presentation only. Dropping the labels keeps it
    lowercase and avoids implying the second list is only tags — genre values are
    folded into it during preprocessing.

    Pass the frequency maps to order by how common each value is across the
    corpus, or omit them to keep the stored order. Which is right depends on
    where the profile came from, because "most frequent" means different things:

      reviewer / author  – stored order is frequency across the games they rated
                           or wrote, which is the informative one: it says what
                           this person actually gravitates to.
      game / vibe        – no such history to count, so corpus order, matching
                           how the results table orders its own columns.
    """
    systems, tags = parse_profile_text(query_text)
    if system_freq is not None:
        systems = sorted(systems, key=lambda v: (-system_freq.get(v, 0), v))
    if tag_freq is not None:
        tags = sorted(tags, key=lambda v: (-tag_freq.get(v, 0), v))
    left, right = ", ".join(systems), ", ".join(tags)
    return f"{left} // {right}" if left and right else (left or right)


def profile_vocabulary(
    game_docs: pd.DataFrame, n_systems: int = 12, n_tags: int = 40
) -> Tuple[List[str], List[str]]:
    """
    Most common system and tag values, for populating a UI's pickers.

    Drawn from the `_clean` columns so the options a user selects are exactly the
    strings the encoders were trained on.
    """
    system_counts: Counter = Counter()
    tag_counts: Counter = Counter()
    for value in game_docs[clean_col("system")].fillna(""):
        system_counts.update(v.strip() for v in str(value).split(",") if v.strip())
    for value in game_docs[clean_col("tags")].fillna(""):
        tag_counts.update(v.strip() for v in str(value).split(",") if v.strip())
    return (
        [s for s, _ in system_counts.most_common(n_systems)],
        [t for t, _ in tag_counts.most_common(n_tags)],
    )


def _dedupe_ordered(items: List[str]) -> List[str]:
    """Remove duplicates from a list while preserving first-occurrence order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _filter_tags(tags_str: str) -> List[str]:
    """Split comma-separated tags, drop competition-related ones, and deduplicate."""
    if not tags_str:
        return []
    raw = [t.strip().lower() for t in tags_str.split(",") if t.strip() and not _COMP_RE.search(t.strip())]
    return _dedupe_ordered(raw)


def _split_genre(genre_str: str) -> List[str]:
    """Split a genre string on '/' or ',' into lowercase deduplicated genre values."""
    raw = [g.strip().lower() for g in re.split(r'[\,\/]+', str(genre_str)) if g.strip()]
    return _dedupe_ordered(raw)


_AUTHOR_SPLIT_RE = re.compile(r'\s*/\s*|\s*,\s*|\s+and\s+')
_YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')
_SYSTEM_PARENS_RE = re.compile(r'\([^)]*\)')       # strip (parentheticals)
_SYSTEM_NUMS_RE   = re.compile(r'\s*\b\d[\d.]*\b') # strip standalone numbers / versions


def _split_authors(author_str: str) -> List[str]:
    """Split author string on ',', '/', or ' and ' into individual author names."""
    raw = [a.strip() for a in _AUTHOR_SPLIT_RE.split(str(author_str).strip()) if a.strip()]
    return _dedupe_ordered(raw)


def _split_system(system_str: str) -> List[str]:
    """Clean and split a system string into canonical lowercase values.

    Strips parentheticals, standalone numbers/versions, then splits on ',' or '/'.
    Example: '  Inform 7 (alternative)' → ['inform']
    """
    cleaned = _SYSTEM_PARENS_RE.sub('', str(system_str)).lower()
    parts = [_SYSTEM_NUMS_RE.sub('', p).strip() for p in re.split(r'[,\/]+', cleaned)]
    return _dedupe_ordered([p for p in parts if p])


# ---------------------------------------------------------------------------
# Game documents
# ---------------------------------------------------------------------------

def build_game_documents(
    games: pd.DataFrame,
    reviews: pd.DataFrame,
    min_reviews: int = 2,
    bayesian_prior_mean: float = 3.5,
    bayesian_prior_weight: int = 10,
) -> pd.DataFrame:
    """
    Build a structured text document for each game (item-tower input).

    Normalisation is written to `_clean` columns and the IFDB originals are kept
    untouched beside them, so the interactive system can show exactly what
    ifdb.org shows while the encoders see the normalised text:

      author  → author_clean   split on ',' '/' 'and', deduplicated, rejoined
      system  → system_clean   parentheticals and version numbers stripped, lowercased
      tags    → tags_clean     genre folded in, competition tags dropped, capped at 20

    `genre` has no `_clean` variant: its values are folded into tags_clean.
    `year` is extracted from the original `published` timestamp for range filters.
    A Bayesian-smoothed rating is included as a quality signal.

    Only includes games where title, author, and desc are all non-empty and
    that have received at least min_reviews ratings.
    """
    docs = games.copy()

    for col in ("tags", "desc", "genre", "system", "title", "author", "published"):
        if col not in docs.columns:
            docs[col] = ""
        else:
            docs[col] = docs[col].fillna("")

    # Require title, author, system, desc, and tags to all be non-empty
    mask = (
        docs["title"].astype(str).str.strip().ne("") &
        docs["author"].astype(str).str.strip().ne("") &
        docs["system"].astype(str).str.strip().ne("") &
        docs["desc"].astype(str).str.strip().ne("") &
        docs["tags"].astype(str).str.strip().ne("")
    )
    docs = docs[mask].copy()

    # Review stats + Bayesian-smoothed average
    review_stats = reviews.groupby("gameid").agg(
        review_count=("rating", "count"),
        avg_rating=("rating", "mean"),
    ).reset_index()
    review_stats["avg_rating"] = review_stats["avg_rating"].round(2)
    review_stats["bayesian_avg"] = (
        (bayesian_prior_weight * bayesian_prior_mean
         + review_stats["review_count"] * review_stats["avg_rating"])
        / (bayesian_prior_weight + review_stats["review_count"])
    ).round(2)

    docs = docs.merge(review_stats, on="gameid", how="left")
    docs["review_count"]  = docs["review_count"].fillna(0).astype(int)
    docs["avg_rating"]    = docs["avg_rating"].fillna(bayesian_prior_mean)
    docs["bayesian_avg"]  = docs["bayesian_avg"].fillna(bayesian_prior_mean)
    docs = docs[docs["review_count"] >= min_reviews].copy()

    # Clean author: split on , / 'and', deduplicate, rejoin as comma-separated
    docs[clean_col("author")] = docs["author"].apply(
        lambda a: ", ".join(_split_authors(str(a)))
    )

    # Clean system: strip parentheticals and version numbers, split on , /,
    # deduplicate, store as comma-separated lowercase
    docs[clean_col("system")] = docs["system"].apply(
        lambda s: ", ".join(_split_system(str(s)))
    )

    # Extract publication year from 'published' field (display + range filters)
    if "published" in docs.columns:
        docs["year"] = docs["published"].apply(
            lambda p: (m := _YEAR_RE.search(str(p))) and m.group(0) or ""
        )
    else:
        docs["year"] = ""

    # Fold genre into tags: genre values are prepended (bypassing the competition
    # filter since they are explicitly set) then combined with filtered tag values.
    def _merge_genre_tags(row: pd.Series) -> str:
        genre_vals = _split_genre(str(row["genre"]))   # lowercased, clean
        tag_vals   = _filter_tags(str(row["tags"]))    # filtered, lowercased, deduped
        combined   = _dedupe_ordered(genre_vals + tag_vals)[:_MAX_TAGS]
        return ", ".join(combined)

    docs[clean_col("tags")] = docs.apply(_merge_genre_tags, axis=1)

    # Encoder inputs are built from the normalised columns only.
    def _make_doc(row: pd.Series) -> str:
        parts = [f"Title: {row['title']}", f"Author: {row[clean_col('author')]}"]
        if str(row[clean_col("system")]).strip():
            parts.append(f"Systems: {row[clean_col('system')]}")
        if str(row[clean_col("tags")]).strip():
            parts.append(f"Tags: {row[clean_col('tags')]}")
        snippet = str(row["desc"])[:500].strip()
        if snippet:
            parts.append(f"Description: {snippet}")
        return ". ".join(parts)

    def _make_query_doc(row: pd.Series) -> str:
        """Game profile text without title/author/desc — matches user profile format."""
        systems = [str(row[clean_col("system")])] if str(row[clean_col("system")]).strip() else []
        tags = [str(row[clean_col("tags")])] if str(row[clean_col("tags")]).strip() else []
        return format_profile_text(systems, tags)

    docs["doc_text"]   = docs.apply(_make_doc, axis=1)
    docs["query_text"] = docs.apply(_make_query_doc, axis=1)

    keep = ["gameid", "title",
            # IFDB originals — what the interactive system displays
            "author", "genre", "system", "tags", "published", "year",
            # normalised variants — what the models and filters consume
            clean_col("author"), clean_col("system"), clean_col("tags"),
            "avg_rating", "bayesian_avg", "review_count", "doc_text", "query_text"]
    return docs[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Interaction matrix
# ---------------------------------------------------------------------------

def build_interactions(
    reviews: pd.DataFrame,
    users: pd.DataFrame,
    game_docs: pd.DataFrame,
    rating_deviation_threshold: float = 0.25,
    min_reviews_per_user: int = 2,
    min_reviews_per_game: int = 2,
) -> pd.DataFrame:
    """
    Build labelled interactions from review ratings relative to each game's quality.

    label=1 → positive (rating > bayesian_avg + rating_deviation_threshold)
    label=0 → negative (rating < bayesian_avg - rating_deviation_threshold)

    Ratings within the threshold band are discarded (ambiguous signal).
    Uses game_docs["bayesian_avg"] so that a 3-star review of a 2-star game
    is treated as positive, while the same rating for a 5-star game is negative.

    Filters to real users only: reviews must belong to a userid present in the
    users table AND the userid must be a plain lowercase alphanumeric string
    (rejects system/bot accounts such as '$system').
    """
    if reviews.empty:
        return pd.DataFrame(columns=["userid", "gameid", "label"])

    base = reviews[["userid", "gameid", "rating"]].copy()

    # Keep only reviews from real registered users
    if not users.empty and "userid" in users.columns:
        valid_ids = set(users["userid"].dropna().astype(str))
        base = base[base["userid"].isin(valid_ids)]

    # Belt-and-suspenders: drop any userid that isn't purely lowercase alphanumeric
    base = base[base["userid"].astype(str).str.fullmatch(r'[a-z0-9]+')]

    # Merge in bayesian_avg per game; fall back to prior for unrecognised games
    base = base.merge(game_docs[["gameid", "bayesian_avg"]], on="gameid", how="left")
    base["bayesian_avg"] = base["bayesian_avg"].fillna(3.5)

    pos = base[base["rating"] > base["bayesian_avg"] + rating_deviation_threshold][["userid", "gameid"]].copy()
    pos["label"] = 1
    neg = base[base["rating"] < base["bayesian_avg"] - rating_deviation_threshold][["userid", "gameid"]].copy()
    neg["label"] = 0

    interactions = pd.concat([pos, neg], ignore_index=True)
    interactions = interactions.drop_duplicates(subset=["userid", "gameid"], keep="first")

    user_counts = interactions.groupby("userid").size()
    valid_users = user_counts[user_counts >= min_reviews_per_user].index
    interactions = interactions[interactions["userid"].isin(valid_users)]

    game_counts = interactions.groupby("gameid").size()
    valid_games = game_counts[game_counts >= min_reviews_per_game].index
    interactions = interactions[interactions["gameid"].isin(valid_games)]

    return interactions.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Train / val / test split (user-level)
# ---------------------------------------------------------------------------

def split_interactions(
    interactions: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split interactions per-user so every user appears in all three partitions.

    For each user, positive interactions are shuffled and distributed:
      - 1 positive guaranteed to train (for profile building)
      - remaining positives split into test (~test_frac) and val (~val_frac)
      - users with only 1 positive keep it in train and are skipped in eval
      - all negative interactions go to train (used only for training signal)

    Returns (train, val, test).
    """
    rng = np.random.RandomState(random_state)
    train_rows: List[pd.DataFrame] = []
    val_rows:   List[pd.DataFrame] = []
    test_rows:  List[pd.DataFrame] = []

    n_users_with_test = 0
    for _, grp in interactions.groupby("userid"):
        pos = grp[grp["label"] == 1]
        neg = grp[grp["label"] == 0]

        # Shuffle positives with a reproducible RNG
        pos = pos.iloc[rng.permutation(len(pos))]
        n = len(pos)

        if n == 1:
            train_rows.append(pos)
        elif n == 2:
            train_rows.append(pos.iloc[:1])
            test_rows.append(pos.iloc[1:])
            n_users_with_test += 1
        elif n == 3:
            train_rows.append(pos.iloc[:1])
            val_rows.append(pos.iloc[1:2])
            test_rows.append(pos.iloc[2:])
            n_users_with_test += 1
        else:
            n_test  = max(1, round(n * test_frac))
            n_val   = max(1, round(n * val_frac))
            n_train = n - n_test - n_val
            if n_train < 1:
                n_train = 1
                if n_val > 1:
                    n_val -= 1
                else:
                    n_test -= 1
            train_rows.append(pos.iloc[:n_train])
            val_rows.append(pos.iloc[n_train : n_train + n_val])
            test_rows.append(pos.iloc[n_train + n_val :])
            n_users_with_test += 1

        train_rows.append(neg)  # all negatives go to train

    def _concat(rows: List[pd.DataFrame]) -> pd.DataFrame:
        if rows:
            return pd.concat(rows, ignore_index=True)
        return pd.DataFrame(columns=interactions.columns)

    train = _concat(train_rows)
    val   = _concat(val_rows)
    test  = _concat(test_rows)

    n_total = interactions["userid"].nunique()
    logger.info(
        "Split: %d users total / %d with test items / %d val items / %d test items",
        n_total, n_users_with_test, len(val), len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# User profiles
# ---------------------------------------------------------------------------

def build_user_profiles(
    interactions: pd.DataFrame,
    game_docs: pd.DataFrame,
    reviews: Optional[pd.DataFrame] = None,
    min_absolute_rating: float = 4.0,
    users: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a text profile for each user from their positively-rated games.

    Profile text aggregates top system and tag values (tags include genre values
    since genre is folded into tags during game document building) from:
      1. Games the user rated positively in the interaction matrix (label=1)
      2. Games the user gave an absolute rating >= min_absolute_rating (if reviews provided)

    Profile format: "Systems: Z. Tags: a, b, c."
    Returns columns: userid, name, profile_text.
    """
    # Build game_id sets per user from training positives (label=1)
    pos = interactions[interactions["label"] == 1][["userid", "gameid"]]

    # Also include games with absolute rating >= min_absolute_rating
    if reviews is not None and not reviews.empty:
        high_rated = reviews[reviews["rating"] >= min_absolute_rating][["userid", "gameid"]]
        known_users = set(pos["userid"].unique())
        high_rated = high_rated[high_rated["userid"].isin(known_users)]
        combined = pd.concat([pos, high_rated], ignore_index=True).drop_duplicates()
    else:
        combined = pos.copy()

    # Pre-build lookup maps from game_docs. Profile text is encoder input, so it
    # is built from the normalised columns (falling back to the originals for
    # game_docs files written before the clean/original split).
    game_docs_idx = game_docs.set_index("gameid")
    system_map = game_docs_idx[clean_col_in(game_docs.columns, "system")].to_dict()
    tags_map   = game_docs_idx[clean_col_in(game_docs.columns, "tags")].to_dict()

    # Name lookup from users table
    name_lookup: Dict[str, str] = {}
    if users is not None and "name" in users.columns:
        name_lookup = dict(zip(
            users["userid"].astype(str),
            users["name"].fillna("").astype(str),
        ))

    profiles = []
    for uid, grp in combined.groupby("userid"):
        systems: List[str] = []
        tags: List[str] = []

        for gid in grp["gameid"]:
            # system column is comma-separated; split into individual values
            systems.extend(
                sv.strip() for sv in system_map.get(gid, "").split(",") if sv.strip()
            )
            tags.extend(
                t.strip() for t in str(tags_map.get(gid, "")).split(",") if t.strip()
            )

        top_systems = [s for s, _ in Counter(systems).most_common(3)]
        top_tags    = [t for t, _ in Counter(tags).most_common(_MAX_TAGS)]

        profile_text = format_profile_text(top_systems, top_tags)
        if not profile_text:
            continue

        profiles.append({
            "userid":       uid,
            "name":         name_lookup.get(str(uid), ""),
            "profile_text": profile_text,
        })

    return pd.DataFrame(profiles)
