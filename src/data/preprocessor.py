"""Transform raw IFDB parquet tables into training artefacts."""

import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_COMP_RE = re.compile(
    r"(comp)|(award)|(\d\d\d\d)|(transcript)|(xyzzy)|(spring thing)|(source available)|(prize)|(showcase)|(jam)|"
    r"(walkthrough)|(cover art)|(available)|(adrift)|(inform)|(windrift)|(twine)|(interactive fiction)|(top \d\d)|"
    r"(winner)|(playoff)|(inklewriter)|(storynexus)|(choicescript)|(ink)|(student project)", re.IGNORECASE)

_MAX_TAGS = 20


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

    Genres and systems are stored as comma-separated lowercase values.
    Tags are filtered of competition noise, deduplicated (order-preserving),
    and capped at 20 entries. A Bayesian-smoothed rating is included as a
    quality signal.

    Only includes games where title, author, and desc are all non-empty and
    that have received at least min_reviews ratings.

    Returns columns: gameid, title, author, genre, system, tags,
                     avg_rating, bayesian_avg, review_count, doc_text, query_text.
    """
    docs = games.copy()

    for col in ("tags", "desc", "genre", "system", "title", "author"):
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
    docs["author"] = docs["author"].apply(
        lambda a: ", ".join(_split_authors(str(a)))
    )

    # Clean system: strip parentheticals and version numbers, split on , /,
    # deduplicate, store as comma-separated lowercase
    docs["system"] = docs["system"].apply(
        lambda s: ", ".join(_split_system(str(s)))
    )

    # Extract publication year from 'published' field (display-only, not used in model)
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

    docs["tags"] = docs.apply(_merge_genre_tags, axis=1)

    def _make_doc(row: pd.Series) -> str:
        parts = [f"Title: {row['title']}", f"Author: {row['author']}"]
        if str(row["system"]).strip():
            parts.append(f"Systems: {row['system']}")
        if str(row["tags"]).strip():
            parts.append(f"Tags: {row['tags']}")
        snippet = str(row["desc"])[:500].strip()
        if snippet:
            parts.append(f"Description: {snippet}")
        return ". ".join(parts)

    def _make_query_doc(row: pd.Series) -> str:
        """Game profile text without title/author/desc — matches user profile format."""
        parts = []
        if str(row["system"]).strip():
            parts.append(f"Systems: {row['system']}")
        if str(row["tags"]).strip():
            parts.append(f"Tags: {row['tags']}")
        return ". ".join(parts)

    docs["doc_text"]   = docs.apply(_make_doc, axis=1)
    docs["query_text"] = docs.apply(_make_query_doc, axis=1)

    keep = ["gameid", "title", "author", "year", "system", "tags",
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

    # Pre-build lookup maps from game_docs
    game_docs_idx = game_docs.set_index("gameid")
    system_map = game_docs_idx["system"].to_dict()  # comma-separated lowercase systems
    tags_map   = game_docs_idx["tags"].to_dict()    # filtered, genre-prepended, capped

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

        parts: List[str] = []
        if top_systems:
            parts.append(f"Systems: {', '.join(top_systems)}")
        if top_tags:
            parts.append(f"Tags: {', '.join(top_tags)}")

        if not parts:
            continue

        profiles.append({
            "userid":       uid,
            "name":         name_lookup.get(str(uid), ""),
            "profile_text": ". ".join(parts),
        })

    return pd.DataFrame(profiles)
