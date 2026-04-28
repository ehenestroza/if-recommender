"""Transform raw IFDB parquet tables into training artefacts."""

import logging
import re
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_COMP_RE = re.compile(r"(comp)|(award)|(\d\d\d\d)|(transcript)|(xyzzy)|(spring thing)|(source available)|(prize)|(showcase)|(jam)|(walkthrough)|(cover art)", re.IGNORECASE)


def _filter_tags(tags_str: str) -> List[str]:
    """Split comma-separated tags and drop any containing 'comp'."""
    if not tags_str:
        return []
    return [t.strip().lower() for t in tags_str.split(",") if t.strip() and not _COMP_RE.search(t.strip())]


# ---------------------------------------------------------------------------
# Game documents
# ---------------------------------------------------------------------------

def _split_genre(genre_str: str) -> List[str]:
    """Split a genre string on '/' or ',' and return cleaned individual genre values."""
    return [g.strip() for g in re.split(r'[\,\/]+', str(genre_str)) if g.strip() and g != g.lower()]


def build_game_documents(
    games: pd.DataFrame,
    reviews: pd.DataFrame,
    min_reviews: int = 2,
    bayesian_prior_mean: float = 3.5,
    bayesian_prior_weight: int = 10,
) -> pd.DataFrame:
    """
    Build a structured text document for each game (item-tower input).

    Categorical fields (author, genre, system) each appear as their own
    phrase so the model can learn field-level alignment. Genre is split on '/'
    to handle multi-genre entries. Tags are filtered of competition noise and
    stored cleaned. A Bayesian-smoothed rating is included as a quality signal.

    Only includes games where title, author, and desc are all non-empty and
    that have received at least min_reviews ratings.

    Returns columns: gameid, title, author, genre, system, tags,
                     avg_rating, bayesian_avg, review_count, doc_text.
    """
    docs = games.copy()

    for col in ("tags", "desc", "genre", "system", "title", "author"):
        if col not in docs.columns:
            docs[col] = ""
        else:
            docs[col] = docs[col].fillna("")

    # Require title, author, and desc to be non-empty
    mask = (
        docs["title"].astype(str).str.strip().ne("") &
        docs["author"].astype(str).str.strip().ne("") &
        docs["desc"].astype(str).str.strip().ne("")
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

    # Store filtered tags (removes competition noise) so user profiles inherit clean tags
    docs["tags"] = docs["tags"].apply(
        lambda t: ", ".join(_filter_tags(str(t)))
    )

    def _make_doc(row: pd.Series) -> str:
        parts = [f"Title: {row['title']}", f"Author: {row['author']}"]
        # Each genre value as its own phrase (handles "Fiction/Horror" → two phrases)
        for g in _split_genre(str(row["genre"])):
            parts.append(f"Genre: {g}")
        if str(row["system"]).strip():
            parts.append(f"System: {row['system']}")
        parts.append(f"Rating: {row['bayesian_avg']:.1f}/5")
        if str(row["tags"]).strip():
            parts.append(f"Tags: {row['tags']}")
        snippet = str(row["desc"])[:500].strip()
        if snippet:
            parts.append(f"Description: {snippet}")
        return ". ".join(parts)

    docs["doc_text"] = docs.apply(_make_doc, axis=1)

    keep = ["gameid", "title", "author", "genre", "system", "tags",
            "avg_rating", "bayesian_avg", "review_count", "doc_text"]
    return docs[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Interaction matrix
# ---------------------------------------------------------------------------

def build_interactions(
    reviews: pd.DataFrame,
    users: pd.DataFrame,
    min_rating_positive: int = 4,
    max_rating_negative: int = 3,
    min_reviews_per_user: int = 2,
    min_reviews_per_game: int = 2,
) -> pd.DataFrame:
    """
    Build labelled interactions from review ratings only.

    label=1 → positive (rating >= min_rating_positive)
    label=0 → negative (rating <= max_rating_negative)

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
    pos = base[base["rating"] >= min_rating_positive][["userid", "gameid"]].copy()
    pos["label"] = 1
    neg = base[base["rating"] <= max_rating_negative][["userid", "gameid"]].copy()
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
) -> pd.DataFrame:
    """
    Build a text profile for each user from their positively-rated games.

    Profile text aggregates top genre, system, and tag values from the games
    each user rated positively. Returns columns: userid, profile_text.
    """
    pos = interactions[interactions["label"] == 1]

    # Pre-build lookup maps from game_docs
    game_docs_idx = game_docs.set_index("gameid")
    genre_map  = game_docs_idx["genre"].to_dict()
    system_map = game_docs_idx["system"].to_dict()
    tags_map   = game_docs_idx["tags"].to_dict()

    profiles = []
    for uid, grp in pos.groupby("userid"):
        genres: List[str] = []
        systems: List[str] = []
        tags: List[str] = []

        for gid in grp["gameid"]:
            # Split genre on '/' to match how game docs are built
            genre_str = genre_map.get(gid, "")
            if genre_str:
                genres.extend(_split_genre(str(genre_str)))
            s = system_map.get(gid, "")
            if s:
                systems.append(str(s))
            # tags_map now holds pre-filtered comma-separated tags from game_docs
            raw = str(tags_map.get(gid, ""))
            tags.extend([t.strip() for t in raw.split(",") if t.strip()])

        top_genres  = [g for g, _ in Counter(genres).most_common(5)]
        top_systems = [s for s, _ in Counter(systems).most_common(3)]
        top_tags    = [t for t, _ in Counter(tags).most_common(20)]

        # Build profile using the same field-per-phrase format as game docs so
        # the model learns to align "Genre: Fiction" ↔ "Genre: Fiction" etc.
        parts: List[str] = []
        for g in top_genres:
            parts.append(f"Genre: {g}")
        for s in top_systems:
            parts.append(f"System: {s}")
        if top_tags:
            parts.append(f"Tags: {', '.join(top_tags)}")

        if not parts:
            continue

        profile_text = ". ".join(parts)
        profiles.append({"userid": uid, "profile_text": profile_text})

    return pd.DataFrame(profiles)
