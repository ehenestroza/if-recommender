"""Transform raw IFDB parquet tables into training artefacts."""

import logging
from collections import Counter
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Game documents
# ---------------------------------------------------------------------------

def _pick_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return the first candidate column that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_game_documents(games: pd.DataFrame, gametags: pd.DataFrame) -> pd.DataFrame:
    """
    Build a rich text document for each game (item-tower input).

    Returns a DataFrame with columns: gameid, title, author, genre, doc_text.
    """
    desc_col = _pick_col(games, "desc", "description", "blurb")

    docs = games.copy()

    # games.tags already contains comma-separated community tags in the IFDB
    # dump. Fall back to aggregating from the gametags join table only if that
    # column is absent.
    if "tags" in docs.columns:
        docs["tags"] = docs["tags"].fillna("")
    elif not gametags.empty and "tag" in gametags.columns and "gameid" in gametags.columns:
        tag_series = (
            gametags[gametags["tag"].notna()][["gameid", "tag"]]
            .drop_duplicates()
            .sort_values("tag")
            .groupby("gameid")["tag"]
            .agg(", ".join)
        )
        tag_agg = pd.DataFrame({"gameid": tag_series.index, "tags": tag_series.to_numpy()})
        docs = docs.merge(tag_agg, on="gameid", how="left")
        docs["tags"] = docs["tags"].fillna("")
    else:
        docs["tags"] = ""

    def _make_doc(row: pd.Series) -> str:
        parts = []
        if row.get("title"):
            parts.append(f"Title: {row['title']}")
        if row.get("author"):
            parts.append(f"Author: {row['author']}")
        if row.get("genre"):
            parts.append(f"Genre: {row['genre']}")
        if row.get("tags"):
            parts.append(f"Tags: {row['tags']}")
        if desc_col and row.get(desc_col):
            snippet = str(row[desc_col])[:500].strip()
            if snippet:
                parts.append(f"Description: {snippet}")
        return " | ".join(parts) if parts else str(row.get("title", row["gameid"]))

    docs["doc_text"] = docs.apply(_make_doc, axis=1)

    keep = ["gameid", "title", "author", "genre", "doc_text"]
    for col in keep:
        if col not in docs.columns:
            docs[col] = ""

    return docs[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Interaction matrix
# ---------------------------------------------------------------------------

def build_interactions(
    reviews: pd.DataFrame,
    wishlists: pd.DataFrame,
    playedgames: pd.DataFrame,
    min_rating_positive: int = 4,
    max_rating_negative: int = 2,
    min_reviews_per_user: int = 3,
    min_reviews_per_game: int = 3,
) -> pd.DataFrame:
    """
    Combine all interaction signals into a single labelled DataFrame.

    label=1 → positive (liked / wishlisted / played)
    label=0 → negative (low-rated review)
    """
    parts = []

    rating_col = _pick_col(reviews, "rating", "stars", "score")

    if not reviews.empty and rating_col:
        base = reviews[["userid", "gameid", rating_col]].copy()
        pos = base[base[rating_col] >= min_rating_positive][["userid", "gameid"]].copy()
        pos["label"] = 1
        neg = base[base[rating_col] <= max_rating_negative][["userid", "gameid"]].copy()
        neg["label"] = 0
        parts.extend([pos, neg])

    if not wishlists.empty:
        wl = wishlists[["userid", "gameid"]].copy()
        wl["label"] = 1
        parts.append(wl)

    if not playedgames.empty:
        pg = playedgames[["userid", "gameid"]].copy()
        pg["label"] = 1
        parts.append(pg)

    if not parts:
        return pd.DataFrame(columns=["userid", "gameid", "label"])

    interactions = pd.concat(parts, ignore_index=True)

    # Drop duplicate (user, game) pairs; reviews-derived rows come first so
    # an explicit rating takes precedence over implicit signals.
    interactions = interactions.drop_duplicates(subset=["userid", "gameid"], keep="first")

    # Require minimum interaction counts to reduce noise
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
    Split by user so no user appears in more than one partition.
    Returns (train, val, test).
    """
    users = interactions["userid"].unique().tolist()
    holdout_frac = val_frac + test_frac

    train_users, temp_users = train_test_split(
        users, test_size=holdout_frac, random_state=random_state
    )
    relative_test = test_frac / holdout_frac
    val_users, test_users = train_test_split(
        temp_users, test_size=relative_test, random_state=random_state
    )

    train = interactions[interactions["userid"].isin(train_users)].copy()
    val   = interactions[interactions["userid"].isin(val_users)].copy()
    test  = interactions[interactions["userid"].isin(test_users)].copy()

    logger.info(
        "Split: train=%d users / val=%d users / test=%d users",
        len(train_users), len(val_users), len(test_users),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# User profiles
# ---------------------------------------------------------------------------

def build_user_profiles(
    interactions: pd.DataFrame,
    game_docs: pd.DataFrame,
    gametags: pd.DataFrame,
    min_rating_positive: int = 4,
) -> pd.DataFrame:
    """
    Build a text query for each user from the tags of their positively-rated games.

    Format: "A player who enjoys: mystery, puzzle, historical, …"
    """
    pos = interactions[interactions["label"] == 1]

    # Tag lookup: gameid → list of tags
    if not gametags.empty and "tag" in gametags.columns:
        tag_map: dict[str, list[str]] = (
            gametags.groupby("gameid")["tag"].apply(list).to_dict()
        )
    else:
        # Fall back to genre from game_docs
        tag_map = {
            row["gameid"]: [row["genre"]]
            for _, row in game_docs.iterrows()
            if row.get("genre")
        }

    profiles = []
    for uid, grp in pos.groupby("userid"):
        all_tags: list[str] = []
        for gid in grp["gameid"]:
            all_tags.extend(tag_map.get(gid, []))

        if not all_tags:
            continue

        tag_counts = Counter(all_tags)
        top_tags = [t for t, _ in tag_counts.most_common(20)]
        profile_text = "A player who enjoys: " + ", ".join(top_tags)
        profiles.append({"userid": uid, "profile_text": profile_text})

    return pd.DataFrame(profiles)
