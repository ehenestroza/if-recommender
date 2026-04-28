#!/usr/bin/env python
"""
Step 2 – Prepare training data from raw Parquet files.

What this script does
---------------------
1. Loads raw tables from data/
2. Builds game document strings (item-tower inputs)
3. Assembles interaction matrix from review ratings and splits train/val/test
4. Builds user profile strings and review text lists (query-tower inputs)
5. Saves processed artefacts to data/

After this step data/ will contain:
  game_docs.parquet     – gameid, title, author, genre, system, tags, avg_rating, bayesian_avg, review_count, doc_text
  interactions.parquet  – all interactions with label + split column
  user_profiles.parquet – userid, profile_text, review_texts

Usage
-----
    python scripts/02_preprocess.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_parquet
from src.data.preprocessor import (
    build_game_documents,
    build_interactions,
    split_interactions,
    build_user_profiles,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])
    tr_cfg   = cfg["training"]
    pp_cfg   = cfg.get("preprocessing", {})

    # ------------------------------------------------------------------ #
    # 1. Load raw tables
    # ------------------------------------------------------------------ #
    logger.info("Loading raw tables …")
    games   = load_parquet(data_dir, "games.parquet")
    reviews = load_parquet(data_dir, "reviews.parquet")
    users   = load_parquet(data_dir, "users.parquet")
    logger.info("Loaded: games=%d, reviews=%d, users=%d", len(games), len(reviews), len(users))

    # ------------------------------------------------------------------ #
    # 2. Game documents
    # ------------------------------------------------------------------ #
    logger.info("Building game documents …")
    game_docs = build_game_documents(
        games=games,
        reviews=reviews,
        min_reviews=tr_cfg["min_reviews_per_game"],
        bayesian_prior_mean=pp_cfg.get("bayesian_prior_mean", 3.5),
        bayesian_prior_weight=pp_cfg.get("bayesian_prior_weight", 10),
    )
    game_docs.to_parquet(data_dir / "game_docs.parquet", index=False)
    logger.info("Saved game_docs.parquet (%d games)", len(game_docs))

    sample_1 = game_docs.iloc[0]
    logger.info(
        "Sample game document:\n  gameid=%s\n  doc_text=%s",
        sample_1["gameid"], sample_1["doc_text"],
    )

    sample_2 = game_docs.iloc[1]
    logger.info(
        "Sample game document:\n  gameid=%s\n  doc_text=%s",
        sample_2["gameid"], sample_2["doc_text"],
    )

    # ------------------------------------------------------------------ #
    # 3. Interactions + split
    # ------------------------------------------------------------------ #
    logger.info("Building interaction matrix …")
    interactions = build_interactions(
        reviews=reviews,
        users=users,
        min_rating_positive=tr_cfg["min_rating_positive"],
        max_rating_negative=tr_cfg["max_rating_negative"],
        min_reviews_per_user=tr_cfg["min_reviews_per_user"],
        min_reviews_per_game=tr_cfg["min_reviews_per_game"],
    )
    logger.info(
        "Interactions: %d total  (%d positive, %d negative)",
        len(interactions),
        (interactions["label"] == 1).sum(),
        (interactions["label"] == 0).sum(),
    )

    sample_ix = interactions.iloc[0]
    logger.info(
        "Sample interaction: userid=%s | gameid=%s | label=%d",
        sample_ix["userid"], sample_ix["gameid"], int(sample_ix["label"]),
    )

    train, val, test = split_interactions(
        interactions,
        val_frac=tr_cfg["val_frac"],
        test_frac=tr_cfg["test_frac"],
    )

    train["split"] = "train"
    val["split"]   = "val"
    test["split"]  = "test"
    all_splits = pd.concat([train, val, test], ignore_index=True)
    all_splits.to_parquet(data_dir / "interactions.parquet", index=False)
    logger.info("Saved interactions.parquet (%d rows total)", len(all_splits))

    # ------------------------------------------------------------------ #
    # 4. User profiles  (built from training-set positives only)
    # ------------------------------------------------------------------ #
    logger.info("Building user profiles from training positives …")
    user_profiles = build_user_profiles(
        interactions=train,
        game_docs=game_docs,
    )
    user_profiles.to_parquet(data_dir / "user_profiles.parquet", index=False)
    logger.info("Saved user_profiles.parquet (%d users)", len(user_profiles))

    sample_profile_1 = user_profiles.iloc[0]
    logger.info(
        "Sample user profile 1:\n  userid=%s\n  profile_text=%s",
        sample_profile_1["userid"], sample_profile_1["profile_text"],
    )

    sample_profile_2 = user_profiles.iloc[1]
    logger.info(
        "Sample user profile 2:\n  userid=%s\n  profile_text=%s",
        sample_profile_2["userid"], sample_profile_2["profile_text"],
    )

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    pos_train = (train["label"] == 1).sum()
    logger.info(
        "Training set: %d positive pairs across %d users and %d games",
        pos_train,
        train["userid"].nunique(),
        train["gameid"].nunique(),
    )
    logger.info("✓ Data preparation complete.")


if __name__ == "__main__":
    main()
