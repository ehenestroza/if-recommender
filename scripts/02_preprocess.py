#!/usr/bin/env python
"""
Step 2 – Prepare training data from raw Parquet files.

What this script does
---------------------
1. Loads raw tables from data/
2. Builds two game document sets:
     game_docs.parquet           – strict set (≥ min_reviews_per_game) for training
     game_docs_retrieval.parquet – broad set (all valid games, even 0 reviews) for indexing
3. Assembles interaction matrix from review ratings and splits train/val/test
4. Builds two user profile sets:
     user_profiles.parquet           – training-set positives only
     user_profiles_retrieval.parquet – all users with ≥1 positive interaction

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

from src.utils.env import configure_logging
configure_logging()

from src.data.loader import load_parquet
from src.data.preprocessor import (
    build_game_documents,
    build_interactions,
    split_interactions,
    build_user_profiles,
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
    # 2a. Strict game docs (for training interactions)
    # ------------------------------------------------------------------ #
    logger.info("Building strict game documents (min_reviews=%d) …",
                tr_cfg["min_reviews_per_game"])
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
    # 2b. Broad game docs (for retrieval index — includes 0-review games)
    # ------------------------------------------------------------------ #
    logger.info("Building broad game documents (min_reviews=0) …")
    game_docs_retrieval = build_game_documents(
        games=games,
        reviews=reviews,
        min_reviews=0,
        bayesian_prior_mean=pp_cfg.get("bayesian_prior_mean", 3.5),
        bayesian_prior_weight=pp_cfg.get("bayesian_prior_weight", 10),
    )
    game_docs_retrieval.to_parquet(data_dir / "game_docs_retrieval.parquet", index=False)
    logger.info("Saved game_docs_retrieval.parquet (%d games)", len(game_docs_retrieval))

    # ------------------------------------------------------------------ #
    # 3. Interactions + split  (uses strict game_docs for bayesian_avg)
    # ------------------------------------------------------------------ #
    threshold = pp_cfg.get("rating_deviation_threshold", 0.25)
    logger.info("Building interaction matrix (threshold=±%.2f) …", threshold)
    interactions = build_interactions(
        reviews=reviews,
        users=users,
        game_docs=game_docs,
        rating_deviation_threshold=threshold,
        min_reviews_per_user=tr_cfg["min_reviews_per_user"],
        min_reviews_per_game=tr_cfg["min_reviews_per_game"],
    )
    logger.info(
        "Interactions: %d total  (%d positive, %d negative)",
        len(interactions),
        (interactions["label"] == 1).sum(),
        (interactions["label"] == 0).sum(),
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
    # 4a. Training user profiles  (training-set positives only)
    # ------------------------------------------------------------------ #
    logger.info("Building training user profiles …")
    user_profiles = build_user_profiles(
        interactions=train,
        game_docs=game_docs,
        reviews=reviews,
        users=users,
    )
    user_profiles.to_parquet(data_dir / "user_profiles.parquet", index=False)
    logger.info("Saved user_profiles.parquet (%d users)", len(user_profiles))

    sample_profile_1 = user_profiles.iloc[0]
    logger.info(
        "Sample user profile 1:\n  userid=%s  name=%s\n  profile_text=%s",
        sample_profile_1["userid"], sample_profile_1.get("name", ""),
        sample_profile_1["profile_text"],
    )
    sample_profile_2 = user_profiles.iloc[1]
    logger.info(
        "Sample user profile 2:\n  userid=%s  name=%s\n  profile_text=%s",
        sample_profile_2["userid"], sample_profile_2.get("name", ""),
        sample_profile_2["profile_text"],
    )

    # ------------------------------------------------------------------ #
    # 4b. Retrieval user profiles  (all users with ≥1 positive interaction)
    # ------------------------------------------------------------------ #
    logger.info("Building retrieval user profiles (all users with ≥1 positive) …")
    all_interactions = build_interactions(
        reviews=reviews,
        users=users,
        game_docs=game_docs_retrieval,
        rating_deviation_threshold=threshold,
        min_reviews_per_user=1,
        min_reviews_per_game=1,
    )
    all_positives = all_interactions[all_interactions["label"] == 1][
        ["userid", "gameid", "label"]
    ].copy()
    user_profiles_retrieval = build_user_profiles(
        interactions=all_positives,
        game_docs=game_docs_retrieval,
        reviews=reviews,
        users=users,
    )
    user_profiles_retrieval.to_parquet(
        data_dir / "user_profiles_retrieval.parquet", index=False
    )
    logger.info(
        "Saved user_profiles_retrieval.parquet (%d users)", len(user_profiles_retrieval)
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
