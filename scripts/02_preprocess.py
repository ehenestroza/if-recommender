#!/usr/bin/env python
"""
Step 2 – Prepare training data from raw Parquet files.

What this script does
---------------------
1. Loads raw tables from data/
2. Builds game document strings (item-tower inputs)
3. Assembles interaction matrix from all signals and splits train/val/test
4. Builds user profile strings (query-tower inputs)
5. Saves processed artefacts to data/

After this step data/ will contain:
  game_docs.parquet     – gameid, title, author, genre, doc_text
  interactions.parquet  – all interactions with label + split column
  user_profiles.parquet – userid, profile_text

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

    # ------------------------------------------------------------------ #
    # 1. Load raw tables
    # ------------------------------------------------------------------ #
    logger.info("Loading raw tables …")
    games        = load_parquet(data_dir, "games.parquet")
    reviews      = load_parquet(data_dir, "reviews.parquet")
    gametags     = load_parquet(data_dir, "gametags.parquet")
    wishlists    = load_parquet(data_dir, "wishlists.parquet")
    playedgames  = load_parquet(data_dir, "playedgames.parquet")

    logger.info(
        "Loaded: games=%d, reviews=%d, gametags=%d, wishlists=%d, playedgames=%d",
        len(games), len(reviews), len(gametags), len(wishlists), len(playedgames),
    )

    # ------------------------------------------------------------------ #
    # 2. Game documents
    # ------------------------------------------------------------------ #
    logger.info("Building game documents …")
    game_docs = build_game_documents(games, gametags)
    game_docs.to_parquet(data_dir / "game_docs.parquet", index=False)
    logger.info("Saved game_docs.parquet (%d games)", len(game_docs))

    # Sample a few for inspection
    for _, row in game_docs.head(3).iterrows():
        logger.info("  [%s] %s\n      → %s…", row["gameid"], row["title"], row["doc_text"][:120])

    # ------------------------------------------------------------------ #
    # 3. Interactions + split
    # ------------------------------------------------------------------ #
    logger.info("Building interaction matrix …")
    interactions = build_interactions(
        reviews=reviews,
        wishlists=wishlists,
        playedgames=playedgames,
        min_rating_positive=tr_cfg["min_rating_positive"],
        max_rating_negative=tr_cfg["max_rating_negative"],
        min_reviews_per_user=tr_cfg["min_reviews_per_user"],
        min_reviews_per_game=tr_cfg["min_reviews_per_game"],
    )

    train, val, test = split_interactions(
        interactions,
        val_frac=tr_cfg["val_frac"],
        test_frac=tr_cfg["test_frac"],
    )

    # Add split label and save as one file for convenience
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
        gametags=gametags,
        min_rating_positive=tr_cfg["min_rating_positive"],
    )
    user_profiles.to_parquet(data_dir / "user_profiles.parquet", index=False)
    logger.info("Saved user_profiles.parquet (%d users)", len(user_profiles))

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
