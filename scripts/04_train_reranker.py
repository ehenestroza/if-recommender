#!/usr/bin/env python
"""
Step 4 – Fine-tune the cross-encoder reranker.

Uses training-split interactions to fine-tune a cross-encoder on
(user_profile, game_doc, label) triples. label=1 for positives,
label=0 for negatives. The cross-encoder is trained with BCE loss
(num_labels=1) so its raw logit can be converted to a probability
via sigmoid at inference time.

Output: models/reranker/

Usage
-----
    python scripts/04_train_reranker.py [--epochs N] [--batch-size N]
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.env import configure_logging
configure_logging()

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir  = Path(cfg["paths"]["data_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    rr_cfg    = cfg.get("reranker_training", {})
    model_cfg = cfg["model"]

    epochs     = args.epochs     or rr_cfg.get("epochs",     2)
    batch_size = args.batch_size or rr_cfg.get("batch_size", 16)
    lr         = rr_cfg.get("learning_rate", 2e-5)

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    logger.info("Loading data …")
    game_docs     = pd.read_parquet(data_dir / "game_docs.parquet")
    user_profiles = pd.read_parquet(data_dir / "user_profiles.parquet")
    interactions  = pd.read_parquet(data_dir / "interactions.parquet")

    train_ixns = interactions[interactions["split"] == "train"]

    profile_map = dict(zip(user_profiles["userid"], user_profiles["profile_text"]))
    doc_map     = dict(zip(game_docs["gameid"],     game_docs["doc_text"]))

    # ------------------------------------------------------------------ #
    # Build training examples
    # ------------------------------------------------------------------ #
    logger.info("Building cross-encoder training examples …")
    train_examples = []
    n_missing = 0
    for _, row in train_ixns.iterrows():
        uid   = row["userid"]
        gid   = row["gameid"]
        label = int(row["label"])
        query = profile_map.get(uid, "")
        doc   = doc_map.get(gid, "")
        if not query or not doc:
            n_missing += 1
            continue
        train_examples.append(InputExample(texts=[query, doc], label=float(label)))

    n_pos = sum(1 for e in train_examples if e.label == 1.0)
    n_neg = sum(1 for e in train_examples if e.label == 0.0)
    logger.info(
        "Examples: %d total  (%d positive, %d negative)  [%d skipped — missing profile/doc]",
        len(train_examples), n_pos, n_neg, n_missing,
    )

    train_loader = DataLoader(
        train_examples, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # ------------------------------------------------------------------ #
    # Load cross-encoder with num_labels=1 (regression / BCE mode)
    # ------------------------------------------------------------------ #
    logger.info("Loading cross-encoder: %s", model_cfg["reranker_model"])
    model = CrossEncoder(model_cfg["reranker_model"], num_labels=1)

    # ------------------------------------------------------------------ #
    # Fine-tune
    # ------------------------------------------------------------------ #
    reranker_out = model_dir / "reranker"
    reranker_out.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Fine-tuning | epochs=%d | batch=%d | lr=%.2e",
        epochs, batch_size, lr,
    )
    model.fit(
        train_dataloader=train_loader,
        epochs=epochs,
        optimizer_params={"lr": lr},
        show_progress_bar=True,
    )

    # save_pretrained writes modules.json so CrossEncoder(path) can reload it
    logger.info("Saving reranker to %s …", reranker_out)
    model.save_pretrained(str(reranker_out))
    logger.info("✓ Reranker saved to %s", reranker_out)


if __name__ == "__main__":
    main()
