#!/usr/bin/env python
"""
Step 3 – Fine-tune the two-tower bi-encoder.

Training setup
--------------
Loss      : MultipleNegativesRankingLoss (in-batch negatives, symmetric InfoNCE)
Data      : (user_profile_text, positive_game_doc) pairs from training-split positives
Backbone  : sentence-transformers/all-MiniLM-L6-v2  (configurable)
Optimiser : AdamW with linear warmup + cosine decay  (handled by ST Trainer)

The fine-tuned model is saved to models/two_tower/.

Why MultipleNegativesRankingLoss?
----------------------------------
Each positive pair (a_i, p_i) in a batch of size B also sees B-1 other
positives {p_j | j≠i} as *in-batch negatives*.  The loss is:

    L = -log [ exp(sim(a_i, p_i) / τ) /
               Σ_{j=1..B} exp(sim(a_i, p_j) / τ) ]

This scales well, requires no explicit negative mining, and matches DPR /
SimCSE / E5 training paradigms — all of which are directly relevant to
the Hugging Face ecosystem signal we're building toward.

Usage
-----
    python scripts/03_train_two_tower.py [--epochs N] [--batch-size N]
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import PairDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# ── sentence-transformers expects InputExample objects ───────────────────────

class STInputExampleDataset(torch.utils.data.Dataset):
    """Wraps our PairDataset to yield InputExample objects for ST trainer."""

    def __init__(self, pair_dataset: PairDataset) -> None:
        self.pairs = pair_dataset.pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> InputExample:
        anchor, positive = self.pairs[idx]
        return InputExample(texts=[anchor, positive])


def build_ir_evaluator(
    val_interactions: pd.DataFrame,
    user_profiles: pd.DataFrame,
    game_docs: pd.DataFrame,
    name: str = "val",
) -> InformationRetrievalEvaluator | None:
    """
    Build an InformationRetrievalEvaluator from validation positives.

    queries   : {userid: profile_text}
    corpus    : {gameid: doc_text}
    relevant  : {userid: {gameid, …}}
    """
    profile_map = dict(zip(user_profiles["userid"], user_profiles["profile_text"]))
    doc_map     = dict(zip(game_docs["gameid"],     game_docs["doc_text"]))

    val_pos = val_interactions[val_interactions["label"] == 1]

    queries  = {}
    relevant = {}
    for uid, grp in val_pos.groupby("userid"):
        if uid not in profile_map:
            continue
        queries[uid]  = profile_map[uid]
        relevant[uid] = set(grp["gameid"])

    # Only keep game IDs that appear in corpus
    corpus = {gid: doc for gid, doc in doc_map.items()}

    logger.info(
        "IR evaluator: %d queries, %d corpus items", len(queries), len(corpus)
    )
    if not queries:
        logger.warning(
            "No val users have pre-built profiles (profiles are derived from "
            "training positives only). Skipping IR evaluator."
        )
        return None

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        name=name,
        show_progress_bar=True,
        precision_recall_at_k=[1, 5, 10],
        ndcg_at_k=[10],
        mrr_at_k=[10],
        batch_size=64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-eval",      action="store_true",
                        help="Skip IR evaluation (faster, useful for debugging)")
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Use a random fraction of training pairs, e.g. 0.05 for 5%%")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir  = Path(cfg["paths"]["data_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    tr_cfg    = cfg["training"]
    model_cfg = cfg["model"]

    epochs     = args.epochs     or tr_cfg["epochs"]
    batch_size = args.batch_size or tr_cfg["batch_size"]

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    logger.info("Loading prepared data …")
    game_docs      = pd.read_parquet(data_dir / "game_docs.parquet")
    user_profiles  = pd.read_parquet(data_dir / "user_profiles.parquet")
    interactions   = pd.read_parquet(data_dir / "interactions.parquet")

    train_ixns = interactions[interactions["split"] == "train"]
    val_ixns   = interactions[interactions["split"] == "val"]

    # ------------------------------------------------------------------ #
    # Build datasets
    # ------------------------------------------------------------------ #
    logger.info("Building pair dataset …")
    pair_ds = PairDataset(train_ixns, user_profiles, game_docs)
    if args.sample_frac is not None:
        import random
        k = max(1, int(len(pair_ds.pairs) * args.sample_frac))
        pair_ds.pairs = random.sample(pair_ds.pairs, k)
        logger.info("Sampled %.1f%% → %d pairs", args.sample_frac * 100, k)
    st_ds   = STInputExampleDataset(pair_ds)
    logger.info("Training pairs: %d", len(st_ds))

    train_loader = DataLoader(
        st_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep 0 for MPS/CUDA compatibility on laptops
    )

    # ------------------------------------------------------------------ #
    # Model + loss
    # ------------------------------------------------------------------ #
    logger.info("Initialising model: %s", model_cfg["base_model"])
    model = SentenceTransformer(model_cfg["base_model"])
    model.max_seq_length = model_cfg["max_seq_length"]

    loss_fn = MultipleNegativesRankingLoss(model)

    # ------------------------------------------------------------------ #
    # Evaluator
    # ------------------------------------------------------------------ #
    evaluator = None
    if not args.no_eval and len(val_ixns) > 0:
        logger.info("Building IR evaluator …")
        evaluator = build_ir_evaluator(val_ixns, user_profiles, game_docs)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    model_out = model_dir / "two_tower"
    model_out.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = len(train_loader)
    warmup_steps = max(1, int(steps_per_epoch * epochs * tr_cfg["warmup_ratio"]))

    logger.info(
        "Starting training | epochs=%d | batch=%d | steps/epoch=%d | warmup=%d",
        epochs, batch_size, steps_per_epoch, warmup_steps,
    )

    model.fit(
        train_objectives=[(train_loader, loss_fn)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": tr_cfg["learning_rate"]},
        output_path=str(model_out),
        show_progress_bar=True,
        evaluation_steps=steps_per_epoch if evaluator is not None else 0,
        save_best_model=evaluator is not None,
    )

    logger.info("✓ Training complete. Model saved to %s", model_out)


if __name__ == "__main__":
    main()
