#!/usr/bin/env python
"""
Step 3 – Fine-tune the asymmetric two-tower bi-encoder.

Two separate SentenceTransformer models are trained with distinct weights:
  query_encoder  →  encodes user profile text
  doc_encoder    →  encodes game document text

Training uses symmetric InfoNCE loss: each (query_i, doc_i) pair treats
{doc_j | j≠i} as negatives for query_i and {query_j | j≠i} as negatives
for doc_i. Both encoders are updated jointly via a shared AdamW optimizer
with linear warmup.

Outputs: models/query_encoder/, models/doc_encoder/

Usage
-----
    python scripts/03_train_two_tower.py [--epochs N] [--batch-size N]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.env import configure_logging
configure_logging()

from src.data.dataset import PairDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_batch(model: SentenceTransformer, texts: List[str]) -> torch.Tensor:
    """Tokenize and forward-pass texts, returning L2-normalised embeddings."""
    device = next(model.parameters()).device
    features = model.tokenize(texts)
    # Only tensors can be moved to device; the dict may also contain strings
    features = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in features.items()}
    out = model(features)
    return F.normalize(out["sentence_embedding"], dim=-1)


def _infonce_loss(q: torch.Tensor, d: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric in-batch InfoNCE loss."""
    logits = (q @ d.T) / temperature          # (B, B)
    labels = torch.arange(len(q), device=q.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


@torch.no_grad()
def _evaluate_val(
    query_encoder: SentenceTransformer,
    doc_encoder: SentenceTransformer,
    val_pos: pd.DataFrame,
    profile_map: Dict[str, str],
    doc_map: Dict[str, str],
    top_k: int = 10,
) -> Dict[str, float]:
    """Recall@K and MRR on val positives using the asymmetric encoders."""
    query_encoder.eval()
    doc_encoder.eval()

    all_gids = list(doc_map.keys())
    doc_embs = doc_encoder.encode(
        [doc_map[g] for g in all_gids],
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    val_users = [u for u in val_pos["userid"].unique() if u in profile_map]
    if not val_users:
        query_encoder.train()
        doc_encoder.train()
        return {f"Recall@{top_k}": 0.0, "MRR": 0.0}

    q_embs = query_encoder.encode(
        [profile_map[u] for u in val_users],
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    sim = q_embs @ doc_embs.T                 # (M, N)
    gt  = val_pos.groupby("userid")["gameid"].apply(set).to_dict()

    recall_sum = mrr_sum = 0.0
    n = 0
    for i, uid in enumerate(val_users):
        if uid not in gt:
            continue
        relevant  = gt[uid]
        top_idx   = np.argsort(-sim[i])[:top_k]
        top_gids  = [all_gids[j] for j in top_idx]

        recall_sum += sum(1 for g in top_gids if g in relevant) / len(relevant)
        for rank, g in enumerate(top_gids, start=1):
            if g in relevant:
                mrr_sum += 1.0 / rank
                break
        n += 1

    query_encoder.train()
    doc_encoder.train()

    if n == 0:
        return {f"Recall@{top_k}": 0.0, "MRR": 0.0}
    return {f"Recall@{top_k}": recall_sum / n, "MRR": mrr_sum / n}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--batch-size",  type=int,   default=None)
    parser.add_argument("--no-eval",     action="store_true",
                        help="Skip per-epoch validation (faster)")
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="Train on a random fraction of pairs, e.g. 0.1")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir  = Path(cfg["paths"]["data_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    tr_cfg    = cfg["training"]
    model_cfg = cfg["model"]

    epochs      = args.epochs     or tr_cfg["epochs"]
    batch_size  = args.batch_size or tr_cfg["batch_size"]
    temperature = tr_cfg.get("temperature", 0.07)

    # ------------------------------------------------------------------ #
    # Device
    # ------------------------------------------------------------------ #
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    logger.info("Loading prepared data …")
    game_docs     = pd.read_parquet(data_dir / "game_docs.parquet")
    user_profiles = pd.read_parquet(data_dir / "user_profiles.parquet")
    interactions  = pd.read_parquet(data_dir / "interactions.parquet")

    train_ixns = interactions[interactions["split"] == "train"]
    val_ixns   = interactions[interactions["split"] == "val"]

    profile_map = dict(zip(user_profiles["userid"], user_profiles["profile_text"]))
    doc_map     = dict(zip(game_docs["gameid"],     game_docs["doc_text"]))

    # ------------------------------------------------------------------ #
    # Build pair dataset
    # ------------------------------------------------------------------ #
    logger.info("Building pair dataset …")
    pair_ds = PairDataset(train_ixns, user_profiles, game_docs)
    if args.sample_frac is not None:
        import random
        k = max(1, int(len(pair_ds.pairs) * args.sample_frac))
        pair_ds.pairs = random.sample(pair_ds.pairs, k)
        logger.info("Sampled %.1f%% → %d pairs", args.sample_frac * 100, k)

    logger.info("Training pairs: %d", len(pair_ds))

    train_loader = DataLoader(
        pair_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: (
            [item[0] for item in batch],
            [item[1] for item in batch],
        ),
    )

    # ------------------------------------------------------------------ #
    # Models (separate weights, same initialisation)
    # ------------------------------------------------------------------ #
    base = model_cfg["base_model"]
    max_seq = model_cfg["max_seq_length"]

    logger.info("Initialising query encoder: %s", base)
    query_encoder = SentenceTransformer(base)
    query_encoder.max_seq_length = max_seq
    query_encoder = query_encoder.to(device)

    logger.info("Initialising doc encoder: %s", base)
    doc_encoder = SentenceTransformer(base)
    doc_encoder.max_seq_length = max_seq
    doc_encoder = doc_encoder.to(device)

    # ------------------------------------------------------------------ #
    # Optimizer + LR schedule
    # ------------------------------------------------------------------ #
    all_params   = list(query_encoder.parameters()) + list(doc_encoder.parameters())
    optimizer    = AdamW(all_params, lr=tr_cfg["learning_rate"])
    total_steps  = len(train_loader) * epochs
    warmup_steps = max(1, int(total_steps * tr_cfg["warmup_ratio"]))
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    logger.info(
        "Starting training | epochs=%d | batch=%d | steps/epoch=%d | warmup=%d | temp=%.3f",
        epochs, batch_size, len(train_loader), warmup_steps, temperature,
    )

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    val_pos = val_ixns[val_ixns["label"] == 1]

    for epoch in range(1, epochs + 1):
        query_encoder.train()
        doc_encoder.train()

        epoch_loss = 0.0
        for queries, docs in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            q_embs = _encode_batch(query_encoder, queries)
            d_embs = _encode_batch(doc_encoder,   docs)

            loss = _infonce_loss(q_embs, d_embs, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        logger.info("Epoch %d/%d | avg_loss=%.4f", epoch, epochs, avg_loss)

        if not args.no_eval and len(val_pos) > 0:
            metrics = _evaluate_val(
                query_encoder, doc_encoder, val_pos, profile_map, doc_map
            )
            logger.info(
                "Val | Recall@10=%.4f | MRR=%.4f",
                metrics["Recall@10"], metrics["MRR"],
            )

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    query_out = model_dir / "query_encoder"
    doc_out   = model_dir / "doc_encoder"
    query_out.mkdir(parents=True, exist_ok=True)
    doc_out.mkdir(parents=True, exist_ok=True)

    logger.info("Saving query encoder → %s", query_out)
    query_encoder.save(str(query_out))

    logger.info("Saving doc encoder   → %s", doc_out)
    doc_encoder.save(str(doc_out))

    logger.info("✓ Training complete.")


if __name__ == "__main__":
    main()
