#!/usr/bin/env python
"""
Step 3b – Fine-tune the item-to-item encoder.

One SentenceTransformer, one input shape: a game document. It is trained so
that games which belong together in someone's mind — a user's liked games, a
wishlist, the entries of one poll, one member's recommended list — embed close
to each other. `game` and `author` searches then become nearest-neighbour
lookups in this space (see src/pipeline/items.py), instead of dressing a game
up as a user profile and asking the profile encoder about it.

Pairs are sampled afresh each epoch from `data/item_sets.parquet` rather than
enumerated: a user with 1,500 liked games would otherwise supply a million
pairs and drown everyone else. Each set contributes at most `pairs_per_set_cap`
pairs, scaled by its source's weight. Symmetric InfoNCE with in-batch
negatives, masking the negatives that are not — the same game, or another pair
from the same set.

The weights saved are from the epoch with the best validation item-to-item
Recall@10: for each validation positive, query with one of that user's
training positives and look for the held-out game.

Outputs: models/item_encoder/

Usage
-----
    python scripts/03b_train_item_encoder.py [--epochs N] [--batch-size N] [--save-last]
"""

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.env import configure_logging
configure_logging()

logger = logging.getLogger(__name__)


def _encode_batch(model: SentenceTransformer, texts: List[str]) -> torch.Tensor:
    device = next(model.parameters()).device
    features = model.tokenize(texts)
    features = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in features.items()}
    return F.normalize(model(features)["sentence_embedding"], dim=-1)


def _cpu_state(model: SentenceTransformer) -> dict:
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _masked_infonce(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor,
                    temperature: float) -> torch.Tensor:
    """
    Symmetric in-batch InfoNCE where `mask[i, j]` marks (i, j) pairs that must
    not count as negatives: the same game on both sides, or two pairs drawn
    from one set. The diagonal is always kept.
    """
    logits = (a @ b.T) / temperature
    logits = logits.masked_fill(mask, float("-inf"))
    labels = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def sample_pairs(sets: Dict[Tuple[str, str], List[str]], weights: Dict[str, float],
                 cap: int, rng: np.random.Generator) -> List[Tuple[str, str, int]]:
    """
    One epoch's (gameid_a, gameid_b, set_index) triples.

    A set of n games yields ceil(weight * min(n, cap)) pairs, each a random
    unordered draw of two of its members — so every member has a chance to
    appear, but no set outweighs a hundred others.
    """
    pairs = []
    for k, ((source, _setid), gids) in enumerate(sets.items()):
        n = len(gids)
        for _ in range(math.ceil(weights.get(source, 1.0) * min(n, cap))):
            i, j = rng.choice(n, size=2, replace=False)
            pairs.append((gids[i], gids[j], k))
    order = rng.permutation(len(pairs))
    return [pairs[i] for i in order]


def item_queries(interactions: pd.DataFrame, split: str, known: set,
                 seed: int = 0) -> Tuple[List[str], Dict[str, set]]:
    """
    Item-to-item evaluation queries for one split: for each user with held-out
    positives, one of their training positives as the seed, and the held-out
    games as the targets. The seed is drawn with a fixed generator so every
    epoch and every script scores the same queries.
    """
    rng = np.random.default_rng(seed)
    train_pos = interactions[(interactions["split"] == "train") & (interactions["label"] == 1)]
    held = interactions[(interactions["split"] == split) & (interactions["label"] == 1)]
    train_by_user = train_pos.groupby("userid")["gameid"].apply(
        lambda s: sorted(g for g in s if g in known)).to_dict()
    seeds, targets = [], {}
    for uid, grp in held.groupby("userid"):
        pool = train_by_user.get(uid)
        gt = {g for g in grp["gameid"] if g in known}
        if not pool or not gt:
            continue
        seed_gid = pool[rng.integers(len(pool))]
        seeds.append(seed_gid)
        targets[seed_gid + "|" + uid] = gt
    return seeds, targets


@torch.no_grad()
def evaluate_item_recall(model: SentenceTransformer, doc_map: Dict[str, str],
                         seeds: List[str], targets: Dict[str, set],
                         top_k: int = 10) -> Dict[str, float]:
    model.eval()
    gids = list(doc_map)
    embs = model.encode([doc_map[g] for g in gids], batch_size=128,
                        normalize_embeddings=True, show_progress_bar=False,
                        convert_to_numpy=True)
    row = {g: i for i, g in enumerate(gids)}
    recall = mrr = 0.0
    n = 0
    for key, gt in targets.items():
        seed_gid = key.split("|")[0]
        sims = embs @ embs[row[seed_gid]]
        sims[row[seed_gid]] = -np.inf
        top = np.argpartition(-sims, top_k)[:top_k]
        top = top[np.argsort(-sims[top])]
        hits = [gids[i] in gt for i in top]
        recall += sum(hits) / len(gt)
        if any(hits):
            mrr += 1.0 / (hits.index(True) + 1)
        n += 1
    model.train()
    return {f"Recall@{top_k}": recall / max(n, 1), "MRR": mrr / max(n, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-last",  action="store_true")
    parser.add_argument("--no-eval",    action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir  = Path(cfg["paths"]["data_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    it_cfg    = cfg["item_training"]
    epochs      = args.epochs or it_cfg["epochs"]
    batch_size  = args.batch_size or it_cfg["batch_size"]
    temperature = it_cfg.get("temperature", 0.05)
    weights     = it_cfg.get("source_weights", {})
    cap         = it_cfg.get("pairs_per_set_cap", 30)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    game_docs = pd.read_parquet(data_dir / "game_docs_retrieval.parquet")
    doc_map: Dict[str, str] = dict(zip(game_docs["gameid"], game_docs["doc_text"]))
    item_sets = pd.read_parquet(data_dir / "item_sets.parquet")
    item_sets = item_sets[item_sets["gameid"].isin(doc_map)]
    sets: Dict[Tuple[str, str], List[str]] = {
        key: sorted(grp["gameid"]) for key, grp in item_sets.groupby(["source", "setid"])
    }
    sets = {k: v for k, v in sets.items() if len(v) >= 2}
    logger.info("Item sets: %d (%s)", len(sets),
                ", ".join(f"{s}={sum(1 for k in sets if k[0] == s)}"
                          for s in sorted({k[0] for k in sets})))

    interactions = pd.read_parquet(data_dir / "interactions.parquet")
    val_seeds, val_targets = item_queries(interactions, "val", set(doc_map))
    logger.info("Validation queries: %d", len(val_targets))

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    init = it_cfg.get("init_from", "doc_encoder")
    init_path = str(model_dir / "doc_encoder") if init == "doc_encoder" else cfg["model"]["base_model"]
    logger.info("Initialising item encoder from %s", init_path)
    model = SentenceTransformer(init_path)
    model.max_seq_length = cfg["model"]["max_seq_length"]
    model = model.to(device)
    model.train()

    rng = np.random.default_rng(it_cfg.get("seed", 42))
    steps_per_epoch = math.ceil(len(sample_pairs(sets, weights, cap, np.random.default_rng(0))) / batch_size)
    total_steps  = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * it_cfg.get("warmup_ratio", 0.1)))
    optimizer = AdamW(model.parameters(), lr=it_cfg.get("learning_rate", 2e-5))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    logger.info("Training | epochs=%d | batch=%d | steps/epoch≈%d | warmup=%d | temp=%.3f | cap=%d",
                epochs, batch_size, steps_per_epoch, warmup_steps, temperature, cap)

    best_recall, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        pairs = sample_pairs(sets, weights, cap, rng)
        total_loss = 0.0
        bar = tqdm(range(0, len(pairs), batch_size), desc=f"Epoch {epoch}/{epochs}", leave=False)
        for start in bar:
            batch = pairs[start:start + batch_size]
            if len(batch) < 2:
                continue
            gid_a = [p[0] for p in batch]
            gid_b = [p[1] for p in batch]
            set_ix = torch.tensor([p[2] for p in batch], device=device)

            embs = _encode_batch(model, [doc_map[g] for g in gid_a] + [doc_map[g] for g in gid_b])
            a, b = embs[:len(batch)], embs[len(batch):]

            # Not negatives: the same game on both sides, or two pairs from
            # one set. The diagonal is the positive and stays.
            same_game = torch.tensor([[x == y for y in gid_b] for x in gid_a], device=device)
            same_set = set_ix[:, None] == set_ix[None, :]
            mask = (same_game | same_set) & ~torch.eye(len(batch), dtype=torch.bool, device=device)

            loss = _masked_infonce(a, b, mask, temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        n_steps = math.ceil(len(pairs) / batch_size)
        msg = f"Epoch {epoch}/{epochs} | pairs={len(pairs)} | loss={total_loss / n_steps:.4f}"
        if not args.no_eval and val_targets:
            metrics = evaluate_item_recall(model, doc_map, val_seeds, val_targets)
            msg += " | val " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            if metrics["Recall@10"] > best_recall:
                best_recall, best_state, best_epoch = metrics["Recall@10"], _cpu_state(model), epoch
        logger.info(msg)

    if best_state is not None and not args.save_last:
        logger.info("Restoring epoch %d (val Recall@10=%.4f)", best_epoch, best_recall)
        model.load_state_dict(best_state)

    out = model_dir / "item_encoder"
    model.save(str(out))
    logger.info("✓ Item encoder saved to %s", out)


if __name__ == "__main__":
    main()
