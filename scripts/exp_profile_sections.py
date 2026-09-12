#!/usr/bin/env python
"""
Which profile sections earn their place? Raw-retrieval ablation.

Re-runs the test-split retrieval evaluation with one section stripped from
every profile at query time — the encoders are not retrained, so this measures
what the trained query tower gets from each section, not what a tower trained
without it would do. Cheap (a minute a run) and enough to rank the sections.

Usage
-----
    python scripts/exp_profile_sections.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.env import configure_logging
configure_logging()

from src.data.preprocessor import PROFILE_SECTIONS, format_profile_text, parse_profile_sections  # noqa: E402
from src.pipeline.ranker import evaluate_retrieval  # noqa: E402

logger = logging.getLogger(__name__)
KS = (5, 10, 50)


def strip(text: str, drop: set) -> str:
    s = parse_profile_sections(text)
    return format_profile_text(*[[] if label in drop else s.get(label, []) for label in PROFILE_SECTIONS])


def neg_hit(preds, negatives, k):
    users = [u for u in negatives if u in preds]
    return float(np.mean([len(set(preds[u][:k]) & negatives[u]) / len(negatives[u]) for u in users]))


def main() -> None:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    data_dir, model_dir, index_dir = (Path(cfg["paths"][k]) for k in ("data_dir", "model_dir", "index_dir"))

    import pickle
    embs = np.load(index_dir / "game_embs.npy")
    with open(index_dir / "gameid_to_idx.pkl", "rb") as f:
        g2i = pickle.load(f)
    ids = [None] * len(g2i)
    for g, i in g2i.items():
        ids[i] = g

    profiles = pd.read_parquet(data_dir / "user_profiles.parquet")
    profile_map = dict(zip(profiles["userid"], profiles["profile_text"]))
    inter = pd.read_parquet(data_dir / "interactions.parquet")
    test = inter[inter["split"] == "test"]
    truth = test[test["label"] == 1].groupby("userid")["gameid"].apply(set).to_dict()
    negatives = test[test["label"] == 0].groupby("userid")["gameid"].apply(set).to_dict()
    users = [u for u in truth if u in profile_map]

    enc = SentenceTransformer(str(model_dir / "query_encoder"))
    enc.max_seq_length = cfg["model"]["max_seq_length"]

    variants = [("full", set())] + [(f"– {s}", {s}) for s in PROFILE_SECTIONS[2:]] + \
               [("systems+tags only", set(PROFILE_SECTIONS[2:]))]
    rows = []
    for name, drop in variants:
        q = enc.encode([strip(profile_map[u], drop) for u in users], batch_size=128,
                       normalize_embeddings=True, show_progress_bar=False)
        sims = q @ embs.T
        preds = {}
        for n, u in enumerate(users):
            top = np.argpartition(-sims[n], 50)[:50]
            preds[u] = [ids[i] for i in top[np.argsort(-sims[n][top])]]
        m = evaluate_retrieval(preds, truth, KS)
        m["NegHit@10"] = neg_hit(preds, negatives, 10)
        m["NegHit@50"] = neg_hit(preds, negatives, 50)
        rows.append((name, m))
        logger.info("%-20s %s", name, " ".join(f"{k}={v:.4f}" for k, v in m.items()))

    cols = ["MRR", "Recall@5", "Recall@10", "Recall@50", "NDCG@10", "NegHit@10", "NegHit@50"]
    print(f"\n{'variant':<20}" + "".join(f"{c:>11}" for c in cols))
    for name, m in rows:
        print(f"{name:<20}" + "".join(f"{m[c]:>11.4f}" for c in cols))


if __name__ == "__main__":
    main()
