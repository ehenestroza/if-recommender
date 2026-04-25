#!/usr/bin/env python
"""
Step 4 – Encode all games and build the FAISS retrieval index.

What this does
--------------
1. Loads the fine-tuned bi-encoder from models/two_tower/
2. Encodes all game documents → normalised float32 embeddings
3. Saves embeddings as a numpy array (for "more like these" queries)
4. Builds a FAISS index (flat or HNSW per config) and saves it to outputs/

Outputs in outputs/
-------------------
  game.index      – FAISS index
  id_map.pkl      – ordered list of game IDs (position → gameid)
  game_embs.npy   – (N, D) embedding matrix  (same order as id_map)
  gameid_to_idx.pkl – reverse map: gameid → integer index

Usage
-----
    python scripts/04_build_index.py [--model-dir models/two_tower]
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.index.faiss_index import GameIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--model-dir", default=None,
                        help="Override model directory (default: config paths.model_dir/two_tower)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir   = Path(cfg["paths"]["data_dir"])
    model_dir  = Path(args.model_dir or Path(cfg["paths"]["model_dir"]) / "two_tower")
    index_dir  = Path(cfg["paths"]["index_dir"])
    model_cfg  = cfg["model"]
    retr_cfg   = cfg["retrieval"]

    index_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
    logger.info("Loading encoder from %s …", model_dir)
    model = SentenceTransformer(str(model_dir))
    model.max_seq_length = model_cfg["max_seq_length"]

    # ------------------------------------------------------------------ #
    # Load game documents
    # ------------------------------------------------------------------ #
    logger.info("Loading game docs …")
    game_docs = pd.read_parquet(data_dir / "game_docs.parquet")
    game_ids  = game_docs["gameid"].tolist()
    doc_texts = game_docs["doc_text"].tolist()
    logger.info("Games to encode: %d", len(game_ids))

    # ------------------------------------------------------------------ #
    # Encode
    # ------------------------------------------------------------------ #
    logger.info("Encoding game documents (this may take a few minutes) …")
    embeddings = model.encode(
        doc_texts,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)
    logger.info("Embeddings shape: %s", embeddings.shape)

    # ------------------------------------------------------------------ #
    # Save raw embeddings (used for "more like these" queries)
    # ------------------------------------------------------------------ #
    emb_path = index_dir / "game_embs.npy"
    np.save(emb_path, embeddings)
    logger.info("Saved embeddings to %s", emb_path)

    gameid_to_idx = {gid: i for i, gid in enumerate(game_ids)}
    with open(index_dir / "gameid_to_idx.pkl", "wb") as f:
        pickle.dump(gameid_to_idx, f)

    # ------------------------------------------------------------------ #
    # Build and save FAISS index
    # ------------------------------------------------------------------ #
    index_type = retr_cfg.get("index_type", "flat")
    hnsw_m     = retr_cfg.get("hnsw_m", 32)

    logger.info("Building FAISS index (type=%s) …", index_type)
    game_index = GameIndex()
    game_index.build(embeddings, game_ids, index_type=index_type, hnsw_m=hnsw_m)
    game_index.save(index_dir)

    logger.info("✓ Index built and saved to %s", index_dir)


if __name__ == "__main__":
    main()
