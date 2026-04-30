#!/usr/bin/env python
"""
Step 5 – Encode all games and build the FAISS retrieval index.

What this does
--------------
1. Loads the doc encoder from models/doc_encoder/ (falls back to models/two_tower/)
2. Encodes all game documents → normalised float32 embeddings
3. Saves embeddings as a numpy array (for "more like these" queries)
4. Builds a FAISS index (flat or HNSW per config) and saves it to outputs/
5. Loads the query encoder and encodes each game's profile text (no description)
   → game_query_embs.npy  (used for game_id query mode)

Outputs in outputs/
-------------------
  game.index          – FAISS index
  id_map.pkl          – ordered list of game IDs (position → gameid)
  game_embs.npy       – (N, D) doc-encoder embedding matrix
  game_query_embs.npy – (N, D) query-encoder embeddings of game profile texts
  gameid_to_idx.pkl   – reverse map: gameid → integer index

Usage
-----
    python scripts/05_build_index.py [--model-dir models/doc_encoder]
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

from src.utils.env import configure_logging
configure_logging()

from src.index.faiss_index import GameIndex

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--model-dir", default=None,
                        help="Override model directory (default: config paths.model_dir/two_tower)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir  = Path(cfg["paths"]["data_dir"])
    index_dir = Path(cfg["paths"]["index_dir"])

    # Prefer the asymmetric doc_encoder; fall back to legacy two_tower
    base_model_dir = Path(cfg["paths"]["model_dir"])
    if args.model_dir:
        model_dir = Path(args.model_dir)
    elif (base_model_dir / "doc_encoder").exists():
        model_dir = base_model_dir / "doc_encoder"
    else:
        model_dir = base_model_dir / "two_tower"
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
    logger.info("Loading game docs (retrieval set) …")
    game_docs = pd.read_parquet(data_dir / "game_docs_retrieval.parquet")
    game_ids  = game_docs["gameid"].tolist()
    doc_texts = game_docs["doc_text"].tolist()
    logger.info("Games to encode: %d", len(game_ids))

    # ------------------------------------------------------------------ #
    # Encode with doc encoder → FAISS index
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
    # Save raw doc embeddings + gameid index map
    # ------------------------------------------------------------------ #
    emb_path = index_dir / "game_embs.npy"
    np.save(emb_path, embeddings)
    logger.info("Saved game_embs.npy to %s", emb_path)

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

    # ------------------------------------------------------------------ #
    # Encode game profile texts with query encoder → game_query_embs.npy
    # These are used for game_id query mode (game as proxy for a user taste)
    # ------------------------------------------------------------------ #
    if "query_text" in game_docs.columns:
        query_encoder_dir = base_model_dir / "query_encoder"
        if not query_encoder_dir.exists():
            query_encoder_dir = model_dir  # fall back to the same model

        logger.info("Loading query encoder from %s …", query_encoder_dir)
        query_model = SentenceTransformer(str(query_encoder_dir))
        query_model.max_seq_length = model_cfg["max_seq_length"]

        query_texts = game_docs["query_text"].tolist()
        logger.info("Encoding game profile texts (no description) with query encoder …")
        query_embeddings = query_model.encode(
            query_texts,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        np.save(index_dir / "game_query_embs.npy", query_embeddings)
        logger.info("Saved game_query_embs.npy (shape %s)", query_embeddings.shape)
    else:
        logger.warning("game_docs has no 'query_text' column; skipping game_query_embs.npy")

    logger.info("✓ Index built and saved to %s", index_dir)


if __name__ == "__main__":
    main()
