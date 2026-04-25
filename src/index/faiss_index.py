"""FAISS-based ANN index for fast game retrieval."""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Union

import faiss
import numpy as np

logger = logging.getLogger(__name__)

_INDEX_FILE  = "game.index"
_ID_MAP_FILE = "id_map.pkl"


class GameIndex:
    """
    Wraps a FAISS index with a game-ID ↔ integer-position mapping.

    Supports flat (exact) and HNSW (approximate) index types.
    Inner-product similarity is used, so embeddings must be L2-normalised
    before indexing or querying.
    """

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        self.game_ids: List[str] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        embeddings: np.ndarray,
        game_ids: List[str],
        index_type: str = "flat",
        hnsw_m: int = 32,
    ) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        dim = embeddings.shape[1]

        if index_type == "hnsw":
            idx = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = 200
        else:
            idx = faiss.IndexFlatIP(dim)

        idx.add(embeddings)
        self._index = idx
        self.game_ids = list(game_ids)
        logger.info(
            "Built %s FAISS index: %d vectors, dim=%d", index_type, len(game_ids), dim
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query_emb: np.ndarray, top_k: int = 100
    ) -> List[Tuple[str, float]]:
        if self._index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")
        q = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
        scores, indices = self._index.search(q, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.game_ids):
                results.append((self.game_ids[idx], float(score)))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_dir: Union[str, Path]) -> None:
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / _INDEX_FILE))
        with open(path / _ID_MAP_FILE, "wb") as f:
            pickle.dump(self.game_ids, f)
        logger.info("Index saved to %s", path)

    @classmethod
    def load(cls, index_dir: Union[str, Path]) -> "GameIndex":
        path = Path(index_dir)
        obj = cls()
        obj._index = faiss.read_index(str(path / _INDEX_FILE))
        with open(path / _ID_MAP_FILE, "rb") as f:
            obj.game_ids = pickle.load(f)
        logger.info(
            "Loaded index from %s (%d vectors)", path, obj._index.ntotal
        )
        return obj
