"""Thin wrapper around a SentenceTransformer bi-encoder."""

import logging
from pathlib import Path
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TwoTowerModel:
    """
    Wraps a SentenceTransformer for symmetric query/item encoding.

    Both towers share the same weights (shared-encoder paradigm), which is
    parameter-efficient when queries and items live in the same text space.
    """

    def __init__(self, model_name_or_path: str, max_seq_length: int = 256) -> None:
        logger.info("Loading encoder: %s", model_name_or_path)
        self.model = SentenceTransformer(model_name_or_path)
        self.model.max_seq_length = max_seq_length

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path], max_seq_length: int = 256) -> "TwoTowerModel":
        obj = cls.__new__(cls)
        obj.model = SentenceTransformer(str(path))
        obj.model.max_seq_length = max_seq_length
        logger.info("Model loaded from %s", path)
        return obj
