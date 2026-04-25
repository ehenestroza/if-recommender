"""Stage 1: ANN retrieval via FAISS."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.index.faiss_index import GameIndex

logger = logging.getLogger(__name__)


class Retriever:
    """
    Encode a query and retrieve the nearest game embeddings from FAISS.

    Supports three query modes:
      text     – encode a free-text string directly
      userid   – look up the user's pre-built profile text, then encode
      game_ids – average the embeddings of seed games ("more like these")
    """

    def __init__(
        self,
        model: Optional[SentenceTransformer],
        index: GameIndex,
        user_profiles: Dict[str, str],
        game_embeddings: Dict[str, np.ndarray],
    ) -> None:
        self.bi_encoder = model          # may be patched after construction
        self.index = index
        self.user_profiles = user_profiles
        self.game_embeddings = game_embeddings

    # ------------------------------------------------------------------
    # Query encoders
    # ------------------------------------------------------------------

    def _encode_text(self, text: str) -> np.ndarray:
        emb = self.bi_encoder.encode([text], normalize_embeddings=True)
        return emb[0]

    def _encode_userid(self, userid: str) -> Optional[np.ndarray]:
        profile = self.user_profiles.get(userid)
        if not profile:
            logger.warning("No profile found for user '%s'", userid)
            return None
        return self._encode_text(profile)

    def _encode_game_ids(self, game_ids: List[str]) -> Optional[np.ndarray]:
        embs = [self.game_embeddings[gid] for gid in game_ids if gid in self.game_embeddings]
        if not embs:
            logger.warning("None of the seed game IDs found in embeddings")
            return None
        avg = np.mean(embs, axis=0).astype(np.float32)
        norm = np.linalg.norm(avg)
        return avg / norm if norm > 0 else avg

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query,
        query_type: str = "text",
        top_k: int = 100,
    ) -> List[Tuple[str, float]]:
        """
        Return a ranked list of (gameid, score) pairs.

        query: str for text/userid, list[str] for game_ids
        """
        if query_type == "text":
            emb = self._encode_text(query)
        elif query_type == "userid":
            emb = self._encode_userid(query)
        elif query_type == "game_ids":
            emb = self._encode_game_ids(query)
        else:
            raise ValueError(f"Unknown query_type: {query_type!r}")

        if emb is None:
            return []

        return self.index.search(emb, top_k=top_k)
