"""
Item-to-item retrieval: "more like this game", and "in the spirit of this author".

The item encoder (`scripts/03b_train_item_encoder.py`) embeds game documents
so that games people like together sit close. Ranking in that space is a
nearest-neighbour search and nothing more — no cross-encoder. The reranker was
trained on (profile, document) pairs and has nothing to say about two
documents, and item-to-item encoders are ordinarily the whole model.

`ItemSpace.rank` returns the same `(scored, relevance)` shape as
`Reranker.rerank`, so the app, the precompute job and the evaluation share one
code path and cannot diverge. `scored` carries the blended selection score
(relevance and rating, weighted like the reranked modes) and `relevance` is
what a page shows.

Relevance is cosine rescaled, not raw cosine. The item space is narrow: two
games picked at random sit at cosine ~0.75 on average and a game's 500th
neighbour well above 0.8, so raw values would read as "everything matches". The
rescaling is affine and corpus-wide — `(cos - baseline) / (1 - baseline)`,
where `baseline` is the mean cosine of random pairs, measured when the index
is built — so it stays absolute: a query with only weak neighbours still looks
weak, which per-query normalisation would hide. On this scale a first
neighbour sits around 0.8 and the 500th around 0.4, comparable to what the
cross-encoder modes show.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.index.faiss_index import GameIndex

logger = logging.getLogger(__name__)

ITEM_INDEX_DIR = "item"            # under paths.index_dir
ITEM_EMBS_FILE = "item_embs.npy"   # beside it, row order = index id_map
ITEM_SCALE_FILE = "item_scale.json"  # {"baseline": mean random-pair cosine}


def random_pair_baseline(embeddings: np.ndarray, n_pairs: int = 200_000, seed: int = 0) -> float:
    """Mean cosine between random pairs of rows — the space's "unrelated" level."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(embeddings), n_pairs)
    b = rng.integers(0, len(embeddings), n_pairs)
    return float((embeddings[a] * embeddings[b]).sum(axis=1).mean())


class ItemSpace:
    def __init__(self, index: GameIndex, embeddings: np.ndarray, baseline: float = 0.0) -> None:
        self.index = index
        self._embs = embeddings
        self._row = {gid: i for i, gid in enumerate(index.game_ids)}
        self.baseline = baseline

    def rescale(self, cos: np.ndarray) -> np.ndarray:
        """Cosine → relevance on a 0-1 scale where `baseline` is 0."""
        return np.clip((cos - self.baseline) / (1.0 - self.baseline), 0.0, 1.0)

    @classmethod
    def load(cls, index_dir: Path) -> Optional["ItemSpace"]:
        """The item index if `05_build_index.py` produced one, else None."""
        index_dir = Path(index_dir)
        item_dir = index_dir / ITEM_INDEX_DIR
        embs_path = index_dir / ITEM_EMBS_FILE
        if not item_dir.exists() or not embs_path.exists():
            logger.warning("No item index at %s — game and author modes fall back "
                           "to profile queries", item_dir)
            return None
        index = GameIndex.load(item_dir)
        embs = np.load(embs_path)
        if len(embs) != len(index.game_ids):
            raise RuntimeError(f"{embs_path} has {len(embs)} rows but the index "
                               f"holds {len(index.game_ids)} ids")
        scale_path = index_dir / ITEM_SCALE_FILE
        baseline = json.loads(scale_path.read_text())["baseline"] if scale_path.exists() else 0.0
        logger.info("Item space: %d games, random-pair baseline %.3f", len(embs), baseline)
        return cls(index, embs, baseline)

    def __contains__(self, gameid: str) -> bool:
        return gameid in self._row

    def embedding(self, gameid: str) -> Optional[np.ndarray]:
        i = self._row.get(gameid)
        return None if i is None else self._embs[i]

    def centroid(self, game_ids: Iterable[str]) -> Optional[np.ndarray]:
        """Normalised mean of the seeds' embeddings; None if none are known."""
        embs = [self._embs[self._row[g]] for g in game_ids if g in self._row]
        if not embs:
            return None
        avg = np.mean(embs, axis=0).astype(np.float32)
        norm = np.linalg.norm(avg)
        return avg / norm if norm > 0 else avg

    def similarities(self, game_ids: Iterable[str], merge: str = "centroid") -> Dict[str, float]:
        """
        Relevance (rescaled cosine, see module docstring) of every game in the
        index to the seed set.

        `centroid` queries once with the seeds' mean — the shape of one game
        blended from several. `max` scores each candidate by its nearest seed,
        which keeps an author whose games span two styles reachable from both
        rather than from the average of neither.
        """
        seeds = [g for g in game_ids if g in self._row]
        if not seeds:
            return {}
        if merge == "max":
            rows = self._embs[[self._row[g] for g in seeds]]
            sims = (rows @ self._embs.T).max(axis=0)
        elif merge == "centroid":
            sims = self._embs @ self.centroid(seeds)
        else:
            raise ValueError(f"Unknown merge: {merge!r}")
        return dict(zip(self.index.game_ids, self.rescale(sims).astype(float)))

    def rank(
        self,
        game_ids: Iterable[str],
        exclude: Optional[set] = None,
        min_score: float = 0.0,
        merge: str = "centroid",
        top_n: Optional[int] = None,
        bayesian_avg_map: Optional[Dict[str, float]] = None,
        rating_weight: float = 0.5,
        max_rating: float = 5.0,
        allowed: Optional[set] = None,
    ) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
        """
        Rank the corpus against a seed set, in the shape `Reranker.rerank` uses.

        Seeds are always excluded — a game is not a recommendation for itself.
        `allowed`, when given, is the set of ids that may appear (the app uses
        it to keep "not a game" entries out). `min_score` is a floor on the
        rescaled relevance. The blend selects, relevance orders: see NOTES.md,
        "The blend selects; relevance orders".
        """
        sims = self.similarities(game_ids, merge=merge)
        if not sims:
            return [], {}
        drop = set(exclude or ()) | set(game_ids)
        relevance: Dict[str, float] = {}
        scored: List[Tuple[str, float]] = []
        for gid, cos in sims.items():
            if cos < min_score or gid in drop or (allowed is not None and gid not in allowed):
                continue
            score = cos
            if bayesian_avg_map is not None:
                rating = bayesian_avg_map.get(gid, 0.0) / max_rating
                score = (1 - rating_weight) * cos + rating_weight * rating
            relevance[gid] = cos
            scored.append((gid, score))
        scored.sort(key=lambda gs: -gs[1])
        if top_n is not None:
            scored = scored[:top_n]
            relevance = {g: relevance[g] for g, _ in scored}
        return scored, relevance
