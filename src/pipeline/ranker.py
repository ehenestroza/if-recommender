"""Stage 2: cross-encoder reranking and offline evaluation metrics."""

import logging
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker:
    """Score (query, document) pairs with a cross-encoder and return top-K."""

    def __init__(self, model_name: str) -> None:
        logger.info("Loading cross-encoder: %s", model_name)
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query_text: str,
        candidates: List[Tuple[str, float]],
        game_doc_lookup: Dict[str, str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Re-score candidates with the cross-encoder and return the top-K.

        candidates: list of (gameid, retrieval_score) from the bi-encoder stage
        game_doc_lookup: gameid → document text
        """
        if not candidates:
            return []

        game_ids = [gid for gid, _ in candidates]
        pairs = [(query_text, game_doc_lookup.get(gid, "")) for gid in game_ids]

        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(game_ids, scores.tolist()), key=lambda x: -x[1])
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Offline evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, set],
    ks: Tuple[int, ...] = (1, 5, 10, 20, 50),
) -> Dict[str, float]:
    """
    Compute Recall@K, NDCG@K, and MRR against ground-truth positives.

    predictions:   {userid → ordered list of predicted gameids}
    ground_truth:  {userid → set of relevant gameids}
    """
    max_k = max(ks)

    recall_sums: Dict[int, float] = {k: 0.0 for k in ks}
    ndcg_sums:   Dict[int, float] = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    n_users = 0

    for uid, relevant in ground_truth.items():
        if uid not in predictions or not relevant:
            continue

        pred = predictions[uid][:max_k]
        n_users += 1

        # MRR
        for rank, gid in enumerate(pred, start=1):
            if gid in relevant:
                mrr_sum += 1.0 / rank
                break

        for k in ks:
            pred_k = pred[:k]
            hits = sum(1 for gid in pred_k if gid in relevant)

            # Recall@K
            recall_sums[k] += hits / len(relevant)

            # NDCG@K
            dcg = sum(
                1.0 / np.log2(i + 2)
                for i, gid in enumerate(pred_k)
                if gid in relevant
            )
            ideal_len = min(len(relevant), k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))
            ndcg_sums[k] += dcg / idcg if idcg > 0 else 0.0

    if n_users == 0:
        logger.warning("No users with predictions and ground truth found")
        return {}

    metrics: Dict[str, float] = {"MRR": mrr_sum / n_users}
    for k in ks:
        metrics[f"Recall@{k}"] = recall_sums[k] / n_users
        metrics[f"NDCG@{k}"]   = ndcg_sums[k]   / n_users

    return metrics
