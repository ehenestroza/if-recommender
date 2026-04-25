"""PyTorch dataset classes for the two-tower training pipeline."""

import logging
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PairDataset(Dataset):
    """
    Dataset of (query_text, positive_doc_text) pairs for MultipleNegativesRankingLoss.

    Only positive interactions (label == 1) are used; in-batch negatives are
    derived implicitly during training.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        user_profiles: pd.DataFrame,
        game_docs: pd.DataFrame,
    ) -> None:
        profile_map: dict[str, str] = dict(
            zip(user_profiles["userid"], user_profiles["profile_text"])
        )
        doc_map: dict[str, str] = dict(
            zip(game_docs["gameid"], game_docs["doc_text"])
        )

        pos = interactions[interactions["label"] == 1]
        self.pairs: List[Tuple[str, str]] = []

        for _, row in pos.iterrows():
            uid, gid = row["userid"], row["gameid"]
            if uid in profile_map and gid in doc_map:
                self.pairs.append((profile_map[uid], doc_map[gid]))

        logger.info("PairDataset: %d positive pairs", len(self.pairs))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.pairs[idx]


class TripletDataset(Dataset):
    """
    Dataset of (anchor, positive, negative) text triples for TripletLoss.

    Explicit negatives are drawn from interactions with label == 0.
    If a user has no explicit negatives, the sample is skipped.
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        user_profiles: pd.DataFrame,
        game_docs: pd.DataFrame,
    ) -> None:
        profile_map: dict[str, str] = dict(
            zip(user_profiles["userid"], user_profiles["profile_text"])
        )
        doc_map: dict[str, str] = dict(
            zip(game_docs["gameid"], game_docs["doc_text"])
        )

        pos = (
            interactions[interactions["label"] == 1]
            .groupby("userid")["gameid"]
            .apply(list)
            .to_dict()
        )
        neg = (
            interactions[interactions["label"] == 0]
            .groupby("userid")["gameid"]
            .apply(list)
            .to_dict()
        )

        self.triplets: List[Tuple[str, str, str]] = []
        for uid, pos_games in pos.items():
            if uid not in profile_map or uid not in neg:
                continue
            anchor = profile_map[uid]
            neg_games = neg[uid]
            for pos_gid in pos_games:
                if pos_gid not in doc_map:
                    continue
                # pair with the first available negative (simple strategy)
                for neg_gid in neg_games:
                    if neg_gid in doc_map:
                        self.triplets.append(
                            (anchor, doc_map[pos_gid], doc_map[neg_gid])
                        )
                        break

        logger.info("TripletDataset: %d triplets", len(self.triplets))

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Tuple[str, str, str]:
        return self.triplets[idx]
