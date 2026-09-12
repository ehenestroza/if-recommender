#!/usr/bin/env python
"""
Evaluate the `game` and `author` search modes against held-out ratings.

Neither mode had ground truth before: a game's "similar games" were whatever
the profile encoder returned for its tags, and nothing checked them. The
protocol here uses the same test split as the user evaluation, read the other
way round:

  game    for each user with test positives, one of their *training* positives
          is the seed game; the test positives are what a good "more like this"
          list should contain.
  author  the seed is the author of one of the user's training positives; the
          author's own games are excluded (as the mode does), and the targets
          are the user's test positives by anyone else.

Seeds are drawn with a fixed generator, so every method scores the same
queries. Methods:

  profile        the shipped approach — the seed's systems and tags formatted
                 as a profile, through the query encoder, against the document
                 index (for authors: the author profile)
  profile+rerank the same, then the cross-encoder, as the app served it —
                 slow, so `--rerank-sample N` limits it to N queries
  item           the item encoder: cosine to the seed (game), or to the
                 centroid / max over the author's games (author)

Usage
-----
    python scripts/eval_item_modes.py [--modes game,author] [--rerank-sample 300]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.env import configure_logging
configure_logging()

from src.data.preprocessor import (  # noqa: E402
    author_game_map, build_author_profiles, drop_non_games, parse_profile_text,
)
from src.pipeline.items import ItemSpace  # noqa: E402
from src.pipeline.ranker import Reranker, evaluate_retrieval  # noqa: E402
from src.pipeline.retriever import filter_by_tag_overlap  # noqa: E402

logger = logging.getLogger(__name__)

KS = (5, 10, 25, 50)
NON_AUTHORS = {"anonymous", "unknown", "various", "n/a", "none"}


def _load_doc_space(index_dir: Path) -> Tuple[np.ndarray, List[str]]:
    import pickle
    embs = np.load(index_dir / "game_embs.npy")
    with open(index_dir / "gameid_to_idx.pkl", "rb") as f:
        gameid_to_idx = pickle.load(f)
    ids = [None] * len(gameid_to_idx)
    for g, i in gameid_to_idx.items():
        ids[i] = g
    return embs, ids


def _rank_from_sims(sims: np.ndarray, ids: List[str], exclude: set, allowed: set,
                    top_k: int) -> List[str]:
    order = np.argsort(-sims)
    out = []
    for i in order:
        g = ids[i]
        if g in exclude or g not in allowed:
            continue
        out.append(g)
        if len(out) >= top_k:
            break
    return out


def build_queries(interactions: pd.DataFrame, game_docs: pd.DataFrame, allowed: set,
                  seed: int = 0):
    """(game queries, author queries): per user, one seed and the target set."""
    rng = np.random.default_rng(seed)
    train_pos = interactions[(interactions["split"] == "train") & (interactions["label"] == 1)]
    test_pos = interactions[(interactions["split"] == "test") & (interactions["label"] == 1)]
    train_by_user = train_pos.groupby("userid")["gameid"].apply(
        lambda s: sorted(g for g in s if g in allowed)).to_dict()
    authors_of = {g: [a.strip().lower() for a in str(v).split(",") if a.strip()]
                  for g, v in zip(game_docs["gameid"], game_docs["author_clean"].fillna(""))}
    by_author = author_game_map(game_docs)

    game_q: Dict[str, Tuple[str, set]] = {}     # uid → (seed gameid, targets)
    author_q: Dict[str, Tuple[str, set]] = {}   # uid → (authorid, targets)
    for uid, grp in test_pos.groupby("userid"):
        pool = train_by_user.get(uid)
        targets = {g for g in grp["gameid"] if g in allowed}
        if not pool or not targets:
            continue
        seed_gid = pool[rng.integers(len(pool))]
        game_q[uid] = (seed_gid, targets)

        candidates = sorted({a for g in pool for a in authors_of.get(g, [])
                             if a not in NON_AUTHORS and a in by_author})
        if not candidates:
            continue
        author = candidates[rng.integers(len(candidates))]
        own = set(by_author[author])
        author_targets = targets - own
        if author_targets:
            author_q[uid] = (author, author_targets)
    return game_q, author_q, by_author


def report(title: str, results: Dict[str, Dict[str, float]]) -> None:
    methods = list(results)
    print(f"\n{title}")
    print("  " + f"{'metric':<12}" + "".join(f"{m:>16}" for m in methods))
    metrics = ["MRR"] + [f"{name}@{k}" for k in KS for name in ("Recall", "NDCG")]
    for metric in metrics:
        print("  " + f"{metric:<12}" + "".join(f"{results[m].get(metric, float('nan')):>16.4f}" for m in methods))
    print("  " + f"{'queries':<12}" + "".join(f"{results[m].get('n', 0):>16d}" for m in methods))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--modes", default="game,author")
    parser.add_argument("--rerank-sample", type=int, default=0,
                        help="Also score the profile+rerank baseline on N queries (0 = skip)")
    parser.add_argument("--top-k", type=int, default=max(KS))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_dir, model_dir, index_dir = (Path(cfg["paths"][k]) for k in ("data_dir", "model_dir", "index_dir"))
    retr = cfg["retrieval"]

    game_docs = pd.read_parquet(data_dir / "game_docs_retrieval.parquet")
    game_docs, _ = drop_non_games(game_docs)
    allowed = set(game_docs["gameid"])
    query_text = dict(zip(game_docs["gameid"], game_docs["query_text"]))
    doc_map = dict(zip(game_docs["gameid"], game_docs["doc_text"]))
    info = game_docs.set_index("gameid")[["tags_clean"]].to_dict("index")
    reviewed = game_docs[game_docs["review_count"] > 0]
    bayes = dict(zip(reviewed["gameid"], reviewed["bayesian_avg"]))
    interactions = pd.read_parquet(data_dir / "interactions.parquet")

    game_q, author_q, by_author = build_queries(interactions, game_docs, allowed)
    logger.info("Queries: %d game, %d author", len(game_q), len(author_q))

    doc_embs, doc_ids = _load_doc_space(index_dir)
    query_encoder = SentenceTransformer(str(model_dir / "query_encoder"))
    query_encoder.max_seq_length = cfg["model"]["max_seq_length"]
    items = ItemSpace.load(index_dir)
    reranker = None
    if args.rerank_sample:
        reranker = Reranker(str(model_dir / "reranker"))

    def profile_method(text_by_key: Dict[str, str], exclude_by_key: Dict[str, set],
                       rerank_keys=None) -> Dict[str, List[str]]:
        keys = list(text_by_key)
        q = query_encoder.encode([text_by_key[k] for k in keys], batch_size=128,
                                 normalize_embeddings=True, show_progress_bar=False)
        preds = {}
        for n, key in enumerate(keys):
            sims = q[n] @ doc_embs.T
            if rerank_keys is None:
                preds[key] = _rank_from_sims(sims, doc_ids, exclude_by_key[key], allowed, args.top_k)
                continue
            if key not in rerank_keys:
                continue
            # The served pipeline: cosine floor, tag pre-filter, cross-encoder, blend.
            cands = [(doc_ids[i], float(sims[i])) for i in np.where(sims >= retr["min_retrieval_score"])[0]
                     if doc_ids[i] not in exclude_by_key[key] and doc_ids[i] in allowed]
            _, qtags = parse_profile_text(text_by_key[key])
            if retr.get("prefilter_by_tag", True) and qtags:
                cands = filter_by_tag_overlap(cands, info, set(qtags))
            scored, rel = reranker.rerank(
                query_text=text_by_key[key], candidates=cands, game_doc_lookup=doc_map,
                top_k=len(cands), bayesian_avg_map=bayes,
                rating_weight=retr.get("rating_weight", 0.5),
                min_ce_score=retr.get("min_rerank_score", 0.1))
            # Ordered by relevance, as the page is.
            preds[key] = [g for g, _ in sorted(scored, key=lambda gs: -rel.get(gs[0], 0))][:args.top_k]
            if (len(preds) % 50) == 0:
                logger.info("  reranked %d/%d", len(preds), len(rerank_keys))
        return preds

    def item_method(seeds_by_key: Dict[str, List[str]], exclude_by_key: Dict[str, set],
                    merge: str) -> Dict[str, List[str]]:
        preds = {}
        for key, seeds in seeds_by_key.items():
            scored, rel = items.rank(seeds, exclude=exclude_by_key[key], merge=merge,
                                     bayesian_avg_map=bayes, rating_weight=retr.get("rating_weight", 0.5),
                                     allowed=allowed)
            preds[key] = [g for g, _ in sorted(scored, key=lambda gs: -rel[gs[0]])][:args.top_k]
        return preds

    rng = np.random.default_rng(1)

    if "game" in args.modes and game_q:
        truth = {u: t for u, (_, t) in game_q.items()}
        text = {u: query_text[s] for u, (s, _) in game_q.items()}
        excl = {u: {s} for u, (s, _) in game_q.items()}
        results = {}
        results["profile"] = evaluate_retrieval(profile_method(text, excl), truth, KS)
        results["profile"]["n"] = len(truth)
        if reranker is not None:
            sample = set(rng.choice(list(game_q), size=min(args.rerank_sample, len(game_q)), replace=False))
            preds = profile_method(text, excl, rerank_keys=sample)
            results["profile+rerank"] = evaluate_retrieval(preds, {u: truth[u] for u in sample}, KS)
            results["profile+rerank"]["n"] = len(sample)
            results["profile (same)"] = evaluate_retrieval(
                profile_method({u: text[u] for u in sample}, excl), {u: truth[u] for u in sample}, KS)
            results["profile (same)"]["n"] = len(sample)
        if items is not None:
            results["item"] = evaluate_retrieval(
                item_method({u: [s] for u, (s, _) in game_q.items()}, excl, "centroid"), truth, KS)
            results["item"]["n"] = len(truth)
            if reranker is not None:
                results["item (same)"] = evaluate_retrieval(
                    item_method({u: [game_q[u][0]] for u in sample}, excl, "centroid"),
                    {u: truth[u] for u in sample}, KS)
                results["item (same)"]["n"] = len(sample)
        report("game mode — seed: one training positive; targets: the user's test positives", results)

    if "author" in args.modes and author_q:
        profiles = build_author_profiles(game_docs)
        author_text = dict(zip(profiles["authorid"], profiles["profile_text"]))
        truth = {u: t for u, (_, t) in author_q.items()}
        text = {u: author_text.get(a, "") for u, (a, _) in author_q.items()}
        excl = {u: set(by_author[a]) for u, (a, _) in author_q.items()}
        results = {}
        results["profile"] = evaluate_retrieval(profile_method(text, excl), truth, KS)
        results["profile"]["n"] = len(truth)
        if reranker is not None:
            sample = set(rng.choice(list(author_q), size=min(args.rerank_sample, len(author_q)), replace=False))
            preds = profile_method(text, excl, rerank_keys=sample)
            results["profile+rerank"] = evaluate_retrieval(preds, {u: truth[u] for u in sample}, KS)
            results["profile+rerank"]["n"] = len(sample)
            results["profile (same)"] = evaluate_retrieval(
                profile_method({u: text[u] for u in sample}, excl), {u: truth[u] for u in sample}, KS)
            results["profile (same)"]["n"] = len(sample)
        if items is not None:
            seeds = {u: list(by_author[a]) for u, (a, _) in author_q.items()}
            for merge in ("centroid", "max"):
                results[f"item {merge}"] = evaluate_retrieval(item_method(seeds, excl, merge), truth, KS)
                results[f"item {merge}"]["n"] = len(truth)
                if reranker is not None:
                    results[f"{merge} (same)"] = evaluate_retrieval(
                        item_method({u: seeds[u] for u in sample}, excl, merge),
                        {u: truth[u] for u in sample}, KS)
                    results[f"{merge} (same)"]["n"] = len(sample)
        report("author mode — seed: the author of a training positive; targets: test positives by others", results)


if __name__ == "__main__":
    main()
