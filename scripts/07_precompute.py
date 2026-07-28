#!/usr/bin/env python
"""
Step 7 – Precompute ranked results for the enumerable query modes.

`userid`, `game_id`, and `author_id` queries draw from fixed, known sets: every
user with a profile, every game in the retrieval set, and every author with at
least two games. Their rankings only change when the
data or the models change, so they can be computed once offline and served as a
lookup — which keeps the interactive demo off the CPU entirely for those modes.
Only menu-built `text` queries then need live cross-encoder work.

Rankings are stored exactly as the live path produces them: the full candidate
pool scored by the reranker, truncated to `top_n` so filters still have depth to
narrow into.

Usage
-----
    python scripts/07_precompute.py [--mode userid|game_id|author_id|all] [--top-n 500]
                                    [--limit N] [--out DIR]

Options
-------
--mode      Which query modes to precompute (default: all)
--top-n     Ranked entries stored per key (default: 500)
--limit     Only process the first N keys — for smoke tests
--out       Output directory (default: config paths.data_dir)

IMPORTANT: the output is tied to the reranker that produced it. Re-run this after
retraining the reranker or rebuilding the index, or the demo will serve stale
rankings.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.env import configure_logging
configure_logging()

from src.data.preprocessor import (  # noqa: E402
    author_game_map, build_author_profiles, parse_profile_text,
)
from src.pipeline.retriever import filter_by_tag_overlap  # noqa: E402

logger = logging.getLogger(__name__)

USERID_FILE = "precomputed_userid.parquet"
GAMEID_FILE = "precomputed_gameid.parquet"
AUTHORID_FILE = "precomputed_authorid.parquet"


def _load_pipeline_module():
    """Import 06_run_pipeline.py by path (its name is not a valid identifier)."""
    import importlib.util

    path = Path(__file__).resolve().parent / "06_run_pipeline.py"
    spec = importlib.util.spec_from_file_location("pipeline_mod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rank_one(emb, exclude, retriever, reranker, doc_map, cfg_r,
              bayesian_avg_map, query_text, top_n, game_info_map=None):
    """Retrieve, score the whole pool, and return the top_n (gameid, score, relevance)."""
    candidates = retriever.index.search(emb, min_score=cfg_r.get("min_retrieval_score", 0.25))
    if exclude:
        candidates = [(g, s) for g, s in candidates if g not in exclude]

    # Every stored result shares at least one tag with the query, so a user can
    # always see why a recommendation was made. This also yields more usable
    # depth than a relevance floor alone — see README, "Precompute the
    # enumerable modes".
    if cfg_r.get("prefilter_by_tag", True) and game_info_map is not None:
        _, query_tags = parse_profile_text(query_text)
        if query_tags:
            candidates = filter_by_tag_overlap(candidates, game_info_map, set(query_tags))

    if not candidates:
        return []
    scored, relevance = reranker.rerank(
        query_text=query_text,
        candidates=candidates,
        game_doc_lookup=doc_map,
        top_k=top_n,
        bayesian_avg_map=bayesian_avg_map,
        rating_weight=cfg_r.get("rating_weight", 0.5),
        min_ce_score=cfg_r.get("min_rerank_score", 0.25),
    )
    return [(g, float(s), float(relevance.get(g, 0.0))) for g, s in scored]


class _ChunkWriter:
    """
    Stream rows to Parquet in batches instead of holding the whole job in memory.

    The game_id pass produces a few million rows; accumulating them in a list and
    converting to a DataFrame at the end peaks at roughly twice that, which is a
    lot of resident memory for a job that runs for hours. Writing every
    `chunk_rows` keeps the footprint flat.

    The schema is declared explicitly rather than inferred per chunk — otherwise a
    batch that happened to contain, say, only integral scores could infer a
    different type and the writer would reject it mid-run.
    """

    def __init__(self, path: Path, key_name: str, chunk_rows: int = 200_000) -> None:
        self.path = path
        # Write beside the destination and rename on success. Parquet's footer is
        # only written on close, so a file being streamed to is unreadable —
        # publishing it only when complete keeps readers from seeing a corrupt
        # file, and keeps a failed run from destroying the previous artefact.
        self.temp_path = path.with_suffix(path.suffix + ".partial")
        self.key_name = key_name
        self.chunk_rows = chunk_rows
        self.schema = pa.schema([
            (key_name, pa.string()),
            ("gameid", pa.string()),
            ("score", pa.float64()),
            ("relevance", pa.float64()),
            ("rank", pa.int32()),
        ])
        self._buffer: list = []
        self._writer = None
        self.total_rows = 0
        self.total_keys = 0

    def add(self, rows: list) -> None:
        if not rows:
            return
        self.total_keys += 1
        self._buffer.extend(rows)
        if len(self._buffer) >= self.chunk_rows:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        frame = pd.DataFrame(
            self._buffer,
            columns=[self.key_name, "gameid", "score", "relevance", "rank"],
        )
        table = pa.Table.from_pandas(frame, schema=self.schema, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.temp_path, self.schema)
        self._writer.write_table(table)
        self.total_rows += len(frame)
        self._buffer = []

    def close(self) -> None:
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self.temp_path.replace(self.path)   # atomic publish
        size_mb = self.path.stat().st_size / (1024 * 1024) if self.path.exists() else 0.0
        logger.info("Wrote %s — %d rows, %d keys, %.1f MB",
                    self.path.name, self.total_rows, self.total_keys, size_mb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute lookup rankings")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", default="all",
                        choices=["userid", "game_id", "author_id", "all"])
    parser.add_argument("--top-n", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 06_run_pipeline.py starts with a digit, so it cannot be imported by name.
    # Load it by path to reuse load_artefacts() rather than duplicating it —
    # the precompute must load exactly what the live path loads.
    mod = _load_pipeline_module()

    # Trailing values are any previously precomputed tables, which this script
    # is in the business of replacing — deliberately ignored.
    (retriever, reranker, query_encoder,
     game_docs, doc_map, profile_map, name_map,
     bayesian_avg_map, reviews_df, playedgames_df,
     game_query_text_map, game_info_map, *_) = mod.load_artefacts(cfg)

    out_dir = Path(args.out) if args.out else Path(cfg["paths"]["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_r = cfg["retrieval"]
    top_n = args.top_n

    if args.mode in ("userid", "all"):
        users = list(profile_map)
        if args.limit:
            users = users[: args.limit]
        logger.info("Precomputing %d userid rankings (top %d each) …", len(users), top_n)
        # Group the seen-games lookup once. Filtering the 80k-row reviews frame
        # per user would dominate the run over thousands of users.
        seen_by_user: dict = {}
        for frame in (reviews_df, playedgames_df):
            if frame is None or "userid" not in frame.columns:
                continue
            for uid, games in frame.groupby("userid")["gameid"]:
                seen_by_user.setdefault(uid, set()).update(games)

        writer, t0 = _ChunkWriter(out_dir / USERID_FILE, "userid"), time.time()
        for i, uid in enumerate(users, 1):
            emb = retriever._encode_userid(uid)
            if emb is None:
                continue
            # Same suppression the live path applies: reviewed + played games.
            seen = seen_by_user.get(uid, set())
            ranked = _rank_one(emb, seen, retriever, reranker, doc_map, cfg_r,
                               bayesian_avg_map, profile_map.get(uid, ""), top_n,
                               game_info_map)
            writer.add([(uid, g, s, r, n) for n, (g, s, r) in enumerate(ranked, 1)])
            if i % 200 == 0:
                rate = i / (time.time() - t0)
                logger.info("  %d/%d users (%.1f/s, ETA %.0f min)",
                            i, len(users), rate, (len(users) - i) / rate / 60)
        writer.close()

    if args.mode in ("game_id", "all"):
        games = list(game_query_text_map)
        if args.limit:
            games = games[: args.limit]
        logger.info("Precomputing %d game_id rankings (top %d each) …", len(games), top_n)
        # Key column is named separately from the ranked `gameid` it points to.
        writer, t0 = _ChunkWriter(out_dir / GAMEID_FILE, "seed_gameid"), time.time()
        for i, gid in enumerate(games, 1):
            emb = retriever._encode_game_ids([gid])
            if emb is None:
                continue
            query_text = game_query_text_map.get(gid, doc_map.get(gid, ""))
            ranked = _rank_one(emb, {gid}, retriever, reranker, doc_map, cfg_r,
                               bayesian_avg_map, query_text, top_n,
                               game_info_map)
            writer.add([(gid, g, s, r, n) for n, (g, s, r) in enumerate(ranked, 1)])
            if i % 200 == 0:
                rate = i / (time.time() - t0)
                logger.info("  %d/%d games (%.1f/s, ETA %.0f min)",
                            i, len(games), rate, (len(games) - i) / rate / 60)
        writer.close()

    if args.mode in ("author_id", "all"):
        profiles = build_author_profiles(game_docs)
        if args.limit:
            profiles = profiles.head(args.limit)
        by_author = author_game_map(game_docs)
        logger.info("Precomputing %d author_id rankings (top %d each) …", len(profiles), top_n)
        writer, t0 = _ChunkWriter(out_dir / AUTHORID_FILE, "authorid"), time.time()
        for i, row in enumerate(profiles.itertuples(index=False), 1):
            emb = retriever._encode_text(row.profile_text)
            # Suppress the author's own catalogue, as game_id suppresses its seed.
            ranked = _rank_one(emb, set(by_author.get(row.authorid, [])), retriever,
                               reranker, doc_map, cfg_r, bayesian_avg_map,
                               row.profile_text, top_n, game_info_map)
            writer.add([(row.authorid, g, s, r, n) for n, (g, s, r) in enumerate(ranked, 1)])
            if i % 200 == 0:
                rate = i / (time.time() - t0)
                logger.info("  %d/%d authors (%.1f/s, ETA %.0f min)",
                            i, len(profiles), rate, (len(profiles) - i) / rate / 60)
        writer.close()

    logger.info("✓ Precompute complete.")


if __name__ == "__main__":
    main()
