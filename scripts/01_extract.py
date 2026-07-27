#!/usr/bin/env python
"""
Step 1 – Extract raw IFDB tables → Parquet cache.

By default this reads the IFDB SQL dump (data/ifdb-archive.sql.gz) by loading it
into a disposable MariaDB container, which is removed once extraction finishes.
Pass --source mysql to read from an already-running server instead.

Usage
-----
    python scripts/01_extract.py [--overwrite] [--source dump|mysql] [--keep-container]

Options
-------
--overwrite         Re-fetch tables that already exist in data/
--source            Where to read from (default: dump)
--keep-container    Leave the MariaDB container running so a re-run skips the
                    multi-minute import. Remove it with: docker rm -f ifdb-extract
"""

import argparse
import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.env import configure_logging
configure_logging()

from src.db.connector import IFDBConnector
from src.db.container import DumpConfig, mariadb_from_dump
from src.data.loader import extract_all, write_manifest, TABLES

logger = logging.getLogger(__name__)


@contextmanager
def open_connector(cfg: dict, source: str, keep_container: bool) -> Iterator[IFDBConnector]:
    """Yield a connector to either the live server or a container holding the dump."""
    if source == "mysql":
        yield IFDBConnector.from_config(cfg)
        return

    with mariadb_from_dump(DumpConfig.from_config(cfg, keep=keep_container)) as params:
        yield IFDBConnector(**params)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract IFDB tables to Parquet")
    parser.add_argument("--overwrite", action="store_true", help="Re-fetch cached tables")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--source",
        choices=("dump", "mysql"),
        default="dump",
        help="Read from the SQL dump via a temporary container (default) or a live server",
    )
    parser.add_argument(
        "--keep-container",
        action="store_true",
        help="Keep the temporary MariaDB container alive after extraction",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = Path(cfg["paths"]["data_dir"])

    logger.info("Tables to extract: %s", list(TABLES.keys()))
    try:
        with open_connector(cfg, args.source, args.keep_container) as connector:
            extract_all(connector, data_dir, overwrite=args.overwrite)
    except (RuntimeError, FileNotFoundError, TimeoutError) as exc:
        # Missing runtime / missing dump / slow startup: report the cause, not a traceback.
        logger.error("%s", exc)
        sys.exit(1)

    # The manifest doubles as the sanity check: row counts, columns, and digests
    # for everything in the cache, plus the dump they came from.
    source = DumpConfig.from_config(cfg).path if args.source == "dump" else None
    manifest = write_manifest(data_dir, source)
    for table, entry in manifest["tables"].items():
        logger.info("  %-14s %8d rows  %s", table, entry["rows"], entry["sha256"][:12])


if __name__ == "__main__":
    main()
