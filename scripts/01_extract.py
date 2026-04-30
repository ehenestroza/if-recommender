#!/usr/bin/env python
"""
Step 1 – Extract raw tables from MySQL → Parquet cache.

Usage
-----
    python scripts/01_extract.py [--overwrite]

Options
-------
--overwrite   Re-fetch tables that already exist in data/
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.env import configure_logging
configure_logging()

from src.db.connector import IFDBConnector
from src.data.loader import extract_all, TABLES

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract IFDB tables to Parquet")
    parser.add_argument("--overwrite", action="store_true", help="Re-fetch cached tables")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    connector = IFDBConnector.from_config(cfg)
    data_dir  = Path(cfg["paths"]["data_dir"])

    logger.info("Tables to extract: %s", list(TABLES.keys()))
    extract_all(connector, data_dir, overwrite=args.overwrite)

    # Quick sanity check
    logger.info("Sanity check row counts:")
    for table, filename in TABLES.items():
        path = data_dir / filename
        if path.exists():
            import pandas as pd
            n = len(pd.read_parquet(path))
            logger.info("  %-20s %d rows", table, n)


if __name__ == "__main__":
    main()
