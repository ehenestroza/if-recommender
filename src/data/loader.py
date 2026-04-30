"""Load IFDB tables from MySQL and cache them as Parquet files."""

import logging
from pathlib import Path

import pandas as pd

from src.db.connector import IFDBConnector

logger = logging.getLogger(__name__)

# MySQL table name → parquet filename
TABLES: dict[str, str] = {
    "games":       "games.parquet",
    "reviews":     "reviews.parquet",
    "users":       "users.parquet",
    "playedgames": "playedgames.parquet",
}

# IFDB uses "ifid" as the game primary key in some dumps; normalise to "gameid".
_GAME_ID_ALIASES = ("ifid", "id")


def _normalise_game_id(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the game-ID column to 'gameid' if it isn't already."""
    if "gameid" in df.columns:
        return df
    for alias in _GAME_ID_ALIASES:
        if alias in df.columns:
            df = df.rename(columns={alias: "gameid"})
            logger.debug("Renamed column '%s' → 'gameid'", alias)
            return df
    return df


def _normalise_user_id(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the user primary-key column to 'userid' if it isn't already."""
    if "userid" in df.columns:
        return df
    if "id" in df.columns:
        df = df.rename(columns={"id": "userid"})
        logger.debug("Renamed column 'id' → 'userid'")
    return df


def _coerce_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast mixed-type object columns to string so PyArrow can write them.

    MySQL can return columns with heterogeneous Python types (e.g. int and str
    in the same 'published' column).  PyArrow rejects these at schema-inference
    time, so we normalise every object-dtype column to nullable strings.
    """
    for col in df.select_dtypes("object").columns:
        # Keep genuine nulls as None; stringify everything else.
        df[col] = df[col].where(df[col].isna(), df[col].astype(str))
    return df


def extract_table(
    connector: IFDBConnector,
    table: str,
    dest: Path,
    overwrite: bool = False,
) -> None:
    if dest.exists() and not overwrite:
        logger.info("  [skip] %s already exists", dest.name)
        return

    logger.info("  Fetching table '%s' …", table)
    try:
        df = connector.read_table(table)
    except Exception as exc:
        logger.warning("  Could not fetch '%s': %s", table, exc)
        return

    if table == "users":
        df = _normalise_user_id(df)
    elif table == "playedgames":
        df = _normalise_game_id(df)
        df = _normalise_user_id(df)
    else:
        df = _normalise_game_id(df)
    df = _coerce_object_columns(df)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    logger.info("  Saved %s  (%d rows)", dest.name, len(df))


def extract_all(
    connector: IFDBConnector,
    data_dir: Path,
    overwrite: bool = False,
) -> None:
    data_dir = Path(data_dir)
    for table, filename in TABLES.items():
        extract_table(connector, table, data_dir / filename, overwrite=overwrite)


def load_parquet(data_dir: Path, filename: str) -> pd.DataFrame:
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/01_extract.py first."
        )
    return pd.read_parquet(path)
