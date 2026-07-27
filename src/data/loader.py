"""Load IFDB tables from MySQL and cache them as Parquet files."""

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

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


def _ordering_columns(connector: IFDBConnector, table: str) -> list[str]:
    """
    Columns to sort by so that a re-extraction produces the same row order.

    MySQL makes no promise about the order of an unsorted SELECT, so we sort by
    the primary key.  `playedgames` has none; there we sort by every column,
    which is what determinism requires anyway once rows can tie.
    """
    keys = connector.read_query(f"SHOW KEYS FROM `{table}` WHERE Key_name = 'PRIMARY'")
    if len(keys):
        return keys.sort_values("Seq_in_index")["Column_name"].tolist()
    return connector.read_query(f"SHOW COLUMNS FROM `{table}`")["Field"].tolist()


def _write_parquet(df: pd.DataFrame, dest: Path) -> None:
    """Write `df` without the pandas/pyarrow version stamp in the metadata."""
    arrow = pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(None)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(arrow, dest)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
        order = ", ".join(f"`{c}`" for c in _ordering_columns(connector, table))
        df = connector.read_query(f"SELECT * FROM `{table}` ORDER BY {order}")
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
    _write_parquet(df, dest)
    logger.info("  Saved %s  (%d rows)", dest.name, len(df))


def extract_all(
    connector: IFDBConnector,
    data_dir: Path,
    overwrite: bool = False,
) -> None:
    data_dir = Path(data_dir)
    for table, filename in TABLES.items():
        extract_table(connector, table, data_dir / filename, overwrite=overwrite)


MANIFEST = "manifest.json"


def write_manifest(data_dir: Path, source: Path | None = None) -> dict:
    """
    Record what is in the Parquet cache and what it was built from.

    Deliberately free of timestamps: two manifests compare equal exactly when
    the datasets do, which turns "did the data change?" into one diff.
    """
    data_dir = Path(data_dir)
    manifest: dict = {"source": None, "tables": {}}
    if source is not None and Path(source).exists():
        manifest["source"] = {"path": str(source), "sha256": sha256(Path(source))}

    for table, filename in TABLES.items():
        path = data_dir / filename
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        manifest["tables"][table] = {
            "file": filename,
            "rows": len(frame),
            "columns": list(frame.columns),
            "sha256": sha256(path),
        }

    (data_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote %s", data_dir / MANIFEST)
    return manifest


def load_parquet(data_dir: Path, filename: str) -> pd.DataFrame:
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/01_extract.py first."
        )
    return pd.read_parquet(path)
