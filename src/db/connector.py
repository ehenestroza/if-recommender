"""SQLAlchemy connection to the IFDB MySQL database."""

import logging
import os

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text

logger = logging.getLogger(__name__)


class IFDBConnector:
    def __init__(self, host: str, port: int, user: str, password: str, database: str) -> None:
        url = sa.engine.URL.create(
            "mysql+pymysql",
            username=user,
            password=password,
            host=host,
            port=port,
            database=database,
        )
        self._engine = sa.create_engine(url, pool_pre_ping=True)
        logger.info("Connecting to %s@%s:%s/%s", user, host, port, database)

    @classmethod
    def from_config(cls, cfg: dict) -> "IFDBConnector":
        db = cfg["database"]
        password = os.environ.get("IFDB_DB_PASSWORD", db.get("password", ""))
        return cls(
            host=db.get("host", "localhost"),
            port=int(db.get("port", 3306)),
            user=db.get("user", "root"),
            password=password,
            database=db.get("database", "ifarchive"),
        )

    def read_table(self, table: str) -> pd.DataFrame:
        with self._engine.connect() as conn:
            return pd.read_sql(text(f"SELECT * FROM `{table}`"), conn)

    def read_query(self, query: str) -> pd.DataFrame:
        with self._engine.connect() as conn:
            return pd.read_sql(text(query), conn)

    @property
    def engine(self) -> sa.engine.Engine:
        return self._engine
