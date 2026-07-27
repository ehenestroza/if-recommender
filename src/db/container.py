"""
Load a raw IFDB SQL dump into a disposable MariaDB container.

IFDB publishes its database as a mysqldump (``ifdb-archive.sql.gz``).  Instead of
requiring a permanently installed MySQL server, we start a throwaway MariaDB
container, stream the dump into it, extract the tables we need, and tear the
container down again.  The container is an implementation detail of
``scripts/01_extract.py`` — nothing downstream of the Parquet cache needs it.

Requires a container runtime (``docker`` or ``podman``) on PATH.
"""

import gzip
import logging
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator

import pymysql

logger = logging.getLogger(__name__)

# Container runtimes we know how to drive, in preference order.
RUNTIMES = ("docker", "podman")

# MySQL client binaries, in preference order.  MariaDB 11+ images ship only the
# `mariadb`-prefixed names; 10.x images ship both.
CLIENTS = ("mariadb", "mysql")

# Durability buys us nothing here — the server is destroyed after extraction —
# and turning it off makes the one-shot bulk import several times faster.
SERVER_ARGS = (
    "--innodb-flush-log-at-trx-commit=0",
    "--innodb-doublewrite=0",
    "--sync-binlog=0",
    "--skip-log-bin",
    "--character-set-server=utf8mb4",
    "--collation-server=utf8mb4_general_ci",
)

# How often to log import progress.
_PROGRESS_BYTES = 25 * 1024 * 1024

# Tables used to decide whether a reused container still holds a usable copy of
# the dump.  `wishlists` is written last, so a populated one means the previous
# import ran to completion instead of dying half-way.
_COMPLETION_TABLES = ("games", "wishlists")


@dataclass(frozen=True)
class DumpConfig:
    """Everything needed to stand up a MariaDB container around a dump file."""

    path: Path
    image: str = "mariadb:10.5.26"
    container_name: str = "ifdb-extract"
    database: str = "ifarchive"
    # Datadir size cap when running on tmpfs; "" keeps the data on disk instead.
    tmpfs_size: str = "3g"
    # Throwaway credentials: the server is published on 127.0.0.1 only and lives
    # just long enough to extract the tables.
    user: str = "root"
    password: str = "ifdb"
    startup_timeout_s: int = 180
    import_timeout_s: int = 3600
    keep: bool = False

    @classmethod
    def from_config(cls, cfg: dict, keep: bool = False) -> "DumpConfig":
        db = cfg.get("database", {})
        dump = db.get("dump", {})
        defaults = cls(path=Path("."))
        return cls(
            path=Path(dump.get("path", "data/ifdb-archive.sql.gz")),
            image=dump.get("image", defaults.image),
            container_name=dump.get("container_name", defaults.container_name),
            database=db.get("database", defaults.database),
            tmpfs_size=str(dump.get("tmpfs_size", defaults.tmpfs_size) or ""),
            startup_timeout_s=int(dump.get("startup_timeout_s", defaults.startup_timeout_s)),
            import_timeout_s=int(dump.get("import_timeout_s", defaults.import_timeout_s)),
            keep=keep,
        )


def _runtime() -> str:
    for exe in RUNTIMES:
        if shutil.which(exe):
            return exe
    raise RuntimeError(
        "No container runtime found. Install Docker Desktop (or podman) to load the "
        "IFDB dump, or point scripts/01_extract.py --source mysql at an existing server."
    )


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a runtime command, capturing output as text."""
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _container_state(runtime: str, name: str) -> str:
    """Return the container's state ('running', 'exited', …), or '' if absent."""
    proc = _run([runtime, "inspect", "--format", "{{.State.Status}}", name])
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _remove_container(runtime: str, name: str) -> None:
    if not _container_state(runtime, name):
        return
    logger.info("Removing container '%s'", name)
    _run([runtime, "rm", "--force", "--volumes", name])


def _start_container(runtime: str, cfg: DumpConfig) -> None:
    logger.info("Starting %s container '%s' (%s)", runtime, cfg.container_name, cfg.image)
    argv = [
        runtime, "run", "--detach",
        "--name", cfg.container_name,
        # An empty host port lets the runtime pick a free one, so we never
        # collide with a MySQL the user already runs on 3306.
        "--publish", "127.0.0.1::3306",
        "--env", f"MARIADB_ROOT_PASSWORD={cfg.password}",
        "--env", f"MARIADB_DATABASE={cfg.database}",
    ]
    if cfg.tmpfs_size:
        # Datadir in RAM: the import runs markedly faster and no state can
        # survive the container. The size is a cap, not a reservation.
        argv += ["--tmpfs", f"/var/lib/mysql:rw,size={cfg.tmpfs_size}"]
    proc = _run([*argv, cfg.image, *SERVER_ARGS])
    if proc.returncode != 0:
        raise RuntimeError(f"Could not start container: {proc.stderr.strip()}")


def _host_port(runtime: str, cfg: DumpConfig) -> int:
    """Return the host port the runtime picked for the container's 3306."""
    for attempt in range(3):  # the mapping can lag a fraction behind `run --detach`
        proc = _run([runtime, "port", cfg.container_name, "3306/tcp"])
        if proc.returncode == 0 and proc.stdout.strip():
            # One "addr:port" mapping per line (IPv4, and possibly IPv6).
            return int(proc.stdout.strip().splitlines()[0].rsplit(":", 1)[1])
        time.sleep(1)
    raise RuntimeError(f"Could not read the published port: {proc.stderr.strip()}")


def _exec(runtime: str, cfg: DumpConfig, argv: list[str]) -> subprocess.CompletedProcess:
    """Run a command inside the container, passing the password via the env."""
    return _run([
        runtime, "exec", "--env", f"MYSQL_PWD={cfg.password}", cfg.container_name, *argv
    ])


def _find_client(runtime: str, cfg: DumpConfig) -> str:
    for client in CLIENTS:
        if _exec(runtime, cfg, ["sh", "-c", f"command -v {client}"]).returncode == 0:
            return client
    raise RuntimeError(f"No MySQL client binary found in image {cfg.image}")


def _wait_ready(runtime: str, cfg: DumpConfig, params: dict) -> None:
    """
    Block until the server answers over TCP, or the startup timeout expires.

    The check deliberately goes through the published port rather than `exec`:
    during first-time initialisation the image runs a temporary server that is
    reachable on the in-container socket but has networking disabled, and
    importing into that server would be cut short when it is shut down.  A
    completed TCP handshake means the real server is up.
    """
    logger.info("Waiting for MariaDB to accept connections …")
    deadline = time.monotonic() + cfg.startup_timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        if _container_state(runtime, cfg.container_name) not in ("running", "created"):
            logs = _run([runtime, "logs", "--tail", "20", cfg.container_name])
            raise RuntimeError(f"Container exited during startup:\n{logs.stdout}{logs.stderr}")
        try:
            pymysql.connect(**params, connect_timeout=5).close()
            return
        except pymysql.Error as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"MariaDB not ready after {cfg.startup_timeout_s}s. Last error: {last_error}")


def _already_loaded(params: dict) -> bool:
    """True if a reused container still holds a complete copy of the dump."""
    with pymysql.connect(**params, connect_timeout=5) as conn, conn.cursor() as cur:
        for table in _COMPLETION_TABLES:
            try:
                cur.execute(f"SELECT 1 FROM `{table}` LIMIT 1")
            except pymysql.Error:
                return False  # table absent: nothing (or only part of the dump) loaded
            if cur.fetchone() is None:
                return False
    return True


def _stream(src: IO[bytes], dest: IO[bytes]) -> None:
    """Copy the dump into the client's stdin, logging progress as it goes."""
    copied = next_report = 0
    while chunk := src.read(1 << 20):
        dest.write(chunk)
        copied += len(chunk)
        if copied >= next_report:
            logger.info("  … %d MB sent", copied // (1024 * 1024))
            next_report = copied + _PROGRESS_BYTES
    logger.info("  %d MB sent, waiting for the server to finish …", copied // (1024 * 1024))


def _import_dump(runtime: str, cfg: DumpConfig, client: str) -> None:
    logger.info("Importing %s into '%s' (this takes a few minutes) …", cfg.path, cfg.database)
    opener = gzip.open if cfg.path.suffix == ".gz" else open
    cmd = [
        runtime, "exec", "--interactive",
        "--env", f"MYSQL_PWD={cfg.password}",
        cfg.container_name,
        client, f"--user={cfg.user}", "--default-character-set=utf8mb4", cfg.database,
    ]
    started = time.monotonic()

    # stderr goes to a file rather than a pipe: nothing drains a pipe while we
    # are busy writing 100+ MB to stdin, and a full pipe buffer would deadlock.
    with tempfile.TemporaryFile("w+") as errfile:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errfile)
        try:
            with opener(cfg.path, "rb") as fh:
                _stream(fh, proc.stdin)
        except BrokenPipeError:
            pass  # the client exited early; its stderr explains why
        except BaseException:
            proc.kill()
            raise
        finally:
            with suppress(BrokenPipeError, OSError):
                proc.stdin.close()

        try:
            proc.wait(timeout=cfg.import_timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError(
                f"Dump import exceeded database.dump.import_timeout_s ({cfg.import_timeout_s}s)"
            ) from None

        errfile.seek(0)
        stderr = errfile.read().strip()

    if proc.returncode != 0:
        raise RuntimeError(f"Dump import failed:\n{stderr}")
    if stderr:
        logger.debug("Import stderr: %s", stderr)
    logger.info("Import finished in %.1f min", (time.monotonic() - started) / 60)


@contextmanager
def mariadb_from_dump(cfg: DumpConfig) -> Iterator[dict]:
    """
    Serve `cfg.path` from a temporary MariaDB container.

    Yields connection parameters suitable for ``IFDBConnector(**params)``.  A
    container left behind by a previous ``--keep-container`` run is reused, and
    the import is skipped when its data is already loaded.
    """
    if not cfg.path.exists():
        raise FileNotFoundError(
            f"{cfg.path} not found. Download the IFDB archive dump to that path, "
            "or set database.dump.path in config.yaml."
        )

    runtime = _runtime()
    if _container_state(runtime, cfg.container_name) == "running":
        logger.info("Reusing running container '%s'", cfg.container_name)
    else:
        _remove_container(runtime, cfg.container_name)  # clear any stopped leftover
        _start_container(runtime, cfg)

    try:
        params = {
            "host": "127.0.0.1",
            "port": _host_port(runtime, cfg),
            "user": cfg.user,
            "password": cfg.password,
            "database": cfg.database,
        }
        _wait_ready(runtime, cfg, params)
        logger.info("MariaDB ready on %s:%d", params["host"], params["port"])

        if _already_loaded(params):
            logger.info("Dump already loaded in '%s' — skipping import", cfg.container_name)
        else:
            _import_dump(runtime, cfg, _find_client(runtime, cfg))
        yield params
    finally:
        if cfg.keep:
            logger.info(
                "Leaving container '%s' running (--keep-container). Remove it with: %s rm -f %s",
                cfg.container_name, runtime, cfg.container_name,
            )
        else:
            _remove_container(runtime, cfg.container_name)
