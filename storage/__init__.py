"""Where the app keeps its records: SQLite files on your PC, one Postgres database in Docker and on AWS.

    DATABASE_URL=postgresql://bank:PASSWORD@host:5432/bank     use Postgres for everything
    DATABASE_URL=postgresql://  plus PGHOST, PGUSER, PGPASSWORD...   the same, settings split up (AWS)
    DATABASE_URL not set                                       SQLite files in the project folder

What is kept:
    conversations      LangGraph's checkpoint tables       checkpointer()   SQLite file: SESSION_DB
    security codes     challenges                          OTP              SQLite file: OTP_DB
    service requests   service_requests                    REQUESTS         SQLite file: SERVICE_REQUESTS_DB
    cost ledger        turns, llm_calls, guard_checks      LEDGER           SQLite file: TELEMETRY_DB
    visitors' banks    guest_customers, guest_accounts...  GUESTS           SQLite file: GUEST_DB
    web chat state     web_sessions, chat_turns, chat_pending, signin_codes, chat_threads   WEB   SQLite file: WEB_DB
    chart data         chart_datasets (figures saved for charts)   CHARTS       SQLite file: CHART_DB

The SQL in this project is written once for both databases: ? placeholders, plain types
(TEXT, INTEGER, DOUBLE PRECISION) and ON CONFLICT for upserts. Tables are created on first use,
so there is no separate migration step.

    with OTP.connect() as db:        # one short transaction: committed at the end, rolled back on error
        db.lock("otp:robert")        # Postgres: other processes wait here until this transaction ends
        db.execute("SELECT ... WHERE customer=?", ("robert",)).fetchone()
"""

from __future__ import annotations

import asyncio
import atexit
import os
import sqlite3
import sys
import threading
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent


def database_url() -> str | None:
    """The Postgres URL, or None to use SQLite files."""
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        return None
    if not url.startswith(("postgresql://", "postgres://")):
        raise RuntimeError("DATABASE_URL must start with postgresql:// (or leave it empty to use SQLite files).")
    return url


def is_postgres() -> bool:
    return database_url() is not None


def describe() -> str:
    """Where records go, for start-up banners. Never includes the password."""
    url = database_url()
    if not url:
        return "SQLite files"
    p = urlsplit(url)
    # On AWS the URL is just "postgresql://" and the driver reads PGHOST, PGUSER... instead.
    user = p.username or os.environ.get("PGUSER", "")
    host = p.hostname or os.environ.get("PGHOST", "localhost")
    port = p.port or os.environ.get("PGPORT", "5432")
    name = p.path.lstrip("/") or os.environ.get("PGDATABASE", "")
    return f"Postgres {user}@{host}:{port}/{name}"


def pool_size() -> int:
    return max(2, int(os.environ.get("DB_POOL_SIZE", "5")))


def loop_factory():
    """For asyncio.run(..., loop_factory=storage.loop_factory()).

    Postgres' async driver can't use Windows' default event loop, so on Windows with Postgres this
    returns the selector loop. Everywhere else it returns None (Python's default)."""
    if sys.platform == "win32" and is_postgres():
        return asyncio.SelectorEventLoop
    return None


# ---------- connections ----------

class Conn:
    """The same small API over sqlite3 and psycopg."""

    def __init__(self, raw, postgres: bool):
        self.raw, self.postgres = raw, postgres

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.postgres else sql

    def execute(self, sql: str, params: tuple | list = ()):
        """Returns a cursor: .fetchone(), .fetchall(), .rowcount."""
        if self.postgres:
            return self.raw.execute(self._sql(sql), params or None)
        return self.raw.execute(sql, params)

    def executemany(self, sql: str, rows) -> None:
        rows = list(rows)
        if not rows:
            return
        if self.postgres:
            with self.raw.cursor() as cur:
                cur.executemany(self._sql(sql), rows)
        else:
            self.raw.executemany(sql, rows)

    def lock(self, key: str) -> None:
        """Postgres: work on `key` waits for any other transaction holding it (in any process) until
        this transaction ends. SQLite needs nothing: its callers hold a process lock, and SQLite
        locks the whole file for writes."""
        if self.postgres:
            self.raw.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (key,))


_POOLS: dict[str, object] = {}
_POOL_LOCK = threading.Lock()


def _pool(url: str):
    with _POOL_LOCK:
        pool = _POOLS.get(url)
        if pool is None:
            from psycopg_pool import ConnectionPool
            pool = ConnectionPool(url, min_size=1, max_size=pool_size(), open=True, name="bank",
                                  check=ConnectionPool.check_connection)  # replaces connections the DB dropped
            _POOLS[url] = pool
        return pool


@atexit.register
def _close_pools() -> None:
    for pool in list(_POOLS.values()):
        pool.close()
    _POOLS.clear()


class Store:
    """One kind of record: its tables, and where they live."""

    def __init__(self, name: str, sqlite_env: str, sqlite_file: str, schema: str):
        self.name, self.sqlite_env, self.sqlite_file, self.schema = name, sqlite_env, sqlite_file, schema
        self._ready: set[str] = set()

    def sqlite_path(self) -> str:
        return os.environ.get(self.sqlite_env) or str(ROOT / self.sqlite_file)

    @property
    def where(self) -> str:
        return describe() if is_postgres() else self.sqlite_path()

    @contextmanager
    def connect(self):
        url = database_url()
        if url:
            with _pool(url).connection() as raw:   # commits at the end of the block, rolls back on error
                db = Conn(raw, postgres=True)
                self._create(db, url)
                yield db
            return
        path = self.sqlite_path()
        raw = sqlite3.connect(path, timeout=10)
        try:
            db = Conn(raw, postgres=False)
            self._create(db, path)
            yield db
            raw.commit()
        finally:
            raw.close()

    def _create(self, db: Conn, target: str) -> None:
        """Create the tables once per process, in their own committed transaction."""
        if target in self._ready:
            return
        db.lock(f"schema:{self.name}")  # two processes starting together don't both create tables
        for statement in self.schema.split(";"):
            if statement.strip():
                db.execute(statement)
        db.raw.commit()
        self._ready.add(target)


# ---------- the records ----------

OTP = Store("security codes", "OTP_DB", "verification.db", """
    CREATE TABLE IF NOT EXISTS challenges (
        id TEXT PRIMARY KEY, customer TEXT NOT NULL, tool TEXT NOT NULL, fingerprint TEXT NOT NULL,
        salt TEXT NOT NULL, code_hash TEXT NOT NULL, created DOUBLE PRECISION NOT NULL,
        expires DOUBLE PRECISION NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
        verified DOUBLE PRECISION, used DOUBLE PRECISION);
    CREATE INDEX IF NOT EXISTS challenges_customer ON challenges (customer, fingerprint)
""")

REQUESTS = Store("service requests", "SERVICE_REQUESTS_DB", "service_requests.db", """
    CREATE TABLE IF NOT EXISTS service_requests (
        reference TEXT PRIMARY KEY, created_at TEXT NOT NULL, created_ts DOUBLE PRECISION NOT NULL,
        customer TEXT NOT NULL, username TEXT NOT NULL, type TEXT NOT NULL, details TEXT NOT NULL,
        status TEXT NOT NULL, error TEXT, bank_message TEXT);
    CREATE INDEX IF NOT EXISTS service_requests_username ON service_requests (username, created_ts)
""")

LEDGER = Store("cost ledger", "TELEMETRY_DB", "telemetry.db", """
    CREATE TABLE IF NOT EXISTS turns (
        ts TEXT, turn_id TEXT PRIMARY KEY, thread_id TEXT, customer TEXT, model_calls INTEGER, tokens INTEGER,
        llm_cost_usd DOUBLE PRECISION, guard_cost_usd DOUBLE PRECISION, total_cost_usd DOUBLE PRECISION,
        latency_ms INTEGER, guard_notes TEXT);
    CREATE TABLE IF NOT EXISTS llm_calls (
        ts TEXT, turn_id TEXT, thread_id TEXT, customer TEXT, agent TEXT, model TEXT, input_tokens INTEGER,
        output_tokens INTEGER, cache_read INTEGER, cache_write INTEGER, cost_usd DOUBLE PRECISION);
    CREATE TABLE IF NOT EXISTS guard_checks (
        ts TEXT, turn_id TEXT, thread_id TEXT, customer TEXT, layer TEXT, source TEXT, mode TEXT,
        findings TEXT, applied INTEGER, units TEXT, cost_usd DOUBLE PRECISION, latency_ms INTEGER, error TEXT);
    CREATE INDEX IF NOT EXISTS turns_ts ON turns (ts);
    CREATE INDEX IF NOT EXISTS llm_calls_ts ON llm_calls (ts);
    CREATE INDEX IF NOT EXISTS guard_checks_ts ON guard_checks (ts)
""")


GUESTS = Store("guest banks", "GUEST_DB", "guest_banks.db", """
    CREATE TABLE IF NOT EXISTS guest_customers (
        guest_id TEXT PRIMARY KEY, email TEXT NOT NULL, name TEXT NOT NULL, verified INTEGER NOT NULL,
        ip_hash TEXT, created DOUBLE PRECISION NOT NULL, expires DOUBLE PRECISION NOT NULL);
    CREATE INDEX IF NOT EXISTS guest_customers_email ON guest_customers (email, expires);
    CREATE INDEX IF NOT EXISTS guest_customers_ip ON guest_customers (ip_hash, created);
    CREATE TABLE IF NOT EXISTS guest_accounts (
        account_id TEXT PRIMARY KEY, guest_id TEXT NOT NULL, role TEXT NOT NULL,
        balance DOUBLE PRECISION NOT NULL, currency TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS guest_accounts_guest ON guest_accounts (guest_id);
    CREATE TABLE IF NOT EXISTS guest_transactions (
        txn_id TEXT PRIMARY KEY, guest_id TEXT NOT NULL, account_id TEXT NOT NULL, completed TEXT NOT NULL,
        seq DOUBLE PRECISION NOT NULL, description TEXT NOT NULL, amount DOUBLE PRECISION NOT NULL,
        balance_after DOUBLE PRECISION NOT NULL);
    CREATE INDEX IF NOT EXISTS guest_transactions_account ON guest_transactions (account_id, seq);
    CREATE INDEX IF NOT EXISTS guest_transactions_guest ON guest_transactions (guest_id);
    CREATE TABLE IF NOT EXISTS guest_attributes (
        guest_id TEXT NOT NULL, name TEXT NOT NULL, value TEXT NOT NULL, inserted TEXT NOT NULL,
        PRIMARY KEY (guest_id, name));
    CREATE TABLE IF NOT EXISTS guest_messages (guest_id TEXT NOT NULL, ts TEXT NOT NULL, message TEXT NOT NULL)
""")

WEB = Store("web chat state", "WEB_DB", "web.db", """
    CREATE TABLE IF NOT EXISTS web_sessions (
        sid_hash TEXT PRIMARY KEY, kind TEXT NOT NULL, subject TEXT NOT NULL, owner_key TEXT NOT NULL,
        email TEXT, name TEXT NOT NULL, created DOUBLE PRECISION NOT NULL, last_seen DOUBLE PRECISION NOT NULL);
    CREATE INDEX IF NOT EXISTS web_sessions_owner ON web_sessions (owner_key);
    CREATE TABLE IF NOT EXISTS chat_turns (
        turn_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL, owner_key TEXT NOT NULL, kind TEXT NOT NULL,
        status TEXT NOT NULL, steps TEXT NOT NULL, result TEXT, created DOUBLE PRECISION NOT NULL,
        updated DOUBLE PRECISION NOT NULL);
    CREATE INDEX IF NOT EXISTS chat_turns_thread ON chat_turns (thread_id, created);
    CREATE INDEX IF NOT EXISTS chat_turns_owner ON chat_turns (owner_key, created);
    CREATE TABLE IF NOT EXISTS chat_pending (
        thread_id TEXT PRIMARY KEY, owner_key TEXT NOT NULL, items TEXT NOT NULL, decisions TEXT NOT NULL,
        challenge TEXT, updated DOUBLE PRECISION NOT NULL);
    CREATE TABLE IF NOT EXISTS signin_codes (
        id TEXT PRIMARY KEY, email TEXT NOT NULL, ip_hash TEXT NOT NULL, salt TEXT NOT NULL, code_hash TEXT NOT NULL,
        created DOUBLE PRECISION NOT NULL, expires DOUBLE PRECISION NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
        used DOUBLE PRECISION);
    CREATE INDEX IF NOT EXISTS signin_codes_email ON signin_codes (email, created);
    CREATE INDEX IF NOT EXISTS signin_codes_ip ON signin_codes (ip_hash, created);
    CREATE TABLE IF NOT EXISTS chat_threads (
        thread_id TEXT PRIMARY KEY, owner_key TEXT NOT NULL, title TEXT NOT NULL,
        created DOUBLE PRECISION NOT NULL, updated DOUBLE PRECISION NOT NULL);
    CREATE INDEX IF NOT EXISTS chat_threads_owner ON chat_threads (owner_key, updated)
""")

CHARTS = Store("chart data", "CHART_DB", "chart_data.db", """
    CREATE TABLE IF NOT EXISTS chart_datasets (
        dataset_id TEXT PRIMARY KEY, owner TEXT NOT NULL, kind TEXT NOT NULL,
        created DOUBLE PRECISION NOT NULL, payload TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS chart_datasets_owner ON chart_datasets (owner, created)
""")


# ---------- conversations (LangGraph checkpoints) ----------

@asynccontextmanager
async def checkpointer(sqlite_path: str | None = None):
    """LangGraph's conversation store, pending confirmations included.

    Postgres: AsyncPostgresSaver over a small connection pool (its tables are created or upgraded
    on start). SQLite: AsyncSqliteSaver on SESSION_DB, as before."""
    url = database_url()
    if not url:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        path = sqlite_path or os.environ.get("SESSION_DB") or str(ROOT / "sessions.db")
        async with AsyncSqliteSaver.from_conn_string(path) as saver:
            yield saver
        return

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(url, min_size=1, max_size=pool_size(), open=False, name="checkpoints",
                               check=AsyncConnectionPool.check_connection,
                               kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
    async with pool:
        saver = AsyncPostgresSaver(pool)
        async with pool.connection() as conn:  # one process at a time creates or upgrades the tables
            await conn.execute("SELECT pg_advisory_lock(hashtextextended('schema:checkpoints', 0))")
            try:
                await saver.setup()
            finally:
                await conn.execute("SELECT pg_advisory_unlock(hashtextextended('schema:checkpoints', 0))")
        yield saver
