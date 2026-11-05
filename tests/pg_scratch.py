"""A throwaway Postgres schema for one test run, so tests never touch real records or each other.

    url, drop = scratch_url(os.environ["TEST_DATABASE_URL"])
    os.environ["DATABASE_URL"] = url      # every table the app creates lands in the scratch schema
    ...
    drop()
"""

from __future__ import annotations

import uuid
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit


def scratch_url(base: str):
    import psycopg

    schema = f"test_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(base, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
    parts = urlsplit(base)
    query = dict(parse_qsl(parts.query))
    query["options"] = f"-c search_path={schema}"
    url = urlunsplit(parts._replace(query=urlencode(query, quote_via=quote)))

    def drop() -> None:
        with psycopg.connect(base, autocommit=True) as conn:
            conn.execute(f'DROP SCHEMA "{schema}" CASCADE')

    return url, drop
