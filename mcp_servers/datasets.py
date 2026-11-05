"""Figures saved for charts ("datasets").

The transactions server works out the figures (in code) and saves them here under a short id such
as ds-4f2a9c1e, tied to the customer. Only the id travels through the AI: the chart server opens the
dataset by id and draws it, so the figures in a chart always come from the bank server's code. The
same id can be drawn again in another form later in the conversation, without asking the bank again.

A dataset belongs to the customer whose token saved it; anyone else asking for that id gets
"not found". Visitors' datasets go with their bank (24 hours, or "Start over"); others are deleted
after 7 days (api/state.py tidy).
"""

from __future__ import annotations

import json
import re
import secrets
import time

from storage import CHARTS

ID = re.compile(r"^ds-[0-9a-f]{8}$")
KEEP_SECONDS = 7 * 86400


def save(owner: str, kind: str, payload: dict) -> str:
    dataset_id = f"ds-{secrets.token_hex(4)}"
    with CHARTS.connect() as db:
        db.execute("INSERT INTO chart_datasets (dataset_id, owner, kind, created, payload) VALUES (?,?,?,?,?)",
                   (dataset_id, owner, kind, time.time(), json.dumps(payload)))
    return dataset_id


def load(owner: str, dataset_id: str) -> dict | None:
    """The dataset, if it exists and is this customer's; otherwise None (never someone else's)."""
    dataset_id = (dataset_id or "").strip().lower()
    if not ID.match(dataset_id):
        return None
    with CHARTS.connect() as db:
        row = db.execute("SELECT kind, payload, created FROM chart_datasets WHERE dataset_id=? AND owner=?",
                         (dataset_id, owner)).fetchone()
    if row is None or time.time() - row[2] > KEEP_SECONDS:
        return None
    return {"kind": row[0], **json.loads(row[1]), "dataset_id": dataset_id}


def forget(owner: str) -> int:
    with CHARTS.connect() as db:
        return db.execute("DELETE FROM chart_datasets WHERE owner=?", (owner,)).rowcount


def tidy(older_than_seconds: float = KEEP_SECONDS) -> int:
    with CHARTS.connect() as db:
        return db.execute("DELETE FROM chart_datasets WHERE created<?", (time.time() - older_than_seconds,)).rowcount
