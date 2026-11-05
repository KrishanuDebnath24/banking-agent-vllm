"""What the web chat keeps in the database (storage.WEB), so any copy of the app can answer any
request, and a restart or a new copy signs nobody out.

    web_sessions   who is signed in: the hash of the cookie's random id -> the person (never a bank token)
    chat_turns     each message being answered: running, then its steps and result
    chat_pending   actions waiting for the customer's approval, and the security code sent for one
    chat_threads   the "Your conversations" list: each conversation's title (its first message,
                   without card numbers, codes or contact details) and when it was last used

Each change to one conversation runs in one short transaction holding that conversation's lock
(db.lock), so two copies of the app can't both start an answer, or both approve one slip.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

import storage
from storage import LEDGER, REQUESTS, WEB

_LOCK = threading.Lock()   # SQLite only: Postgres uses db.lock, across every copy of the app


@contextmanager
def _db(thread_id: str | None = None):
    with (nullcontext() if storage.is_postgres() else _LOCK), WEB.connect() as db:
        if thread_id:
            db.lock(f"thread:{thread_id}")
        yield db


def _hash(sid: str) -> str:
    return hashlib.sha256(sid.encode()).hexdigest()


# ---------- who is signed in ----------

@dataclass
class Person:
    sid_hash: str
    kind: str          # "guest" (a visitor with their own pretend bank) or "owner" (demo customer A or B)
    subject: str       # the visitor's guest id, or "A" / "B" ("" while the owner hasn't picked one)
    owner_key: str     # starts their conversation ids; their name in the cost ledger: "g1a2b3c4d5e", "A"
    email: str | None
    name: str

    @property
    def is_guest(self) -> bool:
        return self.kind == "guest"


def new_session(kind: str, subject: str, owner_key: str, email: str | None, name: str) -> str:
    """Returns the session id for the cookie. Only its hash is stored."""
    sid, now = secrets.token_urlsafe(32), time.time()
    with _db() as db:
        db.execute("INSERT INTO web_sessions (sid_hash, kind, subject, owner_key, email, name, created, last_seen)"
                   " VALUES (?,?,?,?,?,?,?,?)", (_hash(sid), kind, subject, owner_key, email, name, now, now))
    return sid


def get_session(sid: str, idle_seconds: float) -> Person | None:
    now = time.time()
    with _db() as db:
        row = db.execute("SELECT sid_hash, kind, subject, owner_key, email, name, last_seen FROM web_sessions"
                         " WHERE sid_hash=?", (_hash(sid),)).fetchone()
        if row is None:
            return None
        if now - row[6] > idle_seconds:
            db.execute("DELETE FROM web_sessions WHERE sid_hash=?", (row[0],))
            return None
        if now - row[6] > 60:   # keep writes down: the page asks for progress every second
            db.execute("UPDATE web_sessions SET last_seen=? WHERE sid_hash=?", (now, row[0]))
    return Person(*row[:6])


def choose_customer(sid: str, label: str) -> None:
    """The owner picks which demo customer (A or B) to be."""
    with _db() as db:
        db.execute("UPDATE web_sessions SET subject=?, owner_key=? WHERE sid_hash=? AND kind='owner'",
                   (label, label, _hash(sid)))


def end_session(sid: str) -> None:
    with _db() as db:
        db.execute("DELETE FROM web_sessions WHERE sid_hash=?", (_hash(sid),))


# ---------- actions waiting for approval ----------

@dataclass
class Pending:
    """Write actions waiting for the customer, across one or more interrupts."""
    items: list                                     # [interrupt id, action]
    decisions: list[dict] = field(default_factory=list)
    challenge: dict | None = None                   # security code sent for the current action: id, sent_to, expires

    @property
    def current(self) -> dict | None:
        return self.items[len(self.decisions)][1] if len(self.decisions) < len(self.items) else None

    @property
    def challenge_id(self) -> str | None:
        return (self.challenge or {}).get("id")

    def resume_value(self, decisions: list[dict]):
        by_interrupt: dict[str, list] = {}
        for (iid, _), decision in zip(self.items, decisions):
            by_interrupt.setdefault(iid, []).append(decision)
        answers = {iid: {"decisions": d} for iid, d in by_interrupt.items()}
        return next(iter(answers.values())) if len(answers) == 1 else answers


def _pending(db, thread_id: str) -> Pending | None:
    row = db.execute("SELECT items, decisions, challenge FROM chat_pending WHERE thread_id=?", (thread_id,)).fetchone()
    if row is None:
        return None
    return Pending(items=json.loads(row[0]), decisions=json.loads(row[1]),
                   challenge=json.loads(row[2]) if row[2] else None)


def _put_pending(db, thread_id: str, owner_key: str, p: Pending) -> None:
    db.execute("INSERT INTO chat_pending (thread_id, owner_key, items, decisions, challenge, updated) VALUES (?,?,?,?,?,?)"
               " ON CONFLICT (thread_id) DO UPDATE SET items=excluded.items, decisions=excluded.decisions,"
               " challenge=excluded.challenge, updated=excluded.updated",
               (thread_id, owner_key, json.dumps(p.items), json.dumps(p.decisions),
                json.dumps(p.challenge) if p.challenge else None, time.time()))


def get_pending(thread_id: str) -> Pending | None:
    with _db() as db:
        return _pending(db, thread_id)


def update_pending(thread_id: str, owner_key: str, seen: Pending, decisions: list[dict],
                   challenge: dict | None) -> bool:
    """Record a decision or a code sent, unless someone else changed this slip since `seen` was read."""
    with _db(thread_id) as db:
        now = _pending(db, thread_id)
        if now is None or len(now.decisions) != len(seen.decisions) or now.challenge_id != seen.challenge_id:
            return False
        _put_pending(db, thread_id, owner_key, Pending(now.items, decisions, challenge))
        return True


# ---------- turns (one message being answered) ----------

@dataclass
class Busy:
    """Why a conversation can't take a new message right now."""
    status: str                    # "running" or "needs_confirmation"
    turn_id: str | None = None
    pending: Pending | None = None


def _running(db, thread_id: str) -> str | None:
    row = db.execute("SELECT turn_id FROM chat_turns WHERE thread_id=? AND status='running'", (thread_id,)).fetchone()
    return row[0] if row else None


def _insert_turn(db, thread_id: str, owner_key: str, kind: str) -> str:
    turn_id, now = uuid.uuid4().hex[:16], time.time()
    db.execute("INSERT INTO chat_turns (turn_id, thread_id, owner_key, kind, status, steps, result, created, updated)"
               " VALUES (?,?,?,?,?,?,?,?,?)", (turn_id, thread_id, owner_key, kind, "running", "[]", None, now, now))
    return turn_id


def start_chat(thread_id: str, owner_key: str) -> tuple[str | None, Busy | None]:
    """Claim the conversation for a new message: (turn id, None), or (None, why not)."""
    with _db(thread_id) as db:
        running = _running(db, thread_id)
        if running:
            return None, Busy("running", turn_id=running)
        pending = _pending(db, thread_id)
        if pending is not None and pending.current is not None:
            return None, Busy("needs_confirmation", pending=pending)
        return _insert_turn(db, thread_id, owner_key, "chat"), None


def start_resume(thread_id: str, owner_key: str, seen: Pending, decisions: list[dict]) -> str | None:
    """Every action on the slip is decided: claim the conversation to carry on. None if someone else
    already did, or changed the slip since `seen` was read."""
    with _db(thread_id) as db:
        if _running(db, thread_id):
            return None
        now = _pending(db, thread_id)
        if now is None or len(now.decisions) != len(seen.decisions) or now.challenge_id != seen.challenge_id:
            return None
        db.execute("DELETE FROM chat_pending WHERE thread_id=?", (thread_id,))
        return _insert_turn(db, thread_id, owner_key, "confirm")


def save_steps(turn_id: str, steps: list) -> None:
    with _db() as db:
        db.execute("UPDATE chat_turns SET steps=?, updated=? WHERE turn_id=? AND status='running'",
                   (json.dumps(steps), time.time(), turn_id))


def finish_turn(turn_id: str, thread_id: str, owner_key: str, status: str, steps: list, result: dict,
                pending: Pending | None) -> None:
    """The answer, or the new slip, and the turn's record, in one transaction."""
    with _db(thread_id) as db:
        db.execute("UPDATE chat_turns SET status=?, steps=?, result=?, updated=? WHERE turn_id=?",
                   (status, json.dumps(steps), json.dumps(result), time.time(), turn_id))
        if pending is not None:
            _put_pending(db, thread_id, owner_key, pending)
        elif status != "failed":
            db.execute("DELETE FROM chat_pending WHERE thread_id=?", (thread_id,))
        db.execute("UPDATE chat_threads SET updated=? WHERE thread_id=?", (time.time(), thread_id))


def get_turn(turn_id: str) -> dict | None:
    with _db() as db:
        row = db.execute("SELECT thread_id, owner_key, status, steps, result FROM chat_turns WHERE turn_id=?",
                         (turn_id,)).fetchone()
    if row is None:
        return None
    return {"turn_id": turn_id, "thread_id": row[0], "owner_key": row[1], "status": row[2],
            "steps": json.loads(row[3]), "result": json.loads(row[4]) if row[4] else {}}


def charts_of(thread_id: str) -> dict[str, list]:
    """The charts drawn under this conversation's answers, by the answer's text, so the page can
    show them again after a reload. (Turn records are kept for 7 days.)"""
    with _db() as db:
        rows = db.execute("SELECT result FROM chat_turns WHERE thread_id=? AND status='answered' ORDER BY created",
                          (thread_id,)).fetchall()
    out: dict[str, list] = {}
    for (raw,) in rows:
        try:
            result = json.loads(raw) if raw else {}
        except ValueError:
            continue
        if result.get("charts") and result.get("answer"):
            out[result["answer"]] = result["charts"]
    return out


def running_turn(thread_id: str) -> str | None:
    with _db() as db:
        return _running(db, thread_id)


def busy_people(owner_keys: list[str]) -> set[str]:
    """Which of these people have an answer in progress."""
    if not owner_keys:
        return set()
    with _db() as db:
        rows = db.execute("SELECT DISTINCT owner_key FROM chat_turns WHERE status='running'").fetchall()
    return {r[0] for r in rows} & set(owner_keys)


def messages_since(owner_key: str, since: float) -> int:
    with _db() as db:
        return db.execute("SELECT COUNT(*) FROM chat_turns WHERE owner_key=? AND kind='chat' AND created>?",
                          (owner_key, since)).fetchone()[0]


def put_pending(thread_id: str, owner_key: str, pending: Pending) -> None:
    with _db(thread_id) as db:
        _put_pending(db, thread_id, owner_key, pending)


# ---------- clearing up ----------

def threads_of(owner_key: str) -> list[str]:
    with _db() as db:
        rows = db.execute("SELECT DISTINCT thread_id FROM chat_turns WHERE owner_key=?", (owner_key,)).fetchall()
        rows += db.execute("SELECT thread_id FROM chat_pending WHERE owner_key=?", (owner_key,)).fetchall()
        rows += db.execute("SELECT thread_id FROM chat_threads WHERE owner_key=?", (owner_key,)).fetchall()
    return sorted({r[0] for r in rows})


def forget_conversations(owner_key: str) -> None:
    with _db() as db:
        db.execute("DELETE FROM chat_turns WHERE owner_key=?", (owner_key,))
        db.execute("DELETE FROM chat_pending WHERE owner_key=?", (owner_key,))
        db.execute("DELETE FROM chat_threads WHERE owner_key=?", (owner_key,))


# ---------- the "Your conversations" list ----------

def _title(message: str) -> str:
    """A conversation's title: its first message, without card numbers, PINs, codes, passwords or
    contact details (the same rules as the chat's guardrail), on one line."""
    from guardrails import pii
    text, _ = pii.redact(message, pii.INPUT_BLOCK)
    text, _ = pii.redact(text, pii.INPUT_MASK)
    text = " ".join(text.split())
    return (text[:77].rstrip() + "…" if len(text) > 80 else text) or "Conversation"


def note_thread(thread_id: str, owner_key: str, message: str) -> None:
    """A message was sent: a new conversation gets its title; an old one moves to the top."""
    now = time.time()
    with _db() as db:
        db.execute("INSERT INTO chat_threads (thread_id, owner_key, title, created, updated) VALUES (?,?,?,?,?)"
                   " ON CONFLICT (thread_id) DO UPDATE SET updated=excluded.updated",
                   (thread_id, owner_key, _title(message), now, now))


def set_title(thread_id: str, owner_key: str, first_message: str, updated: float) -> str:
    """A title for a conversation from before the list existed."""
    title = _title(first_message)
    with _db() as db:
        db.execute("INSERT INTO chat_threads (thread_id, owner_key, title, created, updated) VALUES (?,?,?,?,?)"
                   " ON CONFLICT (thread_id) DO NOTHING", (thread_id, owner_key, title, updated, updated))
    return title


def conversations_of(owner_key: str, limit: int = 20) -> list[dict]:
    """This person's conversations, most recently used first. title is None for one from before the
    list existed (the API fills it in from the conversation's first message)."""
    with _db() as db:
        rows = db.execute("SELECT thread_id, title, updated FROM chat_threads WHERE owner_key=?"
                          " ORDER BY updated DESC LIMIT ?", (owner_key, limit)).fetchall()
        older = db.execute("SELECT thread_id, MAX(updated) FROM chat_turns WHERE owner_key=? GROUP BY thread_id",
                           (owner_key,)).fetchall()
        waiting = {r[0] for r in db.execute("SELECT thread_id FROM chat_pending WHERE owner_key=?",
                                            (owner_key,)).fetchall()}
        running = {r[0] for r in db.execute("SELECT DISTINCT thread_id FROM chat_turns WHERE owner_key=?"
                                            " AND status='running'", (owner_key,)).fetchall()}
    titled = {r[0] for r in rows}
    items = [{"thread_id": t, "title": title, "updated": u} for t, title, u in rows]
    items += [{"thread_id": t, "title": None, "updated": u} for t, u in older if t not in titled]
    items.sort(key=lambda c: -c["updated"])
    for c in items:
        c["waiting"], c["running"] = c["thread_id"] in waiting, c["thread_id"] in running
    return items[:limit]


def end_sessions_of(owner_key: str) -> None:
    with _db() as db:
        db.execute("DELETE FROM web_sessions WHERE owner_key=?", (owner_key,))


def forget_requests(guest_id: str) -> None:
    """A visitor's service requests (cheque books, address changes) go with their bank."""
    with REQUESTS.connect() as db:
        db.execute("DELETE FROM service_requests WHERE username=?", (f"guest:{guest_id}",))


def forget_old_codes(older_than_seconds: float = 86400) -> int:
    """Security code records (they name the customer's email) are only needed for a few minutes."""
    with storage.OTP.connect() as db:
        return db.execute("DELETE FROM challenges WHERE created<?", (time.time() - older_than_seconds,)).rowcount


STOPPED = "The app restarted while answering, so this didn't finish. Please send it again."


def tidy(idle_seconds: float, stale_seconds: float, keep_turns_seconds: float = 7 * 86400) -> dict:
    """Housekeeping, safe to run in every copy at once."""
    now = time.time()
    with _db() as db:
        stale = db.execute("UPDATE chat_turns SET status='failed', result=?, updated=? WHERE status='running'"
                           " AND updated<?", (json.dumps({"error": STOPPED}), now, now - stale_seconds)).rowcount
        idle = db.execute("DELETE FROM web_sessions WHERE last_seen<?", (now - idle_seconds,)).rowcount
        old = db.execute("DELETE FROM chat_turns WHERE status<>'running' AND updated<?",
                         (now - keep_turns_seconds,)).rowcount
    return {"stale_turns": stale, "idle_sessions": idle, "old_turns": old}


def visitor_spend_since(day_start_iso: str) -> float:
    """What visitors' turns have cost since then (USD), from the cost ledger. Visitors' ledger names
    start with g; the owner's demo customers are A and B."""
    with LEDGER.connect() as db:
        row = db.execute("SELECT COALESCE(SUM(total_cost_usd), 0) FROM turns WHERE ts>=? AND customer LIKE ?",
                         (day_start_iso, "g%")).fetchone()
    return float(row[0] or 0)
