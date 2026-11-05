"""A private pretend bank for every visitor, kept in the database.

Visitors to the public demo never touch the real OBP sandbox. Each one gets their own bank:
a current and a savings account with about two months of realistic activity, a profile and an
address. It behaves like OBPClient (the same methods the bank servers call), lives in the same
database as everything else (Postgres on AWS), and is deleted after GUEST_TTL_HOURS (24).

A visitor's bank token is signed with APP_SECRET, so the bank servers can check it without a
lookup table, and a made-up or edited token is refused:

    guest.<guest_id>.<signature>

    guest, is_new = GUEST_BANK.open("sam@example.org", "Sam", ip="81.2.3.4", verified=True)
    GUEST_BANK.balance_amount(guest.token, BANK_ID, GUEST_BANK.accounts_of(guest.guest_id)["current"])
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import random
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from obp.client import OBPError
from storage import GUESTS

BANK_ID = "demo.bank"
CURRENCY = "GBP"
ROLES = ("current", "savings", "income", "merchants")   # income/merchants: the other side of payments, hidden
VISIBLE = ("current", "savings")
DEV_SECRET = "dev-only-secret-not-for-aws"
_LOCK = threading.Lock()   # SQLite: one change at a time in this process (Postgres also uses db.lock)
_warned = {"secret": False}
log = logging.getLogger("bank.guests")


def app_secret() -> bytes:
    """APP_SECRET signs visitor tokens and hashes their addresses. The web chat and the bank servers
    must share it (Docker: .env; AWS: Secrets Manager)."""
    value = os.environ.get("APP_SECRET", "").strip()
    if not value and not _warned["secret"]:
        _warned["secret"] = True
        log.warning("APP_SECRET is not set: using a development secret. Set it in .env before going public.")
    return (value or DEV_SECRET).encode()


def ttl_seconds() -> float:
    return float(os.environ.get("GUEST_TTL_HOURS", "24")) * 3600


def _sign(guest_id: str) -> str:
    return hmac.new(app_secret(), f"guest:{guest_id}".encode(), hashlib.sha256).hexdigest()[:32]


def token_for(guest_id: str) -> str:
    """The bank token for this visitor (the web chat recomputes it; it's never stored)."""
    return f"guest.{guest_id}.{_sign(guest_id)}"


def owner_key_for(guest_id: str) -> str:
    """Prefix of this visitor's conversation ids, and their name in the cost ledger."""
    return f"g{guest_id[:10]}"


def hash_ip(ip: str) -> str:
    """Addresses are only kept as a keyed hash: enough to count sign-ups, not to identify anyone."""
    return hmac.new(app_secret(), f"ip:{ip}".encode(), hashlib.sha256).hexdigest()[:24]


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Guest:
    guest_id: str
    email: str
    name: str
    verified: bool       # True when the address came from a real sign-in (Google, or an emailed code)
    expires: float

    @property
    def token(self) -> str:
        return token_for(self.guest_id)

    @property
    def owner_key(self) -> str:
        return owner_key_for(self.guest_id)


# ---------- sample data ----------

EMPLOYERS = ("ACME Analytics Ltd", "Brightwave Studio", "Northgate Media", "Harbour Software", "Riverside Labs")
LANDLORDS = ("Harbour Lettings", "Dockside Apartments", "Quayside Homes", "Riverview Rentals")
GROCERS = ("Tesco Stores 2041", "Lidl", "Aldi", "Sainsbury's 0612", "Co-op", "Waitrose")
COFFEE = ("Costa Coffee", "Pret A Manger", "Caffe Nero")
ADDRESSES = ("4 Harbour View, Brighton, BN2 5XY", "17 Quay Street, Bristol, BS1 4DJ",
             "22 Mill Lane, York, YO1 7HB", "9 Castle Road, Cardiff, CF10 3AT")


def _sample_activity(rng: random.Random, today: date) -> tuple[dict[str, float], list[tuple]]:
    """Opening balances and ~60 days of (day, account, amount, description), oldest first."""
    employer, landlord = rng.choice(EMPLOYERS), rng.choice(LANDLORDS)
    salary = round(rng.uniform(2400, 3300) / 5) * 5.0
    rent = round(rng.uniform(750, 1150) / 5) * 5.0
    rows: list[tuple] = []
    day = today - timedelta(days=60)
    while day <= today:
        dom = day.day
        if dom == 1:
            rows.append((day, "current", -rent, f"Rent - {landlord}"))
        if dom == 25:
            rows.append((day, "current", salary, f"{employer} - Salary"))
        if dom == 15:
            rows += [(day, "current", -200.0, "Monthly saving"), (day, "savings", 200.0, "Monthly saving")]
        if dom == 5:
            rows.append((day, "current", -10.99, "Netflix.com"))
        if dom == 8:
            rows.append((day, "current", -11.99, "Spotify"))
        if dom == 12:
            rows.append((day, "current", -39.99, "PureGym"))
        if dom == 20:
            rows.append((day, "current", -25.00, "Vodafone UK"))
        if (day + timedelta(days=1)).day == 1:
            rows.append((day, "savings", round(rng.uniform(0.8, 3.2), 2), "Savings interest"))
        if day.weekday() in (1, 5) or rng.random() < 0.08:
            rows.append((day, "current", -round(rng.uniform(18, 85), 2), rng.choice(GROCERS)))
        if rng.random() < 0.18:
            rows.append((day, "current", -round(rng.uniform(3.2, 5.4), 2), rng.choice(COFFEE)))
        if dom % 10 == 3:
            rows.append((day, "current", -30.00, "TfL Travel"))
        if rng.random() < 0.06:
            rows.append((day, "current", -round(rng.uniform(16, 34), 2), "Deliveroo"))
        day += timedelta(days=1)
    opening = {"current": round(rng.uniform(1800, 3200), 2), "savings": round(rng.uniform(600, 2600), 2),
               "income": 0.0, "merchants": 0.0}
    return opening, rows


def _seed(db, guest_id: str, now: float) -> None:
    """Accounts, transactions and an address for a new (or reset) bank."""
    rng = random.Random(secrets.randbits(64))
    opening, rows = _sample_activity(rng, datetime.fromtimestamp(now, timezone.utc).date())
    ids = {role: f"{role[:3]}-{guest_id}" for role in ROLES}
    balances = dict(opening)
    txns = []
    for n, (day, role, amount, desc) in enumerate(rows):
        balances[role] = round(balances[role] + amount, 2)
        start_of_day = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()
        ts = min(start_of_day + 60 * n, now - 1)   # in order, and never in the future
        txns.append((f"t-{guest_id}-{n}", guest_id, ids[role], _iso(ts), ts, desc, amount, balances[role]))
    db.executemany("INSERT INTO guest_accounts (account_id, guest_id, role, balance, currency) VALUES (?,?,?,?,?)",
                   [(ids[r], guest_id, r, balances[r], CURRENCY) for r in ROLES])
    db.executemany("INSERT INTO guest_transactions (txn_id, guest_id, account_id, completed, seq, description,"
                   " amount, balance_after) VALUES (?,?,?,?,?,?,?,?)", txns)
    db.execute("INSERT INTO guest_attributes (guest_id, name, value, inserted) VALUES (?,?,?,?)",
               (guest_id, "ADDRESS", rng.choice(ADDRESSES), _iso(now)))


_DATA_TABLES = ("guest_transactions", "guest_accounts", "guest_attributes", "guest_messages")


def _wipe(db, guest_id: str, everything: bool) -> None:
    for table in _DATA_TABLES + (("guest_customers",) if everything else ()):
        db.execute(f"DELETE FROM {table} WHERE guest_id=?", (guest_id,))  # noqa: S608 - fixed table names


# ---------- the bank ----------

class GuestBank:
    """OBPClient-compatible, backed by the guest_* tables. One instance serves every visitor."""

    @property
    def max_transfer(self) -> float:
        """The same single-transfer limit as the OBP sandbox set-up (OBP_MAX_TRANSFER, default 800)."""
        return float(os.environ.get("OBP_MAX_TRANSFER", "800"))

    currency = CURRENCY
    bank_id = BANK_ID

    # ---- tokens ----
    @staticmethod
    def is_guest_token(token: str) -> bool:
        return token.startswith("guest.")

    @staticmethod
    def verify(token: str) -> str:
        """The guest id inside a genuine token; OBPError(401) for anything else."""
        parts = token.split(".")
        if len(parts) != 3 or parts[0] != "guest" or not parts[1]:
            raise OBPError(401, '{"message":"not a visitor token"}', "guest")
        if not hmac.compare_digest(parts[2], _sign(parts[1])):
            raise OBPError(401, '{"message":"visitor token signature is wrong"}', "guest")
        return parts[1]

    def _guest_for(self, token: str) -> Guest:
        guest = self.get(self.verify(token))
        if guest is None:
            raise OBPError(401, '{"message":"this demo bank has expired"}', "guest")
        return guest

    @staticmethod
    def _own_account(db, guest_id: str, account_id: str) -> tuple[float, str]:
        row = db.execute("SELECT balance, currency FROM guest_accounts WHERE account_id=? AND guest_id=?",
                         (account_id, guest_id)).fetchone()
        if row is None:
            raise OBPError(404, '{"message":"OBP-30018: Bank Account not found"}', "guest")
        return round(row[0], 2), row[1]

    # ---- managing visitors ----
    def open(self, email: str, name: str, ip: str | None = None, verified: bool = False) -> tuple[Guest, bool]:
        """This visitor's bank (the existing one if it hasn't expired, else a new one), and whether it's new."""
        email = email.strip().lower()
        now = time.time()
        with _LOCK, GUESTS.connect() as db:
            db.lock(f"guest-email:{email}")   # two sign-ins at once still make one bank
            row = db.execute("SELECT guest_id, name, verified, expires FROM guest_customers WHERE email=? AND expires>?"
                             " ORDER BY created DESC LIMIT 1", (email, now)).fetchone()
            if row:
                if verified and not row[2]:  # signed in properly this time
                    db.execute("UPDATE guest_customers SET verified=1 WHERE guest_id=?", (row[0],))
                return Guest(row[0], email, row[1], bool(row[2]) or verified, row[3]), False
            guest_id = secrets.token_hex(10)
            expires = now + ttl_seconds()
            db.execute("INSERT INTO guest_customers (guest_id, email, name, verified, ip_hash, created, expires) "
                       "VALUES (?,?,?,?,?,?,?)",
                       (guest_id, email, name[:80], int(verified), hash_ip(ip) if ip else None, now, expires))
            _seed(db, guest_id, now)
        return Guest(guest_id, email, name[:80], verified, expires), True

    def active(self, email: str) -> Guest | None:
        """The visitor's bank for this address, if they have one that hasn't expired."""
        with GUESTS.connect() as db:
            row = db.execute("SELECT guest_id, name, verified, expires FROM guest_customers WHERE email=? AND expires>?"
                             " ORDER BY created DESC LIMIT 1", (email.strip().lower(), time.time())).fetchone()
        return Guest(row[0], email.strip().lower(), row[1], bool(row[2]), row[3]) if row else None

    def get(self, guest_id: str) -> Guest | None:
        with GUESTS.connect() as db:
            row = db.execute("SELECT email, name, verified, expires FROM guest_customers WHERE guest_id=? AND expires>?",
                             (guest_id, time.time())).fetchone()
        return Guest(guest_id, row[0], row[1], bool(row[2]), row[3]) if row else None

    def accounts_of(self, guest_id: str) -> dict[str, str]:
        """The visitor's own accounts: {"current": id, "savings": id}."""
        with GUESTS.connect() as db:
            rows = db.execute("SELECT role, account_id FROM guest_accounts WHERE guest_id=?", (guest_id,)).fetchall()
        return {role: acc for role, acc in rows if role in VISIBLE}

    def reset(self, guest_id: str) -> Guest:
        """Start over: same visitor, new balances and activity, another GUEST_TTL_HOURS."""
        now = time.time()
        with _LOCK, GUESTS.connect() as db:
            db.lock(f"guest:{guest_id}")
            row = db.execute("SELECT email, name, verified FROM guest_customers WHERE guest_id=? AND expires>?",
                             (guest_id, now)).fetchone()
            if row is None:
                raise OBPError(401, '{"message":"this demo bank has expired"}', "guest")
            _wipe(db, guest_id, everything=False)
            expires = now + ttl_seconds()
            db.execute("UPDATE guest_customers SET expires=? WHERE guest_id=?", (expires, guest_id))
            _seed(db, guest_id, now)
        return Guest(guest_id, row[0], row[1], bool(row[2]), expires)

    def signups_from(self, ip: str, within_seconds: float = 3600) -> int:
        with GUESTS.connect() as db:
            return db.execute("SELECT COUNT(*) FROM guest_customers WHERE ip_hash=? AND created>?",
                              (hash_ip(ip), time.time() - within_seconds)).fetchone()[0]

    def expired(self) -> list[str]:
        with GUESTS.connect() as db:
            return [r[0] for r in db.execute("SELECT guest_id FROM guest_customers WHERE expires<=?",
                                             (time.time(),)).fetchall()]

    def delete(self, guest_ids: list[str]) -> None:
        """Delete these visitors' banks (the web chat deletes their conversations)."""
        for guest_id in guest_ids:
            with _LOCK, GUESTS.connect() as db:
                db.lock(f"guest:{guest_id}")
                _wipe(db, guest_id, everything=True)

    def count(self) -> int:
        with GUESTS.connect() as db:
            return db.execute("SELECT COUNT(*) FROM guest_customers WHERE expires>?", (time.time(),)).fetchone()[0]

    # ---- OBPClient-compatible methods (what the bank servers call) ----
    def current_user(self, token: str) -> dict:
        guest = self._guest_for(token)
        return {"user_id": guest.guest_id, "username": guest.email}

    def balance_amount(self, token: str, bank_id: str, account_id: str) -> tuple[float, str]:
        guest = self._guest_for(token)
        with GUESTS.connect() as db:
            return self._own_account(db, guest.guest_id, account_id)

    def transactions(self, token: str, bank_id: str, account_id: str, view_id: str = "owner") -> list[dict]:
        guest = self._guest_for(token)
        with GUESTS.connect() as db:
            _, cur = self._own_account(db, guest.guest_id, account_id)
            rows = db.execute("SELECT completed, description, amount, balance_after FROM guest_transactions "
                              "WHERE account_id=? AND guest_id=? ORDER BY seq DESC",
                              (account_id, guest.guest_id)).fetchall()
        return [{"details": {"completed": c, "description": d,
                             "value": {"currency": cur, "amount": f"{a:.2f}"},
                             "new_balance": {"currency": cur, "amount": f"{b:.2f}"}}} for c, d, a, b in rows]

    def transfer(self, token, bank_id, from_account, to_account, amount, currency, description, view_id="owner"):
        guest = self._guest_for(token)
        amount = round(float(amount), 2)
        if amount <= 0:
            raise OBPError(400, '{"message":"OBP-40008: amount must be positive"}', "guest")
        if amount > self.max_transfer:
            raise OBPError(409, '{"message":"OBP-40044: needs a second approver"}', "guest")
        now = time.time()
        with _LOCK, GUESTS.connect() as db:
            db.lock(f"guest:{guest.guest_id}")   # one change at a time per visitor, across every server
            src, cur = self._own_account(db, guest.guest_id, from_account)
            dst, _ = self._own_account(db, guest.guest_id, to_account)
            if amount > src:
                raise OBPError(400, '{"message":"OBP-40001: insufficient funds"}', "guest")
            ref = secrets.token_hex(4)
            for n, (acc, change, new) in enumerate(((from_account, -amount, round(src - amount, 2)),
                                                    (to_account, amount, round(dst + amount, 2)))):
                db.execute("UPDATE guest_accounts SET balance=? WHERE account_id=?", (new, acc))
                db.execute("INSERT INTO guest_transactions (txn_id, guest_id, account_id, completed, seq, description,"
                           " amount, balance_after) VALUES (?,?,?,?,?,?,?,?)",
                           (f"t-{guest.guest_id}-{ref}-{n}", guest.guest_id, acc, _iso(now), now + n / 10,
                            str(description)[:100], change, new))
        return {"id": f"tr-{ref}", "status": "COMPLETED"}

    def my_customers(self, token: str, bank_id: str) -> list[dict]:
        guest = self._guest_for(token)
        number = str(int(guest.guest_id[:6], 16) % 900000 + 100000)
        return [{"customer_id": f"cust-{guest.guest_id}", "customer_number": number, "legal_name": guest.name,
                 "email": guest.email, "mobile_phone_number": "+44 7700 900123", "kyc_status": True}]

    def my_user_attributes(self, token: str) -> list[dict]:
        guest = self._guest_for(token)
        with GUESTS.connect() as db:
            rows = db.execute("SELECT name, value, inserted FROM guest_attributes WHERE guest_id=?",
                              (guest.guest_id,)).fetchall()
        return [{"user_attribute_id": f"attr-{n}", "name": n, "type": "STRING", "value": v, "insert_date": i}
                for n, v, i in rows]

    def set_my_user_attribute(self, token: str, name: str, value: str) -> dict:
        guest = self._guest_for(token)
        now = _iso(time.time())
        with _LOCK, GUESTS.connect() as db:
            db.execute("INSERT INTO guest_attributes (guest_id, name, value, inserted) VALUES (?,?,?,?) "
                       "ON CONFLICT (guest_id, name) DO UPDATE SET value=excluded.value, inserted=excluded.inserted",
                       (guest.guest_id, name, value[:255], now))
        return {"user_attribute_id": f"attr-{name}", "name": name, "type": "STRING", "value": value[:255],
                "insert_date": now}

    def create_customer_message(self, token, bank_id, customer_id, message, **kw) -> dict:
        guest = self._guest_for(token)
        with _LOCK, GUESTS.connect() as db:
            db.execute("INSERT INTO guest_messages (guest_id, ts, message) VALUES (?,?,?)",
                       (guest.guest_id, _iso(time.time()), message[:2000]))
        return {"success": "Success"}


GUEST_BANK = GuestBank()
