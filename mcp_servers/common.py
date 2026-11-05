"""Shared code for the three MCP servers.

Identity comes only from the customer's bank token, sent by the agent as an
HTTP header on every MCP call. Nothing the model writes can change who the
customer is or which accounts they may touch. Two kinds of token:

    the owner's demo customers (A, B)   an OBP token: OBP says who it belongs to, and their
                                        accounts come from demo_accounts.json
    visitors                            guest.<id>.<signature>: their own pretend bank in the
                                        database (obp/guest_bank.py); the signature (APP_SECRET)
                                        proves the web chat issued it
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# Tests pass their own settings (fake bank, printed codes) and must not pick up a real .env.
if os.environ.get("BANK_TEST") != "1":
    load_dotenv(ROOT / ".env", override=True)

from agents.context import VISIBLE_ROLES, _first_seed_time, account_key  # noqa: E402
from obp.client import OBPClient, OBPError  # noqa: E402
from obp.guest_bank import BANK_ID as GUEST_BANK_ID  # noqa: E402
from obp.guest_bank import GUEST_BANK  # noqa: E402
from obp.search import payee_matches  # noqa: E402
from guardrails.injection import defang  # noqa: E402

DEMO_FILE = Path(os.environ.get("DEMO_ACCOUNTS_FILE", ROOT / "demo_accounts.json"))


def _make_client():
    if os.environ.get("OBP_FAKE") == "1":
        from obp.fake import FakeOBP
        fixture = os.environ.get("OBP_FAKE_FIXTURE")  # the fixed eval bank (evals/fixture.py)
        return FakeOBP.from_fixture(fixture) if fixture else FakeOBP.from_demo(DEMO_FILE)
    return OBPClient.from_env()


CLIENT = _make_client()
_OFFSET = int(os.environ.get("MCP_PORT_OFFSET", "0"))  # evals and tests run a second set of servers
PORTS = {"accounts": 8101 + _OFFSET, "transactions": 8102 + _OFFSET, "service": 8103 + _OFFSET,
         "verify": 8104 + _OFFSET,  # verify: security codes (plain HTTP, not MCP)
         "charts": 8105 + _OFFSET}  # charts: draws saved figures; no bank access


def transport_security():
    """Which addresses the MCP servers answer to.

    The MCP library only accepts requests addressed to 127.0.0.1 / localhost by default (protection
    against DNS rebinding). In Docker the web chat calls the servers by their service name ("mcp"),
    so MCP_ALLOWED_HOSTS adds names to that list; the protection itself stays on."""
    from mcp.server.transport_security import TransportSecuritySettings
    extra = [h.strip() for h in os.environ.get("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    names = ["127.0.0.1", "localhost", "[::1]", *extra]
    return TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                     allowed_hosts=[f"{n}:*" for n in names],
                                     allowed_origins=[f"http://{n}:*" for n in names])


@dataclass
class Customer:
    label: str                        # "A", "B", or "G" for a visitor
    username: str
    token: str = field(repr=False)
    bank_id: str
    currency: str
    accounts: dict[str, str]          # visible name -> account id
    show_since: str | None = None
    client: Any = field(default=None, repr=False, compare=False)   # None: the OBP sandbox (CLIENT)
    guest_id: str | None = None       # visitors only
    email: str | None = None          # where security codes go, if the sign-in proved this address

    @property
    def obp(self):
        """The bank this customer's accounts live in: the OBP sandbox, or the visitors' bank."""
        return self.client or CLIENT

    @property
    def records_key(self) -> str:
        """Whose service requests these are. A visitor's are deleted with their bank."""
        return f"guest:{self.guest_id}" if self.guest_id else self.username

    def resolve(self, account: str | None) -> str | None:
        """ "current", "my current account", or this customer's own account id -> the id; else None."""
        if account is None:
            return None
        key = account_key(account)
        if key in self.accounts:
            return self.accounts[key]
        return account.strip() if account.strip() in self.accounts.values() else None

    def name_of(self, account_id: str) -> str:
        return next((n for n, i in self.accounts.items() if i == account_id), account_id)

    def denied(self) -> str:
        return json.dumps({"error": "access_denied",
                           "message": "That account does not belong to this customer.",
                           "allowed_accounts": list(self.accounts)})


_CACHE: dict[str, tuple[Customer, float]] = {}
_TOKEN_RE = re.compile(r'token="?([^",\s]+)"?', re.I)


def _token_from_ctx(ctx) -> str:
    try:
        headers = ctx.request_context.request.headers
    except Exception as e:  # noqa: BLE001
        raise PermissionError("no HTTP request context") from e
    return token_from_header(headers.get("authorization", ""))


def token_from_header(auth: str) -> str:
    """The OBP token from an Authorization header: DirectLogin token="..." (or Bearer ...)."""
    m = _TOKEN_RE.search(auth)
    if m:
        return m.group(1)
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    raise PermissionError("missing customer token")


def _visitor(token: str) -> Customer:
    try:
        guest = GUEST_BANK.get(GUEST_BANK.verify(token))
    except OBPError as e:
        raise PermissionError("the bank did not accept this login; please sign in again") from e
    if guest is None:
        raise PermissionError("this demo bank has expired; please sign in again")
    return Customer(label="G", username=guest.email, token=token, bank_id=GUEST_BANK_ID,
                    currency=GUEST_BANK.currency, accounts=GUEST_BANK.accounts_of(guest.guest_id),
                    client=GUEST_BANK, guest_id=guest.guest_id, email=guest.email if guest.verified else None)


def _lookup(token: str) -> Customer:
    key = hashlib.sha256(token.encode()).hexdigest()
    hit = _CACHE.get(key)
    if hit and hit[1] > time.time():
        return hit[0]
    if len(_CACHE) > 5000:  # many visitors: forget the stale entries
        for k in [k for k, (_, until) in _CACHE.items() if until <= time.time()]:
            _CACHE.pop(k, None)
    if GUEST_BANK.is_guest_token(token):
        cust = _visitor(token)
        _CACHE[key] = (cust, time.time() + 60)   # short: a visitor's bank can be reset or expire
        return cust
    try:
        username = CLIENT.current_user(token).get("username")
    except OBPError as e:
        raise PermissionError("the bank did not accept this login; please log in again") from e
    demo = json.loads(DEMO_FILE.read_text())
    for label, c in demo["customers"].items():
        if c["username"] == username:
            cust = Customer(label=label, username=username, token=token, bank_id=demo["bank_id"],
                            currency=c["currency"], accounts={r: c[r] for r in VISIBLE_ROLES},
                            # The shared sandbox has years of other people's test data; the fake bank doesn't.
                            show_since=None if os.environ.get("OBP_FAKE") == "1" else _first_seed_time(label))
            _CACHE[key] = (cust, time.time() + 300)
            return cust
    raise PermissionError("this user is not a demo customer")


async def customer(ctx) -> Customer:
    """The customer behind this MCP call (raises PermissionError if unknown)."""
    token = _token_from_ctx(ctx)
    return await asyncio.to_thread(_lookup, token)


async def customer_for_token(token: str) -> Customer:
    """The customer behind a raw OBP token (raises PermissionError if unknown)."""
    return await asyncio.to_thread(_lookup, token)


async def bank(fn, *args, **kwargs):
    """Run a blocking OBP call without blocking the other servers."""
    return await asyncio.to_thread(fn, *args, **kwargs)


def error(code: str, **extra) -> str:
    return json.dumps({"error": code, **extra})


def parse_date(value: str | None, name: str) -> str | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10]).isoformat()
    except ValueError as e:
        raise ValueError(f"{name} must be a date like 2026-10-01") from e


def rows_for(cust: Customer, raw: list[dict], from_date: str | None, to_date: str | None,
             search: str | None) -> list[dict]:
    """OBP transactions -> simple rows, newest first, filtered by date and payee words."""
    terms = [t.strip().lower() for t in (search or "").split(",") if t.strip()]
    rows = []
    for t in raw:
        d = t.get("details", {})
        day = (d.get("completed") or d.get("posted") or "")[:10]
        if cust.show_since and day and day < cust.show_since:
            continue  # pre-demo noise in the shared sandbox
        if from_date and day < from_date:
            continue
        if to_date and day > to_date:
            continue
        desc = d.get("description") or ""
        if terms and not any(payee_matches(desc, term) for term in terms):
            continue
        desc = defang(desc)   # instruction-like text hidden before any model reads it
        rows.append({"date": day, "amount": round(float(d.get("value", {}).get("amount", 0) or 0), 2),
                     "description": desc, "balance_after": d.get("new_balance", {}).get("amount")})
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows


__all__ = ["CLIENT", "PORTS", "Customer", "transport_security", "OBPError", "bank", "customer", "customer_for_token", "error",
           "parse_date", "rows_for", "token_from_header"]
