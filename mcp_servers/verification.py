"""Security codes (one-time passcodes) for risky actions: the "Authorisation" box.

Before a risky action runs, the chat app asks this service to email the
customer a 6-digit code, then passes on what the customer typed. The code goes
from the keyboard to this service only; it never passes through the AI model.
The MCP tool for the action then checks that a code was confirmed for exactly
that action (same accounts and amount, or same address) before it does anything,
so neither the model nor a hand-made request can skip the step.

Which actions need a code:
    change_address                 always
    transfer_between_my_accounts   from OTP_TRANSFER_THRESHOLD (default 500) up to the transfer limit

HTTP API on port 8104, called by the chat app with the customer's OBP token:
    POST /verify/start  {"tool": ..., "args": {...}}
        -> {"required": false}
        -> {"required": true, "challenge_id": ..., "sent_to": "k***@gmail.com", "expires_in": 300}
    POST /verify/check  {"challenge_id": ..., "code": "123456"}
        -> {"verified": true}  |  {"verified": false, "error": ..., "attempts_left": 2}

Where codes go:
    visitors who signed in with Google or an emailed code   their own address
    the owner's demo customers (A, B), and visitors whose    OTP_EMAIL_TO (the demo customers'
    address wasn't checked (the dev sign-in)                 @example.com addresses get no mail,
                                                             and an unchecked address could be anyone's)

Settings (.env): OTP_TRANSFER_THRESHOLD, OTP_EMAIL_TO, EMAIL_BACKEND (console|smtp),
SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, EMAIL_FROM.
Codes are kept (hashed) in the challenges table: Postgres if DATABASE_URL is set, else OTP_DB.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from contextlib import contextmanager

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from mailer import mask, send_email  # noqa: F401 - send_email is replaced in tests
from mcp_servers.common import Customer, customer_for_token, token_from_header
from storage import OTP

CODE_TTL = 300          # seconds a code is valid, and how long a confirmed code may be used
MAX_ATTEMPTS = 3        # wrong guesses before a code is dead
MAX_CODES = 5           # codes per customer per RATE_WINDOW (stops email flooding)
RATE_WINDOW = 600
ADDRESS_MAX = 250       # OBP stores at most 255 characters

TRANSFER = "transfer_between_my_accounts"
ADDRESS = "change_address"
PROTECTED_ARGS = {TRANSFER: ("from_account", "to_account", "amount"), ADDRESS: ("new_address",)}

_LOCK = threading.Lock()  # one process; across processes, Postgres advisory locks (db.lock)


class VerificationError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


# ---------- which actions need a code ----------

def transfer_threshold() -> float | None:
    raw = os.environ.get("OTP_TRANSFER_THRESHOLD", "500").strip().lower()
    if raw in ("", "off", "none", "no"):
        return None
    return float(raw)


def _amount(args: dict) -> float | None:
    try:
        return round(float(args.get("amount")), 2)
    except (TypeError, ValueError):
        return None


def needs_code(tool: str, args: dict, max_transfer: float) -> bool:
    """max_transfer: the customer's bank's limit (cust.obp.max_transfer)."""
    if tool == ADDRESS:
        return True
    if tool == TRANSFER:
        amount, threshold = _amount(args), transfer_threshold()
        # Above the transfer limit the bank refuses anyway, so don't make the customer type a code first.
        return amount is not None and threshold is not None and threshold <= amount <= max_transfer
    return False


def describe(tool: str, args: dict) -> str:
    """What the customer is approving, shown in the email (so they can spot a wrong amount or address)."""
    if tool == TRANSFER:
        return f"Transfer £{_amount(args) or 0:.2f} from {args.get('from_account')} to {args.get('to_account')}"
    if tool == ADDRESS:
        return f"Change your address to: {' '.join(str(args.get('new_address', '')).split())}"
    return tool


def fingerprint(tool: str, args: dict) -> str:
    """Ties a code to one exact action: same tool, accounts and amount, or same address."""
    canon = {}
    for key in PROTECTED_ARGS.get(tool, ()):
        if key == "amount":
            canon[key] = f"{_amount(args) or 0:.2f}"
        else:
            canon[key] = " ".join(str(args.get(key) or "").split()).casefold()
    return hashlib.sha256(json.dumps([tool, canon], sort_keys=True).encode()).hexdigest()


# ---------- storage ----------

@contextmanager
def _db(cust: Customer):
    """One short transaction on the challenges table, holding this customer's lock until it ends,
    so two requests (or two servers) can't both use one code or both slip past the rate limit."""
    with _LOCK, OTP.connect() as db:
        db.lock(f"otp:{cust.username}")
        yield db


def _hash(salt: str, code: str) -> str:
    return hashlib.sha256((salt + code).encode()).hexdigest()


# ---------- email ----------

def recipient(cust: Customer) -> str:
    """A visitor's own address when their sign-in proved it's theirs. Otherwise OTP_EMAIL_TO: the demo
    customers' @example.com addresses can't receive mail, and an unchecked address could be anyone's."""
    if cust.email:
        return cust.email
    to = (os.environ.get(f"OTP_EMAIL_TO_{cust.label}") or os.environ.get("OTP_EMAIL_TO") or "").strip()
    if not to and os.environ.get("EMAIL_BACKEND", "console").strip().lower() == "smtp":
        raise VerificationError("email_failed", "OTP_EMAIL_TO is not set in .env, so there's nowhere to send the code.", 502)
    return to or cust.username


# ---------- the three operations ----------

def start(cust: Customer, tool: str, args: dict) -> dict:
    """Email a code if this action needs one."""
    if not needs_code(tool, args, cust.obp.max_transfer):
        return {"required": False}
    to = recipient(cust)
    now, fp = time.time(), fingerprint(tool, args)
    code = f"{secrets.randbelow(1_000_000):06d}"
    salt, challenge_id = secrets.token_hex(8), secrets.token_urlsafe(12)
    with _db(cust) as con:
        sent = con.execute("SELECT COUNT(*) FROM challenges WHERE customer=? AND created>?",
                           (cust.username, now - RATE_WINDOW)).fetchone()[0]
        if sent >= MAX_CODES:
            raise VerificationError("too_many_codes", "Too many codes requested. Please wait a few minutes.", 429)
        # A new code replaces any earlier unused one for the same action.
        con.execute("UPDATE challenges SET expires=? WHERE customer=? AND fingerprint=? AND used IS NULL",
                    (now, cust.username, fp))
        con.execute("INSERT INTO challenges (id, customer, tool, fingerprint, salt, code_hash, created, expires)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (challenge_id, cust.username, tool, fp, salt, _hash(salt, code), now, now + CODE_TTL))
    body =(f"Your security code is {code}\n\n"
            f"It confirms: {describe(tool, args)}\n"
            f"It expires in {CODE_TTL // 60} minutes.\n\n"
            "Never share this code. The bank will never ask you for it by phone, email or chat.\n"
            "If you didn't ask for this, someone may be using your login: contact the bank.")
    try:
        send_email(to, "Your banking demo security code", body)
    except Exception as e:  # noqa: BLE001
        with _db(cust) as con:
            con.execute("DELETE FROM challenges WHERE id=?", (challenge_id,))
        raise VerificationError("email_failed", f"Could not send the code email ({type(e).__name__}).", 502) from e
    return {"required": True, "challenge_id": challenge_id, "sent_to": mask(to), "expires_in": CODE_TTL}


def check(cust: Customer, challenge_id: str, code: str) -> dict:
    """Check what the customer typed. Three wrong tries and the code is dead."""
    now = time.time()
    with _db(cust) as con:
        row = con.execute("SELECT salt, code_hash, expires, attempts, verified, used FROM challenges"
                          " WHERE id=? AND customer=?", (challenge_id, cust.username)).fetchone()
        if row is None:
            return {"verified": False, "error": "unknown", "attempts_left": 0}
        salt, code_hash, expires, attempts, verified, used = row
        if verified and not used and expires > now:
            return {"verified": True}
        if used or expires <= now:
            return {"verified": False, "error": "expired", "attempts_left": 0}
        if attempts >= MAX_ATTEMPTS:
            return {"verified": False, "error": "too_many_attempts", "attempts_left": 0}
        if hmac.compare_digest(_hash(salt, str(code).strip()), code_hash):
            # The confirmed code may now be used once, within CODE_TTL.
            con.execute("UPDATE challenges SET verified=?, expires=? WHERE id=?", (now, now + CODE_TTL, challenge_id))
            return {"verified": True}
        con.execute("UPDATE challenges SET attempts=attempts+1 WHERE id=?", (challenge_id,))
        left = MAX_ATTEMPTS - attempts - 1
        return {"verified": False, "error": "wrong_code" if left else "too_many_attempts", "attempts_left": left}


def consume(cust: Customer, tool: str, args: dict) -> bool:
    """Called by the MCP tool just before it acts: uses up one confirmed code for exactly this action."""
    now = time.time()
    with _db(cust) as con:
        # "AND used IS NULL" twice: the outer one also holds if another server used it a moment ago.
        cur = con.execute(
            "UPDATE challenges SET used=? WHERE used IS NULL AND id=(SELECT id FROM challenges WHERE customer=?"
            " AND fingerprint=? AND verified IS NOT NULL AND used IS NULL AND expires>? ORDER BY verified DESC LIMIT 1)",
            (now, cust.username, fingerprint(tool, args), now))
        return cur.rowcount == 1


VERIFICATION_REQUIRED = {
    "error": "verification_required",
    "message": "This needs the security code emailed to the customer, and no confirmed code was found, "
               "so nothing was done. The customer can ask again to get a new code.",
}


# ---------- HTTP endpoints for the chat app ----------

async def _customer(request: Request) -> Customer:
    return await customer_for_token(token_from_header(request.headers.get("authorization", "")))


async def _handle(request: Request, fn) -> JSONResponse:
    try:
        cust = await _customer(request)
        body = await request.json()
        # In a thread: sending email can take seconds, and the MCP servers share this event loop.
        return JSONResponse(await asyncio.to_thread(fn, cust, body))
    except PermissionError as e:
        return JSONResponse({"error": "not_authorised", "message": str(e)}, status_code=401)
    except VerificationError as e:
        return JSONResponse({"error": e.code, "message": e.message}, status_code=e.status)
    except (ValueError, KeyError, TypeError):
        return JSONResponse({"error": "bad_request", "message": "Send JSON with the fields shown in the docs."},
                            status_code=400)


async def _start(request: Request) -> JSONResponse:
    return await _handle(request, lambda cust, b: start(cust, str(b["tool"]), dict(b.get("args") or {})))


async def _check(request: Request) -> JSONResponse:
    return await _handle(request, lambda cust, b: check(cust, str(b["challenge_id"]), str(b["code"])))


async def _health(request: Request) -> JSONResponse:
    return JSONResponse({"ok": True})


app = Starlette(routes=[Route("/verify/start", _start, methods=["POST"]),
                        Route("/verify/check", _check, methods=["POST"]),
                        Route("/healthz", _health)])  # for Docker / AWS health checks


def summary() -> str:
    """One line for the server start-up banner."""
    threshold = transfer_threshold()
    when = "address changes" + (f" and transfers from £{threshold:.2f}" if threshold is not None else "")
    if os.environ.get("EMAIL_BACKEND", "console").strip().lower() == "smtp":
        where = (f"emailed to signed-in visitors, else to "
                 f"{mask(os.environ.get('OTP_EMAIL_TO', '').strip() or '(OTP_EMAIL_TO not set)')}")
    else:
        where = "printed in this window (EMAIL_BACKEND=console)"
    return f"Security codes for {when}: {where}"
