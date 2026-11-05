"""Real sign-in (AUTH_MODE=public): "Continue with Google", or a code emailed to you.

Google (OpenID Connect, authorization code flow)
    GET /api/auth/google            sends the person to Google. A random state, nonce and PKCE secret go
                                    with them in a short-lived signed cookie, so any copy of the app can
                                    finish the sign-in, and a link someone else started can't.
    GET /api/auth/google/callback   Google sends them back with a one-time code. The app swaps it, with
                                    its client secret, for an ID token straight from Google over HTTPS
                                    (so the token can't have been tampered with on the way), then checks
                                    issuer, audience, expiry, nonce, and that Google verified the address.
Email code
    POST /api/auth/email/start {email}               a 6-digit code by email, valid 10 minutes
    POST /api/auth/email/check {challenge, code}     three tries per code

Codes are kept hashed. Limits stop the app being used to flood someone's inbox, and keep within
Gmail's 500 emails a day (security codes share that allowance):
    SIGNIN_CODES_PER_EMAIL=3        per address per 15 minutes
    SIGNIN_CODES_PER_IP_HOUR=10     per internet address per hour
    SIGNIN_EMAILS_PER_DAY=300       in total per 24 hours

Invite-only (OPEN_TO_PUBLIC=0): only OWNER_EMAILS and INVITED_EMAILS can sign in; nobody else is
even sent a code.

Who is who: an address in OWNER_EMAILS is the owner (then picks Robert or Anil, the OBP sandbox
customers). Everyone else gets their own pretend bank, and their security codes go to the address
they just proved is theirs.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from contextlib import contextmanager, nullcontext
from urllib.parse import urlencode

import httpx

import storage
from mailer import mask, send_email
from obp.guest_bank import app_secret, hash_ip
from storage import WEB

GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
GOOGLE_ISSUERS = {"https://accounts.google.com", "accounts.google.com"}
OAUTH_COOKIE = "bank_oauth"
OAUTH_SECONDS = 600
CODE_TTL = 600
MAX_ATTEMPTS = 3
_LOCK = threading.Lock()   # SQLite only; Postgres uses db.lock across every copy


class SignInError(Exception):
    """kind: a short word the page turns into a message; message: what to tell the person."""

    def __init__(self, kind: str, message: str, status: int = 400):
        super().__init__(message)
        self.kind, self.message, self.status = kind, message, status


def auth_mode() -> str:
    return os.environ.get("AUTH_MODE", "dev").strip().lower()


def public_url() -> str:
    """Where people open the app. Google sends them back here, so it must match the Google client."""
    return os.environ.get("PUBLIC_URL", "").strip().rstrip("/") or "http://127.0.0.1:8000"


def google_redirect_uri() -> str:
    return f"{public_url()}/api/auth/google/callback"


def google_client() -> tuple[str, str]:
    return os.environ.get("GOOGLE_CLIENT_ID", "").strip(), os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()


def google_enabled() -> bool:
    return all(google_client())


def owner_emails() -> set[str]:
    return {e.strip().lower() for e in os.environ.get("OWNER_EMAILS", "").split(",") if e.strip()}


def is_owner(email: str) -> bool:
    return email.strip().lower() in owner_emails()


def open_to_public() -> bool:
    """OPEN_TO_PUBLIC=0: invite-only. Only OWNER_EMAILS and INVITED_EMAILS can sign in."""
    return os.environ.get("OPEN_TO_PUBLIC", "1").strip() != "0"


def may_sign_in(email: str) -> bool:
    if open_to_public():
        return True
    invited = {e.strip().lower() for e in os.environ.get("INVITED_EMAILS", "").split(",") if e.strip()}
    return email.strip().lower() in owner_emails() | invited


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


# ---------- Google ----------

def seal(data: dict, ttl: float = OAUTH_SECONDS) -> str:
    """A cookie value nobody can read-and-edit without APP_SECRET (it holds no personal data)."""
    payload = _b64(json.dumps({**data, "exp": time.time() + ttl}).encode())
    sig = _b64(hmac.new(app_secret(), f"oauth:{payload}".encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}"


def unseal(value: str) -> dict | None:
    try:
        payload, sig = value.split(".")
        good = _b64(hmac.new(app_secret(), f"oauth:{payload}".encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, good):
            return None
        data = json.loads(_unb64(payload))
    except (ValueError, TypeError):
        return None
    return data if data.get("exp", 0) > time.time() else None


def google_start() -> tuple[str, str]:
    """(Google's sign-in address, the cookie that remembers this attempt)."""
    client_id, _ = google_client()
    state, nonce, verifier = secrets.token_urlsafe(24), secrets.token_urlsafe(24), secrets.token_urlsafe(48)
    params = {"client_id": client_id, "redirect_uri": google_redirect_uri(), "response_type": "code",
              "scope": "openid email profile", "state": state, "nonce": nonce,
              "code_challenge": _b64(hashlib.sha256(verifier.encode()).digest()),
              "code_challenge_method": "S256", "prompt": "select_account"}
    return f"{GOOGLE_AUTH}?{urlencode(params)}", seal({"state": state, "nonce": nonce, "verifier": verifier})


def _claims(id_token: str) -> dict:
    try:
        return json.loads(_unb64(id_token.split(".")[1]))
    except (IndexError, ValueError):
        raise SignInError("google", "Google's reply couldn't be read. Please try again.") from None


async def google_finish(code: str, state: str, cookie: str, transport: httpx.AsyncBaseTransport | None = None) -> dict:
    """Swap Google's one-time code for the person's verified email address and first name."""
    saved = unseal(cookie or "")
    if saved is None:
        raise SignInError("expired", "That sign-in took too long, or was started in another browser. Please try again.")
    if not code or not hmac.compare_digest(str(state), saved["state"]):
        raise SignInError("state", "That sign-in link wasn't started here. Please try again.")
    client_id, client_secret = google_client()
    try:
        async with httpx.AsyncClient(timeout=15, transport=transport) as http:
            r = await http.post(GOOGLE_TOKEN, data={
                "code": code, "client_id": client_id, "client_secret": client_secret,
                "redirect_uri": google_redirect_uri(), "grant_type": "authorization_code",
                "code_verifier": saved["verifier"]})
    except httpx.HTTPError as e:
        raise SignInError("google", "Google couldn't be reached. Please try again.", 502) from e
    if r.status_code != 200:
        raise SignInError("google", "Google didn't accept the sign-in. Please try again.")
    # The token came straight from Google over HTTPS in exchange for our client secret, so (as Google's
    # OpenID Connect guide says) its signature needn't be checked again; its contents still are.
    c = _claims(r.json().get("id_token", ""))
    aud = c.get("aud")
    if (c.get("iss") not in GOOGLE_ISSUERS or (aud != client_id and client_id not in (aud or []))
            or float(c.get("exp", 0)) < time.time() - 60 or not hmac.compare_digest(str(c.get("nonce", "")), saved["nonce"])):
        raise SignInError("google", "Google's reply didn't check out. Please try again.")
    if str(c.get("email_verified")).lower() != "true" or not c.get("email"):
        raise SignInError("unverified", "Your Google account's email address isn't verified, so it can't be used here.")
    return {"email": str(c["email"]).strip().lower(), "name": str(c.get("given_name") or c.get("name") or "")[:40]}


# ---------- email codes ----------

@contextmanager
def _db(lock_key: str | None = None):
    with (nullcontext() if storage.is_postgres() else _LOCK), WEB.connect() as db:
        if lock_key:
            db.lock(lock_key)
        yield db


def _limit(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _hash(salt: str, code: str) -> str:
    return hashlib.sha256((salt + code).encode()).hexdigest()


def start_code(email: str, ip: str) -> dict:
    """Email a sign-in code (if the limits allow)."""
    email = email.strip().lower()
    now, ip_key = time.time(), hash_ip(ip)
    code = f"{secrets.randbelow(1_000_000):06d}"
    salt, challenge = secrets.token_hex(8), secrets.token_urlsafe(16)
    with _db(f"signin:{email}") as db:
        recent = db.execute("SELECT COUNT(*) FROM signin_codes WHERE email=? AND created>?",
                            (email, now - 900)).fetchone()[0]
        if recent >= _limit("SIGNIN_CODES_PER_EMAIL", 3):
            raise SignInError("limit", "We've just sent codes to that address. Please use the latest one, "
                                       "or wait 15 minutes.", 429)
        from_ip = db.execute("SELECT COUNT(*) FROM signin_codes WHERE ip_hash=? AND created>?",
                             (ip_key, now - 3600)).fetchone()[0]
        if from_ip >= _limit("SIGNIN_CODES_PER_IP_HOUR", 10):
            raise SignInError("limit", "Too many sign-in codes from your network. Please try again in an hour.", 429)
        today = db.execute("SELECT COUNT(*) FROM signin_codes WHERE created>?", (now - 86400,)).fetchone()[0]
        if today >= _limit("SIGNIN_EMAILS_PER_DAY", 300):
            raise SignInError("limit", "The demo has sent all the sign-in emails it can today. "
                                       "Please use Google, or come back tomorrow.", 429)
        db.execute("UPDATE signin_codes SET expires=? WHERE email=? AND used IS NULL", (now, email))  # one live code
        db.execute("INSERT INTO signin_codes (id, email, ip_hash, salt, code_hash, created, expires)"
                   " VALUES (?,?,?,?,?,?,?)", (challenge, email, ip_key, salt, _hash(salt, code), now, now + CODE_TTL))
    body = (f"Your Demo Bank sign-in code is {code}\n\n"
            f"It expires in {CODE_TTL // 60} minutes.\n\n"
            "If you didn't ask for it, ignore this email: nobody can sign in without the code.\n"
            "Demo Bank is a demonstration. It never holds real money.")
    try:
        send_email(email, "Your Demo Bank sign-in code", body)
    except Exception as e:  # noqa: BLE001
        with _db() as db:
            db.execute("DELETE FROM signin_codes WHERE id=?", (challenge,))
        raise SignInError("email", "The sign-in email couldn't be sent. Please try again, or use Google.", 502) from e
    return {"challenge": challenge, "sent_to": mask(email), "expires_in": CODE_TTL}


def check_code(challenge: str, code: str) -> dict:
    """{"email": ...} when right; else {"error": wrong_code|expired|too_many_attempts, "attempts_left": n}."""
    now = time.time()
    with _db(f"signin-code:{challenge}") as db:
        row = db.execute("SELECT email, salt, code_hash, expires, attempts, used FROM signin_codes WHERE id=?",
                         (challenge,)).fetchone()
        if row is None:
            return {"error": "expired", "attempts_left": 0}
        email, salt, code_hash, expires, attempts, used = row
        if used or expires <= now:
            return {"error": "expired", "attempts_left": 0}
        if attempts >= MAX_ATTEMPTS:
            return {"error": "too_many_attempts", "attempts_left": 0}
        if hmac.compare_digest(_hash(salt, str(code).strip()), code_hash):
            cur = db.execute("UPDATE signin_codes SET used=? WHERE id=? AND used IS NULL", (now, challenge))
            return {"email": email} if cur.rowcount == 1 else {"error": "expired", "attempts_left": 0}
        db.execute("UPDATE signin_codes SET attempts=attempts+1 WHERE id=?", (challenge,))
        left = MAX_ATTEMPTS - attempts - 1
        return {"error": "wrong_code" if left else "too_many_attempts", "attempts_left": left}


def tidy() -> int:
    """Forget codes older than a day (they only matter for the limits)."""
    with _db() as db:
        return db.execute("DELETE FROM signin_codes WHERE created<?", (time.time() - 86400,)).rowcount
