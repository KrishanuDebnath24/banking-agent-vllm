"""Web API for the banking assistant (phase 4).

    python -m api              # http://127.0.0.1:8000   (start  python -m mcp_servers.run  first)

Who signs in
    visitors   anyone with the link: their own pretend bank (obp/guest_bank.py), made when they first
               sign in and deleted after GUEST_TTL_HOURS (24), with their conversations
    the owner  an address in OWNER_EMAILS; then picks Robert or Anil, the OBP sandbox customers
    AUTH_MODE=public   real sign-in (api/signin.py): "Continue with Google", or a code emailed to you.
                       Visitors' security codes then go to the address they signed in with.
    AUTH_MODE=dev      a simple form for trying things on your PC: type any email, or pick A or B.
                       Nothing is proven, so visitors' security codes go to OTP_EMAIL_TO.

Endpoints (JSON, cookie session)
    GET  /healthz
    GET  /api/config                      sign-in options and limits, for the page
    POST /api/login    {customer} | {email, name?}            AUTH_MODE=dev only
    GET  /api/auth/google, /api/auth/google/callback       AUTH_MODE=public
    POST /api/auth/email/start {email}, /api/auth/email/check {challenge, code}
    POST /api/owner/customer {customer}                    the owner picks Robert (A) or Anil (B)
    POST /api/logout
    GET  /api/me
    POST /api/chat     {message, thread_id?}     -> 202 {turn_id}: the answer comes later
    GET  /api/turns/{turn_id}                    -> running (with the steps so far) | answered (with
                                                    any charts, api/charts.py) | needs_confirmation | failed
    POST /api/confirm  {thread_id, approve, code?}  -> 200 code needed / next slip, or 202 {turn_id}
    GET  /api/history?thread_id=...
    GET  /api/conversations                      the person's recent conversations, newest first
    POST /api/guest/reset                        visitors: new balances, conversations cleared

Send, then check back
    An answer takes 5-30 seconds. Rather than holding the request open (gateways cut long requests),
    /api/chat starts the answer and returns at once; the page asks /api/turns/{id} every second and
    shows each step as it happens. Turns, waiting slips and sessions are kept in the database
    (api/state.py), so any copy of the app can answer, and a restart signs nobody out.

Limits for visitors (the owner has none)
    GUEST_DAILY_MESSAGES=30         messages per visitor per 24 hours
    GUEST_SIGNUPS_PER_IP_HOUR=5     new visitor banks per internet address per hour
    DAILY_SPEND_CAP_USD=2.00        model spend by all visitors together per day (UTC), from the cost
                                    ledger; above it visitors are asked to come back tomorrow
    API_RATE_LIMIT_PER_MIN=30       requests per session per minute
    CLIENT_IP_HEADER                header with the visitor's address when behind a proxy: X-Forwarded-For
                                    behind a load balancer (its last entry), CloudFront-Viewer-Address
                                    behind CloudFront; unset: the connection's address

Security
- No bank token reaches the browser. The browser holds a random session id in an HttpOnly cookie and
  the database keeps only its hash. Owner tokens stay in memory; a visitor's is recomputed with APP_SECRET.
- Conversations belong to one person: conversation ids start with the person's key, checked every time.
- Security codes go from the page to the verify service, never into the conversation.
"""

from __future__ import annotations

import asyncio
import hmac
import importlib
import json
import logging
import os
import re
import time
import uuid
from collections import deque
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if os.environ.get("BANK_TEST") != "1":
    load_dotenv(ROOT / ".env", override=True)

import storage  # noqa: E402
from api import charts, signin, state  # noqa: E402
from api.actions import SPECIALISTS, slip  # noqa: E402
from api.state import Pending, Person  # noqa: E402
from mcp_servers import datasets  # noqa: E402
from obp import guest_bank  # noqa: E402
from obp.guest_bank import GUEST_BANK  # noqa: E402
from observability import costs  # noqa: E402

COOKIE = "bank_session"
SESSION_IDLE_SECONDS = 2 * 3600
COOKIE_SECONDS = 24 * 3600
RATE_LIMIT = int(os.environ.get("API_RATE_LIMIT_PER_MIN", "30"))
TURN_TIMEOUT = float(os.environ.get("TURN_TIMEOUT_SECONDS", "180"))
STALE_TURN_SECONDS = TURN_TIMEOUT + 60        # a turn "running" longer than this was on a copy that stopped
AGENT_IDLE_SECONDS = 15 * 60                  # forget an idle person's agent (it's rebuilt on their next message)
OWNER_TOKEN_SECONDS = 6 * 3600
MAINTENANCE_SECONDS = float(os.environ.get("MAINTENANCE_SECONDS", "60"))
LONG_POLL_SECONDS = 20                        # under API Gateway's 30-second limit
DECLINED = "The customer declined this action."
CODE_NOT_CONFIRMED = "The security code was not confirmed, so nothing was done."
COULDNT_FINISH = "I couldn't finish that just now. Please try again."
BANK_DOWN = "The bank service isn't reachable right now. Please try again in a minute."
THREAD_RE = re.compile(r"^(A|B|g[0-9a-f]{10})-[0-9a-f]{8}$")
TURN_RE = re.compile(r"^[0-9a-f]{16}$")
LOOPBACK = {"localhost", "127.0.0.1", "::1", "[::1]"}
EMAIL_RE = re.compile(r"^[^@\s<>\"']{1,64}@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,24}$")
log = logging.getLogger("bank.api")


def limit(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


# ---------- request bodies ----------

class LoginBody(BaseModel):
    customer: str | None = Field(default=None, pattern="^[AB]$")       # the owner's demo customers
    email: str | None = Field(default=None, max_length=254)             # a visitor
    name: str | None = Field(default=None, max_length=60)


class EmailStartBody(BaseModel):
    email: str = Field(max_length=254)


class EmailCheckBody(BaseModel):
    challenge: str = Field(max_length=64)
    code: str = Field(max_length=12)
    name: str | None = Field(default=None, max_length=60)


class CustomerBody(BaseModel):
    customer: str = Field(pattern="^[AB]$")


class ChatBody(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None


class ConfirmBody(BaseModel):
    thread_id: str
    approve: bool
    code: str | None = Field(default=None, max_length=12)


class LimitReached(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind, self.message = kind, message


class BankUnavailable(Exception):
    pass


# ---------- helpers ----------

def _display_name(username: str) -> str:
    return username.split("@")[0].split(".")[0].split("+")[0].capitalize()[:40] or "Visitor"


def _demo_customers() -> dict[str, str]:
    """label -> username, from demo_accounts.json (or the .env logins)."""
    demo_file = Path(os.environ.get("DEMO_ACCOUNTS_FILE", ROOT / "demo_accounts.json"))
    if demo_file.exists():
        return {label: c["username"] for label, c in json.loads(demo_file.read_text())["customers"].items()}
    return {label: os.environ.get(f"CUSTOMER_{label}_USERNAME", f"customer-{label}") for label in ("A", "B")}


def _obp_token(label: str) -> str:
    """Sign in to the bank on the owner's behalf. The browser never sees the token."""
    if os.environ.get("OBP_FAKE") == "1":
        return f"fake-token-{label}"
    from obp.client import OBPClient
    client = OBPClient.from_env()
    return client.login(os.environ[f"CUSTOMER_{label}_USERNAME"], os.environ[f"CUSTOMER_{label}_PASSWORD"])


def _text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content or [])


def client_ip(request: Request) -> str:
    """The visitor's address. Behind CloudFront the connection comes from CloudFront, so the address
    is read from the header it adds (CLIENT_IP_HEADER). Only set that when a proxy really adds it."""
    header = os.environ.get("CLIENT_IP_HEADER", "").strip()
    entries = [v.strip() for v in request.headers.get(header, "").split(",") if v.strip()] if header else []
    value = ""
    if entries and header.lower() == "x-forwarded-for":
        value = entries[-1]                      # added by the load balancer; earlier ones could be made up
    elif entries:
        value = entries[0]
        if header.lower() == "cloudfront-viewer-address":
            value = value.rsplit(":", 1)[0]      # "ip:port"
    return value or (request.client.host if request.client else "unknown")


def waiting(st) -> list[list]:
    """[interrupt id, action] for every write action this conversation is waiting on."""
    return [[i.id, a] for i in st.interrupts for a in (i.value or {}).get("action_requests", [])]


def _meter(meter) -> dict:
    return {"cost": meter.summary_line(), "cost_usd": meter.total_cost, "guard": meter.guard_notes(),
            "latency_ms": meter.latency_ms}


@dataclass
class AgentEntry:
    agent: Any
    last_used: float


# ---------- the app ----------

def create_app(model_factory=None) -> FastAPI:
    """model_factory: returns the chat model for a new agent (tests pass a scripted one)."""
    stack = AsyncExitStack()
    saver_lock = asyncio.Lock()
    rt: dict[str, Any] = {"saver": None, "spend": (0.0, 0.0)}   # spend: (value, when read)
    agents: dict[str, AgentEntry] = {}                           # owner key -> agent (this copy only)
    agent_locks: dict[str, asyncio.Lock] = {}
    owner_tokens: dict[str, tuple[str, float]] = {}
    hits: dict[str, deque] = {}
    tasks: dict[str, asyncio.Task] = {}                          # turn id -> answer running in this copy

    @asynccontextmanager
    async def lifespan(app):
        if os.environ.get("LANGSMITH_TRACING", "").lower() == "true" and os.environ.get("LANGSMITH_API_KEY"):
            from observability.tracing import install_trace_redaction
            install_trace_redaction()
        # The agent code takes a couple of seconds to load. Load it now, so the first message doesn't
        # hold up everyone else while it does.
        await asyncio.to_thread(importlib.import_module, "agents.coordinator")
        loop = asyncio.create_task(maintenance_loop())
        yield
        loop.cancel()
        running = list(tasks.values())
        for task in running:   # stopping: answers in progress are marked as interrupted
            task.cancel()
        await asyncio.gather(loop, *running, return_exceptions=True)
        await stack.aclose()

    app = FastAPI(title="Banking assistant API", lifespan=lifespan, docs_url=None, redoc_url=None)

    @app.exception_handler(LimitReached)
    async def limit_reached(request: Request, e: LimitReached):
        return JSONResponse({"detail": e.message, "limit": e.kind}, status_code=429)

    @app.middleware("http")
    async def only_through_cloudfront(request: Request, call_next):
        """On AWS, CloudFront adds a secret header (ORIGIN_SECRET) to every request it passes on, so
        requests sent straight to the API Gateway address, around CloudFront, are turned away."""
        secret = os.environ.get("ORIGIN_SECRET", "")
        if secret and request.url.path != "/healthz" and not hmac.compare_digest(
                request.headers.get("x-origin-verify", "").encode(), secret.encode()):
            return JSONResponse({"detail": "Not found."}, status_code=404)
        return await call_next(request)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api") else "no-cache"
        return response

    # ---------- conversation store, agents, tokens ----------

    async def saver():
        async with saver_lock:
            if rt["saver"] is None:
                rt["saver"] = await stack.enter_async_context(storage.checkpointer())
            return rt["saver"]

    async def token_of(p: Person) -> str:
        if p.is_guest:
            return guest_bank.token_for(p.subject)
        hit = owner_tokens.get(p.subject)
        if hit and time.time() - hit[1] < OWNER_TOKEN_SECONDS:
            return hit[0]
        token = await asyncio.to_thread(_obp_token, p.subject)
        owner_tokens[p.subject] = (token, time.time())
        return token

    async def agent_for(p: Person):
        entry = agents.get(p.owner_key)
        if entry is None:
            async with agent_locks.setdefault(p.owner_key, asyncio.Lock()):
                entry = agents.get(p.owner_key)
                if entry is None:
                    from agents.coordinator import build_coordinator
                    try:
                        agent = await build_coordinator(await token_of(p), await saver(),
                                                        model=model_factory() if model_factory else None)
                    except Exception as e:  # noqa: BLE001
                        # The reason goes to the server log (docker compose logs api / CloudWatch).
                        log.exception("can't reach the bank servers (%s): %s", type(e).__name__,
                                      ", ".join(os.environ.get(k, "default") for k in
                                                ("MCP_ACCOUNTS_URL", "MCP_TRANSACTIONS_URL", "MCP_SERVICE_URL", "MCP_CHARTS_URL")))
                        raise BankUnavailable() from e
                    entry = agents[p.owner_key] = AgentEntry(agent, time.time())
        entry.last_used = time.time()
        return entry.agent

    # ---------- who is asking ----------

    async def person(request: Request, count: bool = False, need_customer: bool = True) -> Person:
        sid = request.cookies.get(COOKIE, "")
        p = await asyncio.to_thread(state.get_session, sid, SESSION_IDLE_SECONDS) if sid else None
        if p is None:
            raise HTTPException(401, "Please sign in.")
        if need_customer and not p.subject:   # the owner, signed in, hasn't picked Robert or Anil yet
            raise HTTPException(409, "Choose a demo customer first.")
        if count:  # progress checks don't count: the page makes one a second while an answer runs
            now = time.time()
            q = hits.setdefault(p.sid_hash, deque())
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= RATE_LIMIT:
                raise HTTPException(429, "Too many requests. Please wait a moment.")
            q.append(now)
        return p

    def own_thread(p: Person, thread_id: str | None, create: bool = False) -> str:
        if not thread_id:
            if not create:
                raise HTTPException(400, "thread_id is required.")
            return f"{p.owner_key}-{uuid.uuid4().hex[:8]}"
        if not THREAD_RE.match(thread_id) or not thread_id.startswith(f"{p.owner_key}-"):
            raise HTTPException(404, "Conversation not found.")
        return thread_id

    async def visitor_spend_today() -> float:
        if os.environ.get("TELEMETRY_DB", "") == "off":
            return 0.0
        value, when = rt["spend"]
        if time.time() - when > 30:
            day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            value = await asyncio.to_thread(state.visitor_spend_since, day)
            rt["spend"] = (value, time.time())
        return value

    async def messages_left(p: Person) -> int:
        used = await asyncio.to_thread(state.messages_since, p.owner_key, time.time() - 86400)
        return max(0, int(limit("GUEST_DAILY_MESSAGES", "30")) - used)

    async def check_visitor_limits(p: Person) -> None:
        if not p.is_guest:
            return
        if await messages_left(p) <= 0:
            raise LimitReached("messages", f"You've reached the demo's limit of "
                                           f"{int(limit('GUEST_DAILY_MESSAGES', '30'))} messages a day. "
                                           "Please come back tomorrow.")
        if await visitor_spend_today() >= limit("DAILY_SPEND_CAP_USD", "2.00"):
            raise LimitReached("budget", "The demo has used today's budget for visitors. Please come back tomorrow.")

    # ---------- answering, in the background ----------

    async def stream(agent, payload, config, steps: list, turn_id: str, drawn: list) -> None:
        """Run the agent, saving each specialist and tool it uses as it happens (the page shows them),
        and keeping a chart for each bank result that makes one (api/charts.py)."""
        seen: set = set()
        specialist = None
        async for _ns, chunk in agent.astream(payload, config, stream_mode="updates", subgraphs=True):
            added = False
            for node, data in chunk.items():
                if node == "__interrupt__" or not isinstance(data, dict):
                    continue
                msgs = data.get("messages", [])
                for m in msgs if isinstance(msgs, list) else []:
                    if getattr(m, "type", "") == "tool" and getattr(m, "name", "") in charts.CHART_TOOLS:
                        key = ("result", getattr(m, "tool_call_id", None))
                        if key not in seen:
                            seen.add(key)
                            chart = charts.from_tool(m.name, _text(m))
                            if chart:
                                drawn.append(chart)
                    for call in getattr(m, "tool_calls", None) or []:
                        key = (call.get("id"), call.get("name"), repr(call.get("args")))
                        if key in seen:
                            continue
                        seen.add(key)
                        added = True
                        if call["name"] == "task":
                            specialist = call["args"].get("subagent_type")
                            steps.append({"kind": "specialist", "name": specialist,
                                          "label": SPECIALISTS.get(specialist, specialist)})
                        else:
                            steps.append({"kind": "tool", "name": call["name"], "args": call.get("args", {}),
                                          "specialist": specialist})
            if added:
                await asyncio.to_thread(state.save_steps, turn_id, steps)

    async def run_turn(turn_id: str, p: Person, thread_id: str, payload, new_message: bool) -> None:
        meter = costs.start_turn(thread_id, p.owner_key)
        steps: list = []
        drawn: list = []
        status, result, pending = "failed", {"error": COULDNT_FINISH}, None
        config = {"configurable": {"thread_id": thread_id}}
        try:
            agent = await agent_for(p)
            if new_message:
                items = waiting(await agent.aget_state(config))
                if items:  # a slip from an answer that never finished saving (the app stopped mid-way)
                    pending = Pending(items=items)
                    status, result = "needs_confirmation", {"slip": slip(items[0][1])}
                    return
            run_config = {**config, "callbacks": [costs.CostCallback(meter)], "run_name": "banking_web_turn",
                          "tags": ["web", "visitor" if p.is_guest else f"customer-{p.subject}"],
                          "metadata": {"customer": p.owner_key, "thread_id": thread_id}}
            await asyncio.wait_for(stream(agent, payload, run_config, steps, turn_id, drawn), TURN_TIMEOUT)
            items = waiting(st := await agent.aget_state(config))
            if items:
                pending = Pending(items=items)
                status, result = "needs_confirmation", {"slip": slip(items[0][1])}
            else:
                status, result = "answered", {"answer": _text(st.values["messages"][-1])}
                if drawn:
                    result["charts"] = drawn[-charts.MAX_CHARTS:]
        except BankUnavailable:
            result = {"error": BANK_DOWN}
        except TimeoutError:
            log.warning("turn timed out after %ss (thread %s)", TURN_TIMEOUT, thread_id)
            result = {"error": "That took too long, so I stopped. Please try again."}
        except asyncio.CancelledError:
            result = {"error": state.STOPPED}
        except Exception as e:  # noqa: BLE001 - model or bank failure mid-turn
            # The customer gets a short apology; the reason goes to the server log (CloudWatch on AWS).
            log.exception("turn failed (thread %s): %s", thread_id, type(e).__name__)
        finally:
            try:
                await asyncio.to_thread(costs.end_turn, meter)  # the ledger write mustn't hold up anyone else
            except Exception:  # noqa: BLE001
                log.exception("couldn't save the turn's cost")
            result.update(_meter(meter))
            try:
                await asyncio.to_thread(state.finish_turn, turn_id, thread_id, p.owner_key, status, steps, result,
                                        pending)
            except Exception:  # noqa: BLE001 - housekeeping marks it failed later
                log.exception("couldn't save the answer (thread %s)", thread_id)
            tasks.pop(turn_id, None)

    def launch(turn_id: str, p: Person, thread_id: str, payload, new_message: bool) -> None:
        # Named after the person, so housekeeping keeps their agent while it's in use.
        tasks[turn_id] = asyncio.create_task(run_turn(turn_id, p, thread_id, payload, new_message),
                                             name=p.owner_key)

    async def verify(p: Person, path: str, body: dict) -> dict:
        url = os.environ.get("VERIFY_URL", "http://127.0.0.1:8104")
        try:
            token = await token_of(p)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, "The bank didn't accept the sign-in. Please sign in again.") from e
        try:
            async with httpx.AsyncClient(base_url=url, timeout=30, headers={
                    "Authorization": f'DirectLogin token="{token}"'}) as http:
                r = await http.post(path, json=body)
                data = r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise HTTPException(503, "The security code service isn't reachable.") from e
        if r.status_code != 200:
            raise HTTPException(r.status_code if r.status_code in (400, 401, 429) else 502,
                                data.get("message", "The security code couldn't be sent."))
        return data

    # ---------- housekeeping ----------

    async def forget_conversations(owner_key: str) -> None:
        cp = await saver()
        for thread_id in await asyncio.to_thread(state.threads_of, owner_key):
            await cp.adelete_thread(thread_id)
        await asyncio.to_thread(state.forget_conversations, owner_key)

    async def forget_visitor(guest_id: str) -> None:
        """An expired visitor: their conversations, sessions, requests and bank."""
        owner_key = guest_bank.owner_key_for(guest_id)
        await forget_conversations(owner_key)
        await asyncio.to_thread(state.end_sessions_of, owner_key)
        await asyncio.to_thread(state.forget_requests, guest_id)
        await asyncio.to_thread(datasets.forget, f"guest:{guest_id}")   # figures saved for charts
        await asyncio.to_thread(GUEST_BANK.delete, [guest_id])
        agents.pop(owner_key, None)

    async def maintenance_once() -> dict:
        gone = await asyncio.to_thread(GUEST_BANK.expired)
        for guest_id in gone:
            await forget_visitor(guest_id)
        done = await asyncio.to_thread(state.tidy, SESSION_IDLE_SECONDS, STALE_TURN_SECONDS)
        done["old_signin_codes"] = await asyncio.to_thread(signin.tidy)
        done["old_security_codes"] = await asyncio.to_thread(state.forget_old_codes)
        done["old_chart_data"] = await asyncio.to_thread(datasets.tidy)
        now = time.time()
        busy = {t.get_name() for t in tasks.values()}
        for key in [k for k, e in agents.items() if now - e.last_used > AGENT_IDLE_SECONDS and k not in busy]:
            agents.pop(key, None)
            agent_locks.pop(key, None)
        for sid_hash in [k for k, q in hits.items() if not q or now - q[-1] > 60]:
            hits.pop(sid_hash, None)
        return {"expired_visitors": len(gone), **done, "agents": len(agents)}

    async def maintenance_loop() -> None:
        while True:
            await asyncio.sleep(MAINTENANCE_SECONDS)
            try:
                done = await maintenance_once()
                if any(v for k, v in done.items() if k != "agents"):
                    log.info("housekeeping: %s", done)
            except Exception:  # noqa: BLE001 - try again next time
                log.exception("housekeeping failed")

    app.state.maintenance_once = maintenance_once   # tests call it directly
    app.state.agents = agents

    # ---------- routes ----------

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    def customers() -> list[dict]:
        return [{"customer": label, "name": _display_name(u)} for label, u in sorted(_demo_customers().items())]

    @app.get("/api/config")
    async def config():
        mode = signin.auth_mode()
        return {"auth_mode": mode, "customers": customers() if mode == "dev" else [],
                "signin": {"google": signin.google_enabled(), "email": True,
                           "invite_only": not signin.open_to_public()} if mode == "public" else None,
                "visitors": {"bank_hours": round(guest_bank.ttl_seconds() / 3600),
                             "daily_messages": int(limit("GUEST_DAILY_MESSAGES", "30"))}}

    def set_cookie(request: Request, response: Response, sid: str) -> None:
        response.set_cookie(COOKIE, sid, httponly=True, samesite="strict", secure=secure_cookies(request),
                            max_age=COOKIE_SECONDS, path="/")

    def secure_cookies(request: Request) -> bool:
        return (os.environ.get("COOKIE_SECURE", "0") == "1" or request.url.scheme == "https"
                or signin.public_url().startswith("https://"))

    async def owner_token(label: str) -> None:
        try:
            owner_tokens[label] = (await asyncio.to_thread(_obp_token, label), time.time())
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, "The bank didn't accept the demo sign-in.") from e

    async def visitor_bank(request: Request, email: str, name: str, verified: bool):
        """The visitor's own pretend bank: their existing one, or a new one if the limits allow."""
        guest = await asyncio.to_thread(GUEST_BANK.active, email)
        if guest is not None and (guest.verified or not verified):
            return guest
        if guest is None:
            ip = client_ip(request)
            if await asyncio.to_thread(GUEST_BANK.signups_from, ip) >= limit("GUEST_SIGNUPS_PER_IP_HOUR", "5"):
                raise LimitReached("signups", "Too many new demo banks from your network in the last hour. "
                                              "Please try again later.")
        else:
            ip = None
        guest, _ = await asyncio.to_thread(GUEST_BANK.open, email, name or _display_name(email), ip, verified)
        return guest

    async def sign_in(request: Request, response: Response, email: str, name: str, verified: bool) -> dict:
        """Start a session for someone whose email address is known (and, with real sign-in, proven)."""
        if verified and signin.is_owner(email):   # the owner: picks Robert or Anil next
            name = " ".join(name.split())[:40] or _display_name(email)
            sid = await asyncio.to_thread(state.new_session, "owner", "", "", email, name)
            set_cookie(request, response, sid)
            return {"kind": "owner", "name": name, "choosing": True}
        guest = await visitor_bank(request, email, " ".join(name.split())[:40], verified)
        sid = await asyncio.to_thread(state.new_session, "guest", guest.guest_id, guest.owner_key, email, guest.name)
        set_cookie(request, response, sid)
        return {"customer": guest.owner_key, "kind": "guest", "name": guest.name}

    @app.post("/api/login")
    async def login(body: LoginBody, request: Request, response: Response):
        """The development sign-in: no proof of who you are, so it's off on the public site."""
        if signin.auth_mode() != "dev":
            raise HTTPException(404, "This sign-in is off.")
        if body.customer:  # the owner's demo customers in the OBP sandbox
            username = _demo_customers().get(body.customer)
            if not username:
                raise HTTPException(404, "Unknown demo customer.")
            await owner_token(body.customer)
            name = _display_name(username)
            sid = await asyncio.to_thread(state.new_session, "owner", body.customer, body.customer, None, name)
            set_cookie(request, response, sid)
            return {"customer": body.customer, "kind": "owner", "name": name}
        email = (body.email or "").strip().lower()
        if not EMAIL_RE.match(email):
            raise HTTPException(400, "Enter a valid email address.")
        return await sign_in(request, response, email, body.name or "", verified=False)

    # ---- real sign-in (AUTH_MODE=public) ----

    def public_only() -> None:
        if signin.auth_mode() != "public":
            raise HTTPException(404, "Not found.")

    def back_to_page(error: str | None = None) -> RedirectResponse:
        return RedirectResponse("/" + (f"?signin_error={error}" if error else ""), status_code=303)

    @app.get("/api/auth/google")
    async def google(request: Request):
        public_only()
        if not signin.google_enabled():
            raise HTTPException(404, "Google sign-in isn't set up.")
        # On your PC: opened as localhost but PUBLIC_URL says 127.0.0.1 (or the other way round)? Start at
        # PUBLIC_URL, so the cookie set here is still there when Google sends you back. Only for these local
        # names: on AWS the app sees API Gateway's address, never the public one, and must not redirect.
        public = urlsplit(signin.public_url())
        here = request.url
        if (here.hostname in LOOPBACK and public.hostname in LOOPBACK
                and (here.hostname, here.port) != (public.hostname, public.port)):
            return RedirectResponse(f"{signin.public_url()}/api/auth/google", status_code=303)
        url, cookie = signin.google_start()
        response = RedirectResponse(url, status_code=303)
        response.set_cookie(signin.OAUTH_COOKIE, cookie, httponly=True, samesite="lax", path="/api/auth",
                            secure=secure_cookies(request), max_age=signin.OAUTH_SECONDS)
        return response

    @app.get("/api/auth/google/callback")
    async def google_callback(request: Request, code: str = "", error: str = "",
                              oauth_state: str = Query("", alias="state")):
        public_only()
        if error:   # e.g. they pressed Cancel at Google
            response = back_to_page("cancelled")
        else:
            try:
                who = await signin.google_finish(code, oauth_state, request.cookies.get(signin.OAUTH_COOKIE, ""),
                                                 transport=getattr(app.state, "google_transport", None))
                if not signin.may_sign_in(who["email"]):
                    raise signin.SignInError("invite", "This demo is invite-only for now.", 403)
                response = back_to_page()
                await sign_in(request, response, who["email"], who["name"], verified=True)
            except signin.SignInError as e:
                log.warning("Google sign-in refused: %s", e.kind)
                response = back_to_page(e.kind)
            except LimitReached as e:
                response = back_to_page(e.kind)
        response.delete_cookie(signin.OAUTH_COOKIE, path="/api/auth")
        return response

    @app.post("/api/auth/email/start")
    async def email_start(body: EmailStartBody, request: Request):
        public_only()
        email = body.email.strip().lower()
        if not EMAIL_RE.match(email):
            raise HTTPException(400, "Enter a valid email address.")
        if not signin.may_sign_in(email):   # invite-only: don't email codes to anyone else
            return JSONResponse({"detail": "The demo is invite-only for now, and that address isn't on the list.",
                                 "error": "invite"}, status_code=403)
        try:
            return await asyncio.to_thread(signin.start_code, email, client_ip(request))
        except signin.SignInError as e:
            return JSONResponse({"detail": e.message, "error": e.kind}, status_code=e.status)

    @app.post("/api/auth/email/check")
    async def email_check(body: EmailCheckBody, request: Request, response: Response):
        public_only()
        checked = await asyncio.to_thread(signin.check_code, body.challenge, body.code)
        if "email" not in checked:
            return JSONResponse(checked, status_code=400)
        if not signin.may_sign_in(checked["email"]):   # the invite list changed since the code was sent
            raise HTTPException(403, "The demo is invite-only for now, and that address isn't on the list.")
        return await sign_in(request, response, checked["email"], body.name or "", verified=True)

    @app.post("/api/owner/customer")
    async def owner_customer(body: CustomerBody, request: Request):
        """The owner picks (or switches between) Robert and Anil."""
        p = await person(request, count=True, need_customer=False)
        if p.kind != "owner":
            raise HTTPException(404, "Not found.")
        username = _demo_customers().get(body.customer)
        if not username:
            raise HTTPException(404, "Unknown demo customer.")
        await owner_token(body.customer)
        await asyncio.to_thread(state.choose_customer, request.cookies.get(COOKIE, ""), body.customer)
        return {"customer": body.customer, "kind": "owner", "name": p.name}

    @app.post("/api/logout")
    async def logout(request: Request, response: Response):
        sid = request.cookies.get(COOKIE, "")
        if sid:
            await asyncio.to_thread(state.end_session, sid)
        response.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    @app.get("/api/me")
    async def me(request: Request):
        p = await person(request, need_customer=False)
        out: dict[str, Any] = {"customer": p.owner_key, "kind": p.kind, "name": p.name}
        if p.kind == "owner":   # Robert and Anil, to pick from or switch between
            out.update(choosing=not p.subject, customers=customers())
        if p.is_guest:
            guest = await asyncio.to_thread(GUEST_BANK.get, p.subject)
            if guest is None:   # expired: housekeeping will tidy up the rest
                await asyncio.to_thread(state.end_session, request.cookies.get(COOKIE, ""))
                raise HTTPException(401, "Your demo bank has expired. Sign in again for a new one.")
            out.update(email=p.email, expires=guest.expires, verified=guest.verified,
                       messages_left=await messages_left(p))
        return out

    @app.post("/api/chat", status_code=202)
    async def chat(body: ChatBody, request: Request):
        p = await person(request, count=True)
        thread_id = own_thread(p, body.thread_id, create=True)
        await check_visitor_limits(p)
        turn_id, busy = await asyncio.to_thread(state.start_chat, thread_id, p.owner_key)
        if busy is not None:
            if busy.status == "running":
                return JSONResponse({"detail": "Still working on your last message.", "thread_id": thread_id,
                                     "status": "running", "turn_id": busy.turn_id}, status_code=409)
            return JSONResponse({"detail": "Please approve or decline the waiting request first.",
                                 "thread_id": thread_id, "status": "needs_confirmation",
                                 "slip": slip(busy.pending.current or {})}, status_code=409)
        await asyncio.to_thread(state.note_thread, thread_id, p.owner_key, body.message)
        launch(turn_id, p, thread_id, {"messages": [{"role": "user", "content": body.message}]}, True)
        return {"thread_id": thread_id, "turn_id": turn_id, "status": "running"}

    @app.get("/api/turns/{turn_id}")
    async def turn(turn_id: str, request: Request, wait: float = 0, seen: int = 0):
        """wait: hold the request up to this many seconds (max 20) until the answer is ready or there
        are more than `seen` steps. Far fewer requests than asking every second (each one is paid for)."""
        p = await person(request)
        deadline = time.monotonic() + max(0.0, min(wait, LONG_POLL_SECONDS))
        while True:
            found = await asyncio.to_thread(state.get_turn, turn_id) if TURN_RE.match(turn_id) else None
            if found is None or found["owner_key"] != p.owner_key:
                raise HTTPException(404, "Not found.")
            if (found["status"] != "running" or len(found["steps"]) > seen or time.monotonic() >= deadline
                    or await request.is_disconnected()):
                return {"turn_id": turn_id, "thread_id": found["thread_id"], "status": found["status"],
                        "steps": found["steps"], **found["result"]}
            await asyncio.sleep(0.5)

    def code_box(p: Pending) -> dict | None:
        """The code entry box, if a code was sent for the current action and hasn't expired."""
        ch = p.challenge or {}
        left = int(ch.get("expires", 0) - time.time())
        return {"sent_to": ch.get("sent_to"), "expires_in": left} if ch and left > 0 else None

    @app.post("/api/confirm")
    async def confirm(body: ConfirmBody, request: Request):
        p = await person(request, count=True)
        thread_id = own_thread(p, body.thread_id)
        if await asyncio.to_thread(state.running_turn, thread_id):
            raise HTTPException(409, "Still working on your last request.")
        pending = await asyncio.to_thread(state.get_pending, thread_id)
        if pending is None or pending.current is None:
            raise HTTPException(409, "There's nothing waiting for confirmation.")
        action = pending.current
        changed = HTTPException(409, "This was just answered in another window. Reload the page.")
        typed = (body.code or "").strip()
        if not body.approve:
            decision = {"type": "reject", "message": DECLINED}
        elif pending.challenge is None or (not typed and code_box(pending) is None):  # no code yet, or it expired
            started = await verify(p, "/verify/start", {"tool": action.get("name"), "args": action.get("args", {})})
            if started.get("required"):
                challenge = {"id": started["challenge_id"], "sent_to": started["sent_to"],
                             "expires": time.time() + started["expires_in"]}
                if not await asyncio.to_thread(state.update_pending, thread_id, p.owner_key, pending,
                                               pending.decisions, challenge):
                    raise changed
                return {"thread_id": thread_id, "status": "code_required", "slip": slip(action),
                        "sent_to": started["sent_to"], "expires_in": started["expires_in"]}
            decision = {"type": "approve"}
        elif not typed:   # e.g. the page was reloaded after the code was sent: show the box again
            return {"thread_id": thread_id, "status": "code_required", "slip": slip(action), **code_box(pending)}
        else:
            checked = await verify(p, "/verify/check", {"challenge_id": pending.challenge_id, "code": body.code})
            if checked.get("verified"):
                decision = {"type": "approve"}
            elif checked.get("attempts_left", 0) > 0:
                return {"thread_id": thread_id, "status": "wrong_code", "slip": slip(action),
                        "attempts_left": checked["attempts_left"]}
            else:
                decision = {"type": "reject", "message": CODE_NOT_CONFIRMED}
        decisions = pending.decisions + [decision]
        if len(decisions) < len(pending.items):  # more actions in the same step
            if not await asyncio.to_thread(state.update_pending, thread_id, p.owner_key, pending, decisions, None):
                raise changed
            return {"thread_id": thread_id, "status": "needs_confirmation",
                    "slip": slip(pending.items[len(decisions)][1])}
        turn_id = await asyncio.to_thread(state.start_resume, thread_id, p.owner_key, pending, decisions)
        if turn_id is None:
            raise changed
        from langgraph.types import Command
        launch(turn_id, p, thread_id, Command(resume=pending.resume_value(decisions)), False)
        return JSONResponse({"thread_id": thread_id, "turn_id": turn_id, "status": "running",
                             "decision": decision["type"]}, status_code=202)

    @app.get("/api/history")
    async def history(thread_id: str, request: Request):
        p = await person(request, count=True)
        thread_id = own_thread(p, thread_id)
        try:
            agent = await agent_for(p)
        except BankUnavailable as e:
            raise HTTPException(503, BANK_DOWN) from e
        st = await agent.aget_state({"configurable": {"thread_id": thread_id}})
        drawn = await asyncio.to_thread(state.charts_of, thread_id)
        messages = []
        for m in (st.values or {}).get("messages", []):
            kind = getattr(m, "type", "")
            if kind == "human" or (kind == "ai" and not getattr(m, "tool_calls", None)):
                text = _text(m)
                if text.strip():
                    messages.append({"role": "you" if kind == "human" else "bank", "text": text})
                    if kind == "ai" and text in drawn:
                        messages[-1]["charts"] = drawn[text]
        pending = await asyncio.to_thread(state.get_pending, thread_id)
        return {"thread_id": thread_id, "messages": messages,
                "slip": slip(pending.current) if pending and pending.current else None,
                "code": code_box(pending) if pending and pending.current else None,
                "running_turn": await asyncio.to_thread(state.running_turn, thread_id)}

    @app.get("/api/conversations")
    async def conversations(request: Request):
        """The person's recent conversations, newest first, for the list beside the chat: a title (the
        first message, cleaned of card numbers, codes and contact details), when it was last used,
        and whether it's waiting for an approval or still answering."""
        p = await person(request, count=True)
        items = await asyncio.to_thread(state.conversations_of, p.owner_key)
        untitled = [c for c in items if c["title"] is None]
        if untitled:   # from before the list existed: the title comes from the conversation itself
            try:
                agent = await agent_for(p)
                for c in untitled:
                    st = await agent.aget_state({"configurable": {"thread_id": c["thread_id"]}})
                    first = next((_text(m) for m in (st.values or {}).get("messages", [])
                                  if getattr(m, "type", "") == "human" and _text(m).strip()), None)
                    if first:
                        c["title"] = await asyncio.to_thread(state.set_title, c["thread_id"], p.owner_key,
                                                             first, c["updated"])
            except BankUnavailable:
                pass   # listed next time
        return {"conversations": [{k: c[k] for k in ("thread_id", "title", "updated", "waiting", "running")}
                                  for c in items if c["title"]]}

    @app.post("/api/guest/reset")
    async def guest_reset(request: Request):
        p = await person(request, count=True)
        if not p.is_guest:
            raise HTTPException(404, "Only visitors' demo banks can be reset.")
        if await asyncio.to_thread(state.busy_people, [p.owner_key]):
            raise HTTPException(409, "Please wait for the current answer to finish.")
        try:
            guest = await asyncio.to_thread(GUEST_BANK.reset, p.subject)
        except Exception as e:  # noqa: BLE001 - expired
            raise HTTPException(401, "Your demo bank has expired. Sign in again for a new one.") from e
        await forget_conversations(p.owner_key)
        await asyncio.to_thread(state.forget_requests, p.subject)   # they were about the old bank
        await asyncio.to_thread(datasets.forget, f"guest:{p.subject}")
        return {"ok": True, "expires": guest.expires}

    @app.get("/privacy")
    @app.get("/privacy.html")
    async def privacy():
        """The privacy notice (Google requires one before "Continue with Google" can be public). On AWS
        CloudFront serves it too, even outside opening hours (infra/stacks/opening_hours.py)."""
        return HTMLResponse(privacy_page())

    web = ROOT / "web"
    if web.exists():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app


def contact_email() -> str:
    """Who visitors can write to: CONTACT_EMAIL, else the first OWNER_EMAILS address."""
    owners = [e.strip() for e in os.environ.get("OWNER_EMAILS", "").split(",") if e.strip()]
    return os.environ.get("CONTACT_EMAIL", "").strip() or (owners[0] if owners else "the owner of this demo")


def privacy_page(contact: str | None = None) -> str:
    return (ROOT / "web" / "privacy.html").read_text(encoding="utf-8").replace("__CONTACT__", contact or contact_email())


app = create_app()
