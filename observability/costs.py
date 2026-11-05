"""Cost tracker: prices every model call and guardrail check, per turn, and keeps a ledger.

    meter = start_turn(thread_id, customer)          # before the agent runs
    config["callbacks"] = [CostCallback(meter)]       # sees every model call, subagents included
    ...agent runs; guardrails add their checks to the same meter...
    end_turn(meter)                                   # saves to the ledger
    print(meter.summary_line())

The ledger is the bank's own record, independent of LangSmith, so cost reports keep working
if tracing is off. It lives in Postgres when DATABASE_URL is set (Docker, AWS), else in the
SQLite file TELEMETRY_DB. TELEMETRY_DB=off turns it off.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from storage import LEDGER

ROOT = Path(__file__).resolve().parent.parent

# USD per million tokens. Matched against the model name the provider reports, so the same
# entry covers the Anthropic API ("claude-haiku-5-5") and Bedrock ("eu.anthropic.claude-haiku-5-5...").
# Sources: platform.claude.com/docs/en/about-claude/pricing (October 2026). "long" applies when one
# prompt exceeds `long_after` input tokens.
PRICES: list[tuple[str, dict]] = [
    ("haiku-5-5", {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125,
                   "long_after": 100_000,
                   "long": {"input": 0.50, "output": 2.50, "cache_read": 0.05, "cache_write": 0.625}}),
    ("sonnet-5-5", {"input": 2.00, "output": 10.00, "cache_read": 0.10, "cache_write": 2.50}),
]


def price_for(model: str) -> dict | None:
    name = (model or "").lower()
    extra = os.environ.get("MODEL_PRICES_JSON")  # optional override: {"qwen3-32b": {"input": .., "output": ..}}
    table = list(json.loads(extra).items()) if extra else []
    for pattern, price in table + PRICES:
        if pattern in name:
            return price
    return None


def llm_cost(model: str, usage: dict) -> float | None:
    """Cost of one call from LangChain usage_metadata. None if the model has no known price."""
    price = price_for(model)
    if price is None:
        return None
    details = usage.get("input_token_details") or {}
    cache_read = int(details.get("cache_read") or 0)
    cache_write = int(details.get("cache_creation") or 0)
    total_in = int(usage.get("input_tokens") or 0)
    if price.get("long_after") and total_in > price["long_after"]:
        price = price["long"]
    plain_in = max(total_in - cache_read - cache_write, 0)  # LangChain's input_tokens includes cache tokens
    out = int(usage.get("output_tokens") or 0)
    # A self-hosted price (MODEL_PRICES_JSON) may give only input and output: cached tokens then
    # cost the same as any other input token.
    read_price, write_price = price.get("cache_read", price["input"]), price.get("cache_write", price["input"])
    return round((plain_in * price["input"] + cache_read * read_price
                  + cache_write * write_price + out * price["output"]) / 1_000_000, 8)


def usd(amount: float) -> str:
    """Haiku turns cost fractions of a cent, so show enough digits to see them."""
    return f"${amount:.5f}" if abs(amount) < 0.01 else f"${amount:.4f}"


@dataclass
class TurnMeter:
    thread_id: str
    customer: str
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started: float = field(default_factory=time.perf_counter)
    llm_calls: list[dict] = field(default_factory=list)
    guard_checks: list[dict] = field(default_factory=list)
    finished: float | None = None

    def add_llm(self, model: str, agent: str, usage: dict) -> None:
        details = usage.get("input_token_details") or {}
        self.llm_calls.append({
            "model": model, "agent": agent,
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cache_read": int(details.get("cache_read") or 0),
            "cache_write": int(details.get("cache_creation") or 0),
            "cost_usd": llm_cost(model, usage)})

    def add_guard(self, layer: str, source: str, mode: str, findings: list[str], applied: bool,
                  cost_usd: float = 0.0, units: dict | None = None, latency_ms: int = 0,
                  error: str | None = None) -> None:
        self.guard_checks.append({"layer": layer, "source": source, "mode": mode, "findings": findings,
                                  "applied": applied, "cost_usd": cost_usd, "units": units or {},
                                  "latency_ms": latency_ms, "error": error})

    # ---- totals ----
    @property
    def tokens(self) -> int:
        return sum(c["input_tokens"] + c["output_tokens"] for c in self.llm_calls)

    @property
    def llm_cost(self) -> float:
        return round(sum(c["cost_usd"] or 0 for c in self.llm_calls), 8)

    @property
    def guard_cost(self) -> float:
        return round(sum(g["cost_usd"] or 0 for g in self.guard_checks), 8)

    @property
    def total_cost(self) -> float:
        return round(self.llm_cost + self.guard_cost, 8)

    @property
    def unpriced(self) -> list[str]:
        return sorted({c["model"] for c in self.llm_calls if c["cost_usd"] is None})

    @property
    def latency_ms(self) -> int:
        end = self.finished or time.perf_counter()
        return int((end - self.started) * 1000)

    def guard_notes(self) -> list[str]:
        """Human-readable guardrail findings for this turn, e.g. 'input masked (rules): email'."""
        notes = []
        for g in self.guard_checks:
            if not (g["findings"] or g["error"]):
                continue
            what = ", ".join(g["findings"]) or f"error {g['error']}"
            where = g["source"].lower()
            if g["layer"] == "rules":
                verb = "blocked" if set(g["findings"]) & _BLOCK_KINDS else "masked"
                notes.append(f"{where} {verb} (rules): {what}")
            elif g["layer"] == "limits":
                notes.append("stopped: too many steps in one turn (call limit); the customer was asked to try again")
            elif g["layer"] == "tidy":
                notes.append(f"answer tidied ({what})")
            elif g["layer"] == "format":
                then = "gave up and said sorry" if g["mode"] == "fallback" else "asked again"
                notes.append(f"model reply was broken ({what}): {then}")
            elif g["error"] and not g["findings"]:
                notes.append(f"{where} not checked (Bedrock unavailable, {g['mode']}): {g['error']}")
            elif all(f.startswith(("grounding", "relevance")) for f in g["findings"]):
                notes.append(f"{where} flagged (Bedrock): {what}")
            elif g["applied"]:
                notes.append(f"{where} blocked (Bedrock): {what}")
            else:
                notes.append(f"{where} would be blocked (Bedrock, {g['mode']} mode): {what}")
        return notes

    def summary_line(self) -> str:
        line = (f"{usd(self.total_cost)} · {len(self.llm_calls)} model calls · {self.tokens / 1000:.1f}k tokens"
                f" · {self.latency_ms / 1000:.1f}s")
        if self.guard_cost:
            line += f" (guardrails {usd(self.guard_cost)})"
        if self.unpriced:
            line += f" · no price for {', '.join(self.unpriced)}"
        return line


_BLOCK_KINDS = {"card_number", "cvv", "pin", "password", "security_code", "account_id"}
CURRENT: ContextVar[TurnMeter | None] = ContextVar("current_turn_meter", default=None)


def current() -> TurnMeter | None:
    return CURRENT.get()


def start_turn(thread_id: str, customer: str) -> TurnMeter:
    meter = TurnMeter(thread_id=thread_id, customer=customer)
    CURRENT.set(meter)
    return meter


def end_turn(meter: TurnMeter, save: bool = True) -> TurnMeter:
    meter.finished = time.perf_counter()
    if save and os.environ.get("TELEMETRY_DB", "") != "off":
        Ledger().save(meter)
    return meter


class CostCallback(BaseCallbackHandler):
    """Adds every chat model call (coordinator and subagents) to the turn's meter."""

    def __init__(self, meter: TurnMeter):
        self.meter = meter
        self._runs: dict[Any, dict] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
        md = metadata or {}
        agent = md.get("lc_agent_name") or md.get("langgraph_node") or "agent"
        self._runs[run_id] = {"agent": agent, "model": md.get("ls_model_name") or ""}

    def on_llm_end(self, response, *, run_id, **kwargs):
        info = self._runs.pop(run_id, {"agent": "agent", "model": ""})
        for gens in response.generations or []:
            for gen in gens:
                msg = getattr(gen, "message", None)
                usage = getattr(msg, "usage_metadata", None)
                if not usage:
                    continue
                meta = getattr(msg, "response_metadata", None) or {}
                model = meta.get("model_name") or meta.get("model") or meta.get("model_id") or info["model"]
                self.meter.add_llm(model or "unknown", info["agent"], dict(usage))


# ---------- ledger ----------

_TURN_COLUMNS = ("ts", "turn_id", "thread_id", "customer", "model_calls", "tokens", "llm_cost_usd",
                 "guard_cost_usd", "total_cost_usd", "latency_ms", "guard_notes")
_SAVE_TURN = (f"INSERT INTO turns ({', '.join(_TURN_COLUMNS)}) VALUES ({', '.join('?' * len(_TURN_COLUMNS))})"
              " ON CONFLICT (turn_id) DO UPDATE SET "
              + ", ".join(f"{c}=excluded.{c}" for c in _TURN_COLUMNS if c != "turn_id"))


class Ledger:
    """The turns, llm_calls and guard_checks tables (see storage.LEDGER)."""

    @property
    def where(self) -> str:
        return LEDGER.where

    def save(self, m: TurnMeter) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with LEDGER.connect() as con:
            con.execute(_SAVE_TURN, (ts, m.turn_id, m.thread_id, m.customer, len(m.llm_calls), m.tokens, m.llm_cost,
                                     m.guard_cost, m.total_cost, m.latency_ms, json.dumps(m.guard_notes())))
            con.executemany("INSERT INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            [(ts, m.turn_id, m.thread_id, m.customer, c["agent"], c["model"], c["input_tokens"],
                              c["output_tokens"], c["cache_read"], c["cache_write"], c["cost_usd"])
                             for c in m.llm_calls])
            con.executemany("INSERT INTO guard_checks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            [(ts, m.turn_id, m.thread_id, m.customer, g["layer"], g["source"], g["mode"],
                              json.dumps(g["findings"]), int(g["applied"]), json.dumps(g["units"]),
                              g["cost_usd"], g["latency_ms"], g["error"]) for g in m.guard_checks])

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        with LEDGER.connect() as con:
            return con.execute(sql, params).fetchall()
