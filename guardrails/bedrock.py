"""Amazon Bedrock Guardrails through the ApplyGuardrail API.

ApplyGuardrail checks any text, so it works with Claude on the Anthropic API as well as
with models on Bedrock. Turned on by GUARDRAIL_ID in .env; without it this layer is off and
only the deterministic rules run.

Modes (GUARDRAIL_MODE):
    shadow   (default) check and log what would be blocked, but never block
    enforce  block or rewrite when the guardrail intervenes
Failures (GUARDRAIL_ON_ERROR):
    open     (default) if Bedrock is unreachable, carry on and log it. Safe here because the
             hard controls (server-side account checks, confirmations, security codes)
             don't depend on this layer.
    closed   refuse the turn instead
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

# USD per 1,000 text units (1 unit = up to 1,000 characters), Classic tier, on-demand.
PRICE_PER_1000_UNITS = {
    "topicPolicyUnits": 0.15,
    "contentPolicyUnits": 0.15,
    "sensitiveInformationPolicyUnits": 0.10,
    "contextualGroundingPolicyUnits": 0.10,
    "wordPolicyUnits": 0.0,
    "sensitiveInformationPolicyFreeUnits": 0.0,
}
MAX_CHARS = 8000  # cap per check: bounds cost and latency for long tool outputs


@dataclass
class GuardResult:
    source: str                      # INPUT or OUTPUT
    intervened: bool = False         # Bedrock would block or rewrite this text
    blocked: bool = False            # a BLOCK action (as opposed to only anonymizing)
    findings: list[str] = field(default_factory=list)  # e.g. ["PROMPT_ATTACK", "topic:Investment advice"]
    output_text: str | None = None   # Bedrock's rewritten text (anonymized or blocked message)
    grounding: dict = field(default_factory=dict)       # {"GROUNDING": 0.91, "RELEVANCE": 0.88}
    units: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    latency_ms: int = 0
    error: str | None = None


def _cost(units: dict) -> float:
    return round(sum(units.get(k, 0) * p / 1000 for k, p in PRICE_PER_1000_UNITS.items()), 8)


def parse(source: str, response: dict) -> GuardResult:
    """Turn an ApplyGuardrail response into findings we can log and act on."""
    res = GuardResult(source=source, intervened=response.get("action") == "GUARDRAIL_INTERVENED")
    res.units = {k: v for k, v in (response.get("usage") or {}).items() if isinstance(v, int)}
    res.cost_usd = _cost(res.units)
    for a in response.get("assessments") or []:
        for t in (a.get("topicPolicy") or {}).get("topics", []):
            if t.get("action") == "BLOCKED" or t.get("detected"):
                res.findings.append(f"topic:{t.get('name')}")
                res.blocked |= t.get("action") == "BLOCKED"
        for f in (a.get("contentPolicy") or {}).get("filters", []):
            if f.get("action") == "BLOCKED" or f.get("detected"):
                res.findings.append(f"{f.get('type')}:{f.get('confidence', '').lower()}")
                res.blocked |= f.get("action") == "BLOCKED"
        sens = a.get("sensitiveInformationPolicy") or {}
        for p in sens.get("piiEntities", []) + sens.get("regexes", []):
            if p.get("action") in ("BLOCKED", "ANONYMIZED") or p.get("detected"):
                res.findings.append(f"pii:{p.get('type') or p.get('name')}:{(p.get('action') or '').lower()}")
                res.blocked |= p.get("action") == "BLOCKED"
        for g in (a.get("contextualGroundingPolicy") or {}).get("filters", []):
            res.grounding[g.get("type")] = g.get("score")
            if g.get("action") == "BLOCKED" or (g.get("score") is not None and g.get("threshold") is not None
                                                and g["score"] < g["threshold"]):
                res.findings.append(f"{g.get('type', '').lower()}:{g.get('score'):.2f}<{g.get('threshold')}")
                res.blocked |= g.get("action") == "BLOCKED"
    outputs = response.get("outputs") or []
    if res.intervened and outputs:
        res.output_text = outputs[0].get("text")
    return res


class BedrockGuard:
    def __init__(self, guardrail_id: str, version: str = "DRAFT", region: str = "eu-west-1",
                 mode: str = "shadow", on_error: str = "open", client=None):
        self.guardrail_id, self.version, self.region = guardrail_id, version, region
        self.mode, self.on_error = mode, on_error
        self._client = client

    @classmethod
    def from_env(cls) -> "BedrockGuard | None":
        gid = os.environ.get("GUARDRAIL_ID", "").strip()
        if not gid:
            return None
        return cls(gid, os.environ.get("GUARDRAIL_VERSION", "DRAFT").strip() or "DRAFT",
                   os.environ.get("GUARDRAIL_REGION") or os.environ.get("AWS_REGION", "eu-west-1"),
                   os.environ.get("GUARDRAIL_MODE", "shadow").strip().lower(),
                   os.environ.get("GUARDRAIL_ON_ERROR", "open").strip().lower())

    @property
    def enforce(self) -> bool:
        return self.mode == "enforce"

    @property
    def client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def check(self, source: str, text: str, grounding_source: str | None = None,
              query: str | None = None) -> GuardResult:
        """source "INPUT" for what the customer typed, "OUTPUT" for the assistant's answer.
        For OUTPUT, grounding_source (what the tools returned) and query (the question)
        turn on the contextual grounding check."""
        content = [{"text": {"text": text[:MAX_CHARS], "qualifiers": ["guard_content"]}}]
        if source == "OUTPUT" and grounding_source and query:
            content = [{"text": {"text": grounding_source[:MAX_CHARS], "qualifiers": ["grounding_source"]}},
                       {"text": {"text": query[:2000], "qualifiers": ["query"]}},
                       {"text": {"text": text[:MAX_CHARS], "qualifiers": ["guard_content"]}}]
        started = time.perf_counter()
        try:
            response = self.client.apply_guardrail(guardrailIdentifier=self.guardrail_id,
                                                   guardrailVersion=self.version, source=source,
                                                   content=content, outputScope="INTERVENTIONS")
        except Exception as e:  # noqa: BLE001
            res = GuardResult(source=source, error=f"{type(e).__name__}: {str(e)[:200]}")
            if self.on_error == "closed":
                res.intervened = res.blocked = True
                res.findings.append("guardrail_unavailable")
        else:
            res = parse(source, response)
        res.latency_ms = int((time.perf_counter() - started) * 1000)
        return res
