"""Guardrails as LangChain agent middleware.

    InputGuardrail     before the coordinator runs: blocks card numbers, PINs, passwords and
                       security codes typed into the chat (removing them from the conversation
                       before any model, store or trace sees them), answers a typed account ID
                       with a fixed "only your own accounts" reply, masks contact details, then
                       asks Bedrock (if configured) about prompt attacks, harmful content and
                       denied topics.
    OutputGuardrail    after the coordinator answers: masks card/IBAN/account numbers and internal
                       ids, removes saved-figures ids (ds-...), "Data source" footnotes and image
                       links for charts, makes hyphens plain, adds what it can help with to a bare
                       "I can't help with that", then asks Bedrock about harmful content, denied topics and whether the answer is
                       grounded in what the subagents reported.
    ContactMasking     on the service subagent's tool results: the model sees r***@example.com,
                       not the full email or phone number (it never needs them).
    Call limits        stop a runaway loop of model or tool calls (a cost guardrail).
    ModelOutputCheck   on every model call: a reply with tool-call markup written out as text, a
                       broken tool call, or one character repeated over and over is never used.
                       The call is made again (twice at most); if it is still broken, the agent
                       says sorry instead. Open models such as gpt-oss do this now and then.

Every check is recorded on the turn's cost meter, so the chat and the ledger show what fired.
"""

from __future__ import annotations

import asyncio
import os
import re

from langchain.agents.middleware import (AgentMiddleware, ModelCallLimitMiddleware, ModelResponse,
                                         ToolCallLimitMiddleware, hook_config)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from guardrails import pii
from guardrails.bedrock import BedrockGuard
from observability import costs

BLOCKED_SECRET_REPLY = (
    "For your security, please don't type card numbers, PINs, passwords or security codes into this "
    "chat. I've removed it from our conversation and haven't acted on that message. Please send your "
    "request again without it. If you're asked for a security code, enter it at the separate "
    "\"Security code\" prompt, not in the chat.")

# An account ID typed into the chat is never one we showed (ids are masked in every answer), and the
# customer's own accounts are just "current" and "savings". Small models handled this unevenly (looked
# it up, sent customers to "the app or support"), so a fixed reply answers before any model is asked.
OTHER_ACCOUNT_REPLY = (
    "I can only see and use your own current and savings accounts, so I can't look up or send money to "
    "that account. Just say \"current\" or \"savings\" and I can help with balances, spending, "
    "statements and charts, or a transfer between them.")

BEDROCK_INPUT_REPLIES = {
    "PROMPT_ATTACK": "I can't help with that. I can help with your balances, transactions, transfers "
                     "between your accounts, your address and service requests.",
    "topic:Investment advice": "I can't give investment advice. I can show you your balances and "
                               "spending, or move money between your current and savings accounts.",
    "topic:Tax and legal advice": "I can't give tax or legal advice. For that, please speak to a "
                                  "qualified adviser. I can help with your accounts and transactions.",
}
BEDROCK_DEFAULT_REPLY = "Sorry, I can't help with that request. I can help with your accounts and transactions."
BEDROCK_OUTPUT_FALLBACK = "Sorry, I can't share that response. Please ask me another way or contact the bank."
MALFORMED_FALLBACK = "Sorry, something went wrong while I was working on that. Please ask me again."
LIMIT_REPLY = ("Sorry, I couldn't finish that one. Please ask again, perhaps one thing at a time "
               "(for example one month or one chart per message).")
_LIMIT_MESSAGE = re.compile(r"^\s*(?:Model|Tool) call limits? exceeded")  # LangChain's own wording

# What a broken reply looks like. gpt-oss writes tool calls in its own "harmony" format
# (<|channel|>commentary to=functions.task <|message|>{...}<|call|>); vLLM turns that into a proper
# tool call, but when the model garbles it, the markup comes through as plain text, usually
# followed by the same character hundreds of times until the token limit.
_MARKUP = re.compile(r"<\|(?:start|end|channel|message|call|constrain|return)\|>|\bto=functions\.\w"
                     r"|^\s*(?:analysis|commentary|final)\s*(?:to=|<\|)", re.MULTILINE)
_REPEATED = re.compile(r"([^\s\-=_*#|.])\1{29,}|(?:…\s*){20,}|\.{40,}")  # not markdown rules or tables


# Internal names that mean nothing to a customer: saved-figures ids (ds-4f2a9c1e) and "Data source:
# spending_by_month" footnotes. The prompts say not to show them; this catches the times a model does.
_DATASET_ID = re.compile(r"\bds-[0-9a-f]{8}\b")
_SOURCE_LINE = re.compile(r"^[\s>*_(\[]*(?:data\s*source|dataset(?:[\s_]?ids?)?|source)\b", re.IGNORECASE)
_ID_MENTION = re.compile(r"\s*[(\[]?\s*(?:dataset(?:[\s_]?id)?\s*[:=]?\s*)?`?\bds-[0-9a-f]{8}\b`?\s*[)\]]?",
                         re.IGNORECASE)


def scrub_internals(text: str) -> tuple[str, bool]:
    """The answer without dataset ids or data-source footnotes, and whether anything was removed."""
    if not _DATASET_ID.search(text):
        return text, False
    kept = []
    for line in text.split("\n"):
        if _DATASET_ID.search(line):
            if _SOURCE_LINE.match(line):
                continue  # the whole line is a note about where the figures came from
            line = _ID_MENTION.sub("", line).rstrip()
        kept.append(line)
    return "\n".join(kept).strip(), True


# Small things open models do that look wrong in a bank's chat:
# - gpt-oss writes non-breaking hyphens and narrow spaces ("SR\u201177860B"): a reference copied from the
#   chat wouldn't match the bank's, so they become ordinary hyphens and spaces;
# - an image link for the chart ("![Chart](attachment://chart.png)"): the app draws the chart itself;
# - a bare "I'm sorry, but I can't comply with that.": the customer is also told what they can ask.
HELP_LINE = ("I can help with your balances, spending, statements and charts, transfers between your current "
             "and savings accounts, or your address and cheque books.")
_PLAIN = str.maketrans({"\u2010": "-", "\u2011": "-", "\u00a0": " ", "\u202f": " "})
_FAKE_IMAGE = re.compile(r"!\[[^\]\n]*\]\([^)\n]*\)")
_LONE_HEADING = re.compile(r"^[\s#*_>]*(?:the\s+)?(?:chart|graph|image)s?[\s:*_]*$", re.IGNORECASE)
_BARE_REFUSAL = re.compile(
    r"^\s*(?:I['\u2019]m|I am)\s+sorry,?\s+(?:but\s+)?I\s+(?:can['\u2019]?t|cannot|can not|won['\u2019]t)\s+"
    r"(?:help|comply|assist|do)(?:\s+with)?(?:\s+(?:that|this)(?:\s+request)?)?\s*[.!]?\s*$", re.IGNORECASE)


# "£2\u202f853.91" or "£2 853.91" (gpt-oss groups thousands with narrow spaces): UK style is £2,853.91.
_SPACED_AMOUNT = re.compile(r"([£€$]\s?-?)(\d{1,3}(?:[ \u00a0\u202f]\d{3})+)(?![\d])")


def tidy(text: str) -> tuple[str, list[str]]:
    """The answer with the problems above fixed, and which ones were found (hyphens aren't reported)."""
    found = []
    text = _SPACED_AMOUNT.sub(lambda m: m.group(1) + re.sub(r"[ \u00a0\u202f]", ",", m.group(2)), text)
    text = text.translate(_PLAIN)
    if _FAKE_IMAGE.search(text):
        lines = _FAKE_IMAGE.sub("", text).rstrip().split("\n")
        while lines and (not lines[-1].strip() or _LONE_HEADING.match(lines[-1])):
            lines.pop()  # a "**Chart**" heading left with nothing under it
        text = "\n".join(lines)
        found.append("fake_image")
    if _BARE_REFUSAL.match(text):
        text = f"{text.strip()} {HELP_LINE}"
        found.append("bare_refusal")
    return text, found


def malformed(message) -> str | None:
    """Why a model reply can't be used, or None if it's fine."""
    if getattr(message, "invalid_tool_calls", None):
        return "broken_tool_call"
    text = _text(message)
    if not getattr(message, "tool_calls", None) and not text.strip():
        return "empty_reply"  # no answer and no tool call: a specialist would report nothing
    if _MARKUP.search(text):
        return "tool_markup_as_text"
    if _REPEATED.search(text):
        return "repeated_characters"
    return None


def _text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content or [])


def _last(messages, cls):
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], cls):
            return i, messages[i]
    return None, None


def _input_reply(findings: list[str]) -> str:
    for f in findings:
        if f.startswith("PROMPT_ATTACK"):
            return BEDROCK_INPUT_REPLIES["PROMPT_ATTACK"]
        if f in BEDROCK_INPUT_REPLIES:
            return BEDROCK_INPUT_REPLIES[f]
    return BEDROCK_DEFAULT_REPLY


def _record(layer: str, source: str, mode: str, findings: list[str], applied: bool, **extra) -> None:
    meter = costs.current()
    if meter is not None:
        meter.add_guard(layer, source, mode, findings, applied, **extra)


class InputGuardrail(AgentMiddleware):
    def __init__(self, bedrock: BedrockGuard | None = None):
        super().__init__()
        self.bedrock = bedrock

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state, runtime):
        messages = state.get("messages", [])
        idx, human = _last(messages, HumanMessage)
        if human is None:
            return None
        original = _text(human)

        # 1. Rules: secrets are removed from the conversation and the turn stops here.
        cleaned, blocked = pii.redact(original, pii.INPUT_BLOCK)
        cleaned, masked = pii.redact(cleaned, pii.INPUT_MASK)
        updates = []
        if blocked or masked:
            updates.append(HumanMessage(content=cleaned, id=human.id))  # same id: replaces it in place
            _record("rules", "INPUT", "enforce", sorted({m.kind for m in blocked + masked}), True)
        if blocked:
            return {"messages": updates + [AIMessage(content=BLOCKED_SECRET_REPLY)], "jump_to": "end"}
        if pii.find(cleaned, (pii.ACCOUNT_ID,)):
            _record("rules", "INPUT", "enforce", ["account_id"], True)
            return {"messages": updates + [AIMessage(content=OTHER_ACCOUNT_REPLY)], "jump_to": "end"}

        # 2. Bedrock: prompt attacks, harmful content, denied topics.
        if self.bedrock is not None:
            res = self.bedrock.check("INPUT", cleaned)
            apply = res.intervened and res.blocked and self.bedrock.enforce
            _record("bedrock", "INPUT", self.bedrock.mode, res.findings, apply, cost_usd=res.cost_usd,
                    units=res.units, latency_ms=res.latency_ms, error=res.error)
            if apply:
                if any(f.startswith("pii:") and f.endswith(":blocked") for f in res.findings):
                    # A secret the rules missed: take it out of the conversation as well.
                    updates = [HumanMessage(content="[message removed: it contained sensitive data]", id=human.id)]
                    reply = BLOCKED_SECRET_REPLY
                else:
                    reply = _input_reply(res.findings)
                return {"messages": updates + [AIMessage(content=reply)], "jump_to": "end"}
        return {"messages": updates} if updates else None

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(self, state, runtime):
        return await asyncio.to_thread(self.before_agent, state, runtime)


class OutputGuardrail(AgentMiddleware):
    def __init__(self, bedrock: BedrockGuard | None = None):
        super().__init__()
        self.bedrock = bedrock

    def after_agent(self, state, runtime):
        messages = state.get("messages", [])
        idx, answer = _last(messages, AIMessage)
        if answer is None or answer.tool_calls:
            return None
        original = _text(answer)
        if _LIMIT_MESSAGE.match(original):
            # A call limit stopped the turn; LangChain's message is for developers, not customers.
            _record("limits", "OUTPUT", "enforce", ["call_limit"], True)
            return {"messages": [AIMessage(content=LIMIT_REPLY, id=answer.id)]}
        text, tidied = tidy(original)
        if tidied:
            _record("tidy", "OUTPUT", "enforce", tidied, True)
        text, masked = pii.redact(text, pii.OUTPUT_MASK, style="partial")
        if masked:
            _record("rules", "OUTPUT", "enforce", sorted({m.kind for m in masked}), True)
        text, scrubbed = scrub_internals(text)
        if scrubbed:
            _record("rules", "OUTPUT", "enforce", ["internal_ids"], True)

        if self.bedrock is not None and text.strip():
            h_idx, human = _last(messages, HumanMessage)
            # What the subagents reported this turn: the facts the answer should rest on.
            reports = [_text(m) for m in messages[(h_idx or 0) + 1:] if isinstance(m, ToolMessage)]
            res = self.bedrock.check("OUTPUT", text, grounding_source="\n\n".join(reports) or None,
                                     query=_text(human) if human is not None else None)
            apply = res.intervened and self.bedrock.enforce
            _record("bedrock", "OUTPUT", self.bedrock.mode, res.findings, apply, cost_usd=res.cost_usd,
                    units=res.units, latency_ms=res.latency_ms, error=res.error)
            if apply:
                text = BEDROCK_OUTPUT_FALLBACK if res.blocked or not res.output_text else res.output_text

        if text != original:
            return {"messages": [AIMessage(content=text, id=answer.id)]}
        return None

    async def aafter_agent(self, state, runtime):
        return await asyncio.to_thread(self.after_agent, state, runtime)


class ContactMasking(AgentMiddleware):
    """Mask emails and phone numbers in tool results before the model reads them."""

    def _mask(self, result):
        if not isinstance(result, ToolMessage):
            return result
        content = result.content
        if isinstance(content, str):
            new = pii.redact(content, pii.CONTACT, style="partial")[0]
        else:
            new = [dict(b, text=pii.redact(b["text"], pii.CONTACT, style="partial")[0])
                   if isinstance(b, dict) and isinstance(b.get("text"), str) else b for b in content]
        if new != content:
            result = result.model_copy(update={"content": new})
        return result

    def wrap_tool_call(self, request, handler):
        return self._mask(handler(request))

    async def awrap_tool_call(self, request, handler):
        return self._mask(await handler(request))


class ModelOutputCheck(AgentMiddleware):
    """Ask the model again when its reply is broken (see `malformed`), so garbage never reaches the
    customer or the next agent. Retries happen inside one model step, so the call limits don't
    count them. MODEL_OUTPUT_RETRIES sets how many (default 2)."""

    RETRY_TEMPERATURE = 0.7  # a near-greedy model would repeat the same broken reply

    def __init__(self, retries: int | None = None):
        super().__init__()
        self.retries = int(os.environ.get("MODEL_OUTPUT_RETRIES", "2")) if retries is None else retries

    @staticmethod
    def _problem(response) -> str | None:
        messages = response.result if isinstance(response, ModelResponse) else [response]
        reply = next((m for m in messages if isinstance(m, AIMessage)), None)
        return malformed(reply) if reply is not None else None

    def _again(self, request):
        temperature = getattr(request.model, "temperature", None)
        if temperature is not None and temperature < self.RETRY_TEMPERATURE:
            return request.override(model_settings={**(request.model_settings or {}),
                                                    "temperature": self.RETRY_TEMPERATURE})
        return request

    def _note(self, problem: str, attempt: int) -> None:
        last = attempt >= self.retries
        _record("format", "MODEL", "fallback" if last else "retry", [problem], True)

    def wrap_model_call(self, request, handler):
        for attempt in range(self.retries + 1):
            response = handler(request if attempt == 0 else self._again(request))
            problem = self._problem(response)
            if problem is None:
                return response
            self._note(problem, attempt)
        return ModelResponse(result=[AIMessage(content=MALFORMED_FALLBACK)])

    async def awrap_model_call(self, request, handler):
        for attempt in range(self.retries + 1):
            response = await handler(request if attempt == 0 else self._again(request))
            problem = self._problem(response)
            if problem is None:
                return response
            self._note(problem, attempt)
        return ModelResponse(result=[AIMessage(content=MALFORMED_FALLBACK)])


def coordinator_guardrails(bedrock: BedrockGuard | None = None) -> list[AgentMiddleware]:
    return [InputGuardrail(bedrock), OutputGuardrail(bedrock), ModelOutputCheck(),
            # A normal turn is 2-4 coordinator calls; this only stops loops.
            ModelCallLimitMiddleware(run_limit=8, exit_behavior="end")]


def subagent_guardrails(mask_contacts: bool = False) -> list[AgentMiddleware]:
    limits = [ModelOutputCheck(),
              ModelCallLimitMiddleware(run_limit=8, exit_behavior="end"),
              ToolCallLimitMiddleware(run_limit=8, exit_behavior="end")]
    return ([ContactMasking()] if mask_contacts else []) + limits


__all__ = ["BLOCKED_SECRET_REPLY", "LIMIT_REPLY", "OTHER_ACCOUNT_REPLY", "MALFORMED_FALLBACK", "ContactMasking", "InputGuardrail",
           "ModelOutputCheck",
           "HELP_LINE", "OutputGuardrail", "coordinator_guardrails", "malformed", "scrub_internals",
           "subagent_guardrails", "tidy"]
