"""The chat model for a self-hosted vLLM server (MODEL=vllm:<model>).

vLLM speaks the OpenAI API, so this is LangChain's ChatOpenAI with two additions on every call:

  x-session-id: <thread id>   The router sends every call that carries the same value to the same
                              vLLM pod, so the conversation's history is already in that pod's
                              prefix cache (KV cache in GPU memory) and isn't recomputed.
  cache_salt: <per customer>  vLLM only reuses cached blocks made with the same salt, so one
                              customer's cached prompts are never reused for another customer.

Both come from the turn that is running (observability.costs.start_turn sets it for the web app,
the eval harness and the chat scripts), so nothing else in the agents changes. Calls made outside
a turn carry neither.

Reasoning round trip (gpt-oss and other reasoning models):
  gpt-oss thinks before each tool call, and vLLM returns that thinking in a `reasoning` field.
  LangChain's ChatOpenAI drops the field, so on the next call of the same turn the model saw its
  own tool calls without the thinking behind them. That is outside what gpt-oss was trained on,
  and it is where it garbled its tool-call format ("to=functions.task?commentary?..."). Here the
  reasoning is kept on the reply and sent back with the assistant messages of the current turn
  (after the last final answer), which is what vLLM's gpt-oss format expects; vLLM itself drops
  it once a turn has a final answer. VLLM_SEND_REASONING=off turns this off.

Settings (.env):
    VLLM_BASE_URL         the server's OpenAI address, e.g. http://127.0.0.1:8001/v1 (SSH tunnel to
                          the GPU box) or http://vllm-router-service.llm.svc.cluster.local/v1
    VLLM_API_KEY          only if the server was started with --api-key
    VLLM_MAX_TOKENS       longest answer per call (default 1024)
    VLLM_CACHE_SALT       per-customer (default) or off
    VLLM_SESSION_HEADER   the router's session header (default x-session-id)
    VLLM_TIMEOUT_SECONDS  per call (default 120)
    VLLM_TEMPERATURE      sampling temperature (default: the model's own, 1.0 for gpt-oss). Lower,
                          e.g. 0.3, makes tool calls more regular; check with the evals
    VLLM_TOP_P            nucleus sampling (default: the model's own)
    VLLM_REASONING_EFFORT gpt-oss only: low, medium (its default) or high. Higher thinks longer before
                          each reply: slower, usually more reliable with tools
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

REASONING = "reasoning"  # where the thinking is kept on an AIMessage (additional_kwargs)


def base_url() -> str:
    return os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")


def session_header() -> str:
    return os.environ.get("VLLM_SESSION_HEADER", "x-session-id")


def cache_salt(customer: str) -> str:
    """A stable salt per customer. Keyed with APP_SECRET when it's set, so the salts can't be
    worked out from the customer ids; only the app ever sends them, never the browser."""
    secret = (os.environ.get("APP_SECRET") or "banking-agent-vllm").encode()
    return hmac.new(secret, f"cache-salt:{customer}".encode(), hashlib.sha256).hexdigest()[:32]


def _turn() -> tuple[str | None, str | None]:
    """(thread id, customer) of the turn that is running, if any."""
    from observability import costs

    meter = costs.current()
    return (meter.thread_id, meter.customer) if meter else (None, None)


def _reasoning_of(message: dict) -> str | None:
    """The thinking vLLM returned on a message or a streamed delta ("reasoning"; older versions
    called it "reasoning_content")."""
    return message.get("reasoning") or message.get("reasoning_content") or None


def _send_reasoning() -> bool:
    return os.environ.get("VLLM_SEND_REASONING", "on").strip().lower() not in ("off", "0", "false", "no")


def _attach_reasoning(messages: list, sent: list[dict]) -> None:
    """Put each assistant message's kept reasoning back on its request dict, for the current turn
    only: the messages after the last final answer (an assistant message with text and no tool
    calls). Earlier turns' thinking is left out, as gpt-oss's format drops it anyway."""
    if len(messages) != len(sent):
        return
    last_final = max((i for i, m in enumerate(messages)
                      if isinstance(m, AIMessage) and not m.tool_calls and str(m.content or "").strip()),
                     default=-1)
    for i, (m, d) in enumerate(zip(messages, sent)):
        if i > last_final and isinstance(m, AIMessage) and d.get("role") == "assistant":
            thinking = m.additional_kwargs.get(REASONING)
            if thinking:
                d["reasoning"] = thinking


class VLLMChat(ChatOpenAI):
    """ChatOpenAI that tells the router which conversation a call belongs to, salts the cache per
    customer, and keeps the model's reasoning between tool calls of a turn."""

    def _get_request_payload(self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if _send_reasoning() and payload.get("messages"):
            _attach_reasoning(self._convert_input(input_).to_messages(), payload["messages"])
        thread, customer = _turn()
        if thread:
            payload["extra_headers"] = {**(payload.get("extra_headers") or {}), session_header(): thread}
        if customer and os.environ.get("VLLM_CACHE_SALT", "per-customer").lower() != "off":
            payload["extra_body"] = {**(payload.get("extra_body") or {}), "cache_salt": cache_salt(customer)}
        return payload

    def _create_chat_result(self, response: Any, generation_info: dict | None = None):
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump()
        for gen, choice in zip(result.generations, raw.get("choices") or []):
            thinking = _reasoning_of((choice or {}).get("message") or {})
            if thinking and isinstance(gen.message, AIMessage):
                gen.message.additional_kwargs[REASONING] = thinking
        return result

    def _convert_chunk_to_generation_chunk(self, chunk: dict, default_chunk_class: type,
                                           base_generation_info: dict | None):
        gen = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
        if gen is None:
            return gen
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
        if choices:
            thinking = _reasoning_of((choices[0] or {}).get("delta") or {})
            if thinking:  # streamed pieces join up when LangChain adds the chunks together
                gen.message.additional_kwargs[REASONING] = thinking
        return gen


def _sampling() -> dict:
    """Optional sampling settings from .env; unset ones are left to the model's defaults."""
    out: dict[str, Any] = {}
    if os.environ.get("VLLM_TEMPERATURE", "").strip():
        out["temperature"] = float(os.environ["VLLM_TEMPERATURE"])
    if os.environ.get("VLLM_TOP_P", "").strip():
        out["top_p"] = float(os.environ["VLLM_TOP_P"])
    if os.environ.get("VLLM_REASONING_EFFORT", "").strip():
        out["reasoning_effort"] = os.environ["VLLM_REASONING_EFFORT"].strip().lower()
    return out


def make_vllm_model(name: str, **extra: Any) -> VLLMChat:
    extra = {**_sampling(), **extra}
    return VLLMChat(
        model=name,
        base_url=base_url(),
        api_key=os.environ.get("VLLM_API_KEY") or "not-needed",
        max_tokens=int(os.environ.get("VLLM_MAX_TOKENS", "1024")),
        timeout=float(os.environ.get("VLLM_TIMEOUT_SECONDS", "120")),
        max_retries=2,
        stream_usage=True,          # token counts (and cached tokens) on streamed answers too
        use_responses_api=False,    # vLLM's chat completions endpoint
        **extra,
    )
