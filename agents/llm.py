"""Pick the chat model from .env.

    MODEL=vllm:openai/gpt-oss-20b               self-hosted vLLM (one GPU box or Kubernetes)
    MODEL=bedrock_converse:qwen.qwen3-32b-v1:0  Amazon Bedrock (any Bedrock model id)
    MODEL=ollama:qwen3:14b                      self-hosted Ollama (e.g. on an AWS GPU instance)
    MODEL=anthropic:claude-sonnet-5-5           Anthropic API

vLLM settings: VLLM_BASE_URL and the others in agents/vllm_chat.py.

Bedrock settings:
    AWS_REGION        e.g. eu-west-1 (the model must be offered in this region)
    Credentials: either AWS_BEARER_TOKEN_BEDROCK (a Bedrock API key from the console)
    or normal AWS credentials (aws configure / AWS_ACCESS_KEY_ID ... / SSO profile).

Ollama settings:
    OLLAMA_BASE_URL   where the server listens, e.g. http://localhost:11434 (SSH tunnel)
                      or http://10.0.1.25:11434 (private IP inside your VPC)
    OLLAMA_NUM_CTX    context window in tokens (default 16384). Ollama's own default is
                      small; tool definitions plus transaction lists overflow it and the
                      model silently loses the start of the conversation.
    OLLAMA_AUTH_TOKEN optional bearer token, if you put Ollama behind an auth proxy
"""

from __future__ import annotations

import os
import re

DEFAULT_MODEL = "vllm:openai/gpt-oss-20b"
_CLAUDE_GEN = re.compile(r"claude-(?:[a-z]+-)?(\d+)")


def temperature_kwargs(name: str) -> dict:
    """temperature=0 for repeatable answers, except on models that reject it.

    Newer Claude models (generation 5 and up, e.g. claude-haiku-5-5) return
    "temperature is deprecated for this model", so it isn't sent for them.
    Override with MODEL_TEMPERATURE in .env (a number, or "none" to never send it).
    """
    override = os.environ.get("MODEL_TEMPERATURE", "").strip().lower()
    if override == "none":
        return {}
    if override:
        return {"temperature": float(override)}
    lowered = name.lower()
    if "fable" in lowered or "mythos" in lowered:
        return {}
    m = _CLAUDE_GEN.search(lowered)
    if m and int(m.group(1)) >= 5:
        return {}
    return {"temperature": 0}


def model_spec() -> str:
    return os.environ.get("MODEL", DEFAULT_MODEL)


def ollama_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def make_model(spec: str | None = None):
    """The model named in .env (MODEL), or the given spec, e.g. "anthropic:claude-sonnet-5-5"."""
    spec = spec or model_spec()
    provider, _, name = spec.partition(":")  # "ollama:qwen3:14b" -> ("ollama", "qwen3:14b")

    if provider == "vllm":
        from agents.vllm_chat import make_vllm_model

        return make_vllm_model(name, **temperature_kwargs(name))

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        token = os.environ.get("OLLAMA_AUTH_TOKEN")
        return ChatOllama(
            model=name,
            base_url=ollama_base_url(),
            num_ctx=int(os.environ.get("OLLAMA_NUM_CTX", "16384")),
            **temperature_kwargs(name),
            keep_alive="30m",  # keep the model loaded on the GPU between questions
            client_kwargs={"headers": {"Authorization": f"Bearer {token}"}} if token else {},
        )

    if provider in ("bedrock_converse", "bedrock"):
        from langchain_aws import ChatBedrockConverse

        return ChatBedrockConverse(
            model_id=name,  # model ids contain ":" too, e.g. qwen.qwen3-32b-v1:0
            region_name=os.environ.get("AWS_REGION", "eu-west-1"),
            max_tokens=2048,
            **temperature_kwargs(name),
        )

    from langchain.chat_models import init_chat_model

    return init_chat_model(spec, **temperature_kwargs(name))
