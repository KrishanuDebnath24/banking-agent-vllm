"""Check the model configured in .env before chatting.

For Ollama it first checks (1) the server answers, (2) the model is pulled,
(3) the model is tagged for tools. For every provider (Bedrock, Ollama,
Anthropic) it then checks (4) a test question gets an answer, and how long it
took, and (5) the model really calls a tool through this API, which the agent
needs.

    python -m scripts.check_llm
"""

from __future__ import annotations

import os
import sys
import time

import httpx
from dotenv import load_dotenv
from langchain.tools import tool

from scripts._common import ROOT
from agents.llm import make_model, model_spec, ollama_base_url


def check_ollama(name: str) -> bool:
    base = ollama_base_url()
    token = os.environ.get("OLLAMA_AUTH_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    print(f"1. Server at {base} ...", end=" ")
    try:
        tags = httpx.get(f"{base}/api/tags", headers=headers, timeout=10).json()
    except Exception as e:  # noqa: BLE001
        print(f"NOT REACHABLE ({type(e).__name__})")
        print("   Is the instance running, is your SSH tunnel open, and is the security group right?")
        return False
    print("OK")

    pulled = [m["name"] for m in tags.get("models", [])]
    print(f"2. Model '{name}' pulled ...", end=" ")
    if name not in pulled and f"{name}:latest" not in pulled:
        print("NO")
        print(f"   On the server run:  ollama pull {name}")
        print(f"   Models on the server: {', '.join(pulled) or 'none'}")
        return False
    print("OK")

    print("3. Tool calling supported ...", end=" ")
    try:
        show = httpx.post(f"{base}/api/show", json={"model": name}, headers=headers, timeout=30).json()
        caps = show.get("capabilities")
    except Exception:  # noqa: BLE001
        caps = None
    if caps is None:
        print("UNKNOWN (older Ollama); continuing")
    elif "tools" not in caps:
        print(f"NO (capabilities: {caps})")
        print("   Pick a model tagged 'tools' on ollama.com, e.g. qwen3:14b or gpt-oss:20b.")
        return False
    else:
        print("OK")
    return True


def describe_key(value: str) -> str:
    """Safe summary of a secret: first few characters and length only."""
    return f"starts '{value[:8]}...', {len(value)} characters"


def check_credentials(provider: str) -> bool:
    """Catch the common key mistakes before calling the provider."""
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        print("0. ANTHROPIC_API_KEY ...", end=" ")
        if not key:
            print("MISSING. Add ANTHROPIC_API_KEY=... to .env")
            return False
        problems = []
        if key != key.strip() or any(c in key for c in "\"' "):
            problems.append("contains quotes or spaces; remove them")
        if key.startswith("sk-ant-admin"):
            problems.append("this is an admin key; create a normal API key instead")
        elif not key.startswith("sk-ant-api"):
            problems.append("should start with 'sk-ant-api' (create one at console.anthropic.com > API Keys)")
        print(f"{describe_key(key)} -> " + ("; ".join(problems) if problems else "format OK"))
        return not problems

    if provider in ("bedrock_converse", "bedrock"):
        key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK", "")
        print("0. Bedrock credentials ...", end=" ")
        if not key:
            print("no AWS_BEARER_TOKEN_BEDROCK, so using your AWS profile (aws configure)")
            return True
        problems = []
        if key != key.strip() or any(c in key for c in "\"' "):
            problems.append("contains quotes or spaces; remove them")
        if not (key.startswith("ABSK") or key.startswith("bedrock-api-key-")):
            problems.append("doesn't look like a Bedrock API key (expected 'ABSK...' or 'bedrock-api-key-...'); "
                            "an IAM access key (AKIA...) goes in aws configure instead")
        print(f"{describe_key(key)} -> " + ("; ".join(problems) if problems else "format OK"))
        return not problems

    return True


def main() -> int:
    load_dotenv(ROOT / ".env", override=True)  # .env wins over old values set in Windows
    spec = model_spec()
    print(f"MODEL={spec}")
    provider, _, name = spec.partition(":")
    if not check_credentials(provider):
        return 1
    if provider == "ollama" and not check_ollama(name):
        return 1

    model = make_model()

    print("4. Test question ...", end=" ", flush=True)
    start = time.time()
    try:
        reply = model.invoke("Reply with exactly: ready")
    except Exception as e:  # noqa: BLE001
        print(f"FAILED\n   {type(e).__name__}: {str(e)[:300]}")
        if "verified" in str(e):
            print("   Your AWS account is still being verified. Wait (usually under 2 hours) and retry.")
        elif provider == "anthropic" and "credit" in str(e).lower():
            print("   Add credit under Billing at console.anthropic.com.")
        elif provider.startswith("bedrock"):
            print("   AccessDenied -> check IAM permission bedrock:InvokeModel and that the model is enabled\n"
                  "                   for your account in the Bedrock console (Model catalog).\n"
                  "   ValidationException / not found -> check the model id and AWS_REGION.\n"
                  "   Unable to locate credentials -> set AWS_BEARER_TOKEN_BEDROCK or run aws configure.")
        return 1
    text = reply.text if isinstance(getattr(reply, "text", None), str) else reply.content
    print(f"'{str(text).strip()[:60]}' in {time.time() - start:.1f}s")

    # The agent only works if the model can call tools through this API.
    print("5. Tool calling through this API ...", end=" ", flush=True)

    @tool
    def get_balance(account: str) -> str:
        """Get the balance of an account by name, e.g. 'current'."""
        return "100.00"

    try:
        reply = model.bind_tools([get_balance]).invoke("What is the balance of my current account? Use the tool.")
    except Exception as e:  # noqa: BLE001
        print(f"FAILED\n   {type(e).__name__}: {str(e)[:300]}")
        print("   This model/API combination doesn't accept tools. Pick another model.")
        return 1
    calls = getattr(reply, "tool_calls", None) or []
    if not calls:
        print("NO TOOL CALL (the model answered in text instead)")
        print("   The agent would never read the accounts. Try another model.")
        return 1
    print(f"OK -> {calls[0]['name']}({calls[0]['args']})")

    print("\nModel looks good. Next: python -m scripts.chat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
