"""Check LangSmith tracing and send one test trace.

    python -m scripts.check_tracing

Tracing fails silently, so this checks each piece in turn:
  1. LANGSMITH_TRACING is "true"
  2. LANGSMITH_API_KEY looks like a LangSmith key
  3. the endpoint (US or EU LangSmith) accepts the key
  4. a test trace arrives in LANGSMITH_PROJECT
"""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv

from scripts._common import ROOT


def main() -> int:
    load_dotenv(ROOT / ".env", override=True)

    tracing = os.environ.get("LANGSMITH_TRACING") or os.environ.get("LANGCHAIN_TRACING_V2") or ""
    key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY") or ""
    project = os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT") or "default"
    endpoint = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    workspace = os.environ.get("LANGSMITH_WORKSPACE_ID", "")

    print(f"1. LANGSMITH_TRACING = '{tracing}' ...", end=" ")
    if tracing.strip().lower() != "true":
        print("NOT ON. Add this line to .env:  LANGSMITH_TRACING=true")
        return 1
    print("OK")

    print("2. LANGSMITH_API_KEY ...", end=" ")
    if not key:
        print("MISSING. Add LANGSMITH_API_KEY=... to .env (smith.langchain.com > Settings > API Keys)")
        return 1
    if key != key.strip() or any(c in key for c in "\"' "):
        print(f"starts '{key[:8]}...' -> contains quotes or spaces; remove them")
        return 1
    if not key.startswith("lsv2_"):
        print(f"starts '{key[:8]}...' -> LangSmith keys start with 'lsv2_'; check you copied the right key")
        return 1
    print(f"starts '{key[:8]}...', {len(key)} characters, format OK")

    print(f"   Project:  {project}")
    print(f"   Endpoint: {endpoint}" + ("   (EU LangSmith)" if "eu." in endpoint else "   (US LangSmith)"))
    if workspace:
        print(f"   Workspace id: {workspace}")

    from langsmith import Client, traceable

    client = Client()
    print("3. Key accepted by this endpoint ...", end=" ", flush=True)
    try:
        list(client.list_projects(limit=1))
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        print(f"FAILED\n   {type(e).__name__}: {msg[:250]}")
        if "401" in msg or "403" in msg or "Unauthorized" in msg:
            print("   If your LangSmith web address starts with eu.smith.langchain.com, add to .env:\n"
                  "     LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com\n"
                  "   If your key covers several workspaces, also add LANGSMITH_WORKSPACE_ID=<id>\n"
                  "   (Settings > Workspaces). Otherwise create a new API key.")
        return 1
    print("OK")

    print(f"4. Sending a test trace to project '{project}' ...", end=" ", flush=True)

    @traceable(name="tracing_check", project_name=project)
    def ping(text: str) -> str:
        return f"pong: {text}"

    ping("hello from check_tracing")
    client.flush()
    # Give LangSmith a moment, then confirm the project exists.
    for _ in range(10):
        try:
            client.read_project(project_name=project)
            print("OK")
            print(f"\nOpen LangSmith, pick the workspace this key belongs to, and look for project '{project}'.")
            print("Then run python -m scripts.chat; each question appears there as a trace.")
            return 0
        except Exception:  # noqa: BLE001
            time.sleep(1)
    print("NOT FOUND after 10s. The trace may still be arriving; refresh LangSmith in a minute.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
