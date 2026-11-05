"""PII redaction for LangSmith traces.

LangSmith is a third party, so traces are scrubbed before they leave the machine: card
numbers, codes, emails, phones, IBANs, tokens and postcodes are masked, and any field whose
name contains "address" (tool arguments and results) is replaced with [address].

Call install_trace_redaction() once at start-up, before the agent runs. Set
LANGSMITH_HIDE_CONTENT=true to send no message content at all (timings and structure only),
which is the safer default for production.
"""

from __future__ import annotations

import os

from guardrails.pii import trace_replacer


def install_trace_redaction() -> str:
    """Configure the shared LangSmith client that LangChain's tracer uses. Returns a status line."""
    from langsmith import anonymizer, run_trees

    hide = os.environ.get("LANGSMITH_HIDE_CONTENT", "").strip().lower() in ("1", "true", "yes")
    scrub = anonymizer.create_anonymizer(trace_replacer)
    if hide:
        client = run_trees.get_cached_client(hide_inputs=True, hide_outputs=True)
        client._hide_inputs = client._hide_outputs = True  # in case the client already existed
        return "content hidden"
    client = run_trees.get_cached_client(anonymizer=scrub)
    client._anonymizer = scrub  # the client may have been created earlier without it
    return "PII redacted"
