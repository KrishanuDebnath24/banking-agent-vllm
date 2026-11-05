"""Turn your .env into the cluster's "bank-env" Secret, without writing it to disk.

    python deploy/k8s/env_to_secret.py | kubectl apply -f -

Reads .env the same way the app does (comments and quotes handled), leaves out empty values and
the settings the cluster sets itself (database, addresses, model, ports), and prints a Secret
manifest for kubectl. Run it again after changing .env, then restart the app:
    kubectl -n bank rollout restart deploy/api deploy/mcp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]

# Set by the manifests in deploy/k8s/bank (or meaningless inside the cluster).
CLUSTER_SETS = {
    "DATABASE_URL", "POSTGRES_PASSWORD", "POSTGRES_PORT", "MODEL", "VLLM_BASE_URL", "LANGSMITH_PROJECT",
    "MCP_ACCOUNTS_URL", "MCP_TRANSACTIONS_URL", "MCP_SERVICE_URL", "MCP_CHARTS_URL", "VERIFY_URL",
    "MCP_HOST", "MCP_ALLOWED_HOSTS", "MCP_PORT_OFFSET", "API_HOST", "PORT", "COOKIE_SECURE",
    "SESSION_DB", "OTP_DB", "TELEMETRY_DB", "SERVICE_REQUESTS_DB", "GUEST_DB", "WEB_DB", "CHART_DB",
    # Phase 0 (one GPU machine) settings: the cluster's vLLM is configured in helm/vllm-stack.values.yaml
    "VLLM_MODEL", "VLLM_TOOL_PARSER", "VLLM_TAG", "VLLM_PREFIX_CACHING_FLAG", "VLLM_KV_CACHE_DTYPE",
    "VLLM_MAX_NUM_SEQS", "VLLM_MAX_NUM_BATCHED_TOKENS", "VLLM_MAX_MODEL_LEN",
}


def secret(env_file: Path, name: str = "bank-env", namespace: str = "bank") -> dict:
    values = dotenv_values(env_file)
    data = {k: v for k, v in values.items() if v not in (None, "") and k not in CLUSTER_SETS}
    return {"apiVersion": "v1", "kind": "Secret", "type": "Opaque",
            "metadata": {"name": name, "namespace": namespace}, "stringData": data}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = ap.parse_args()
    if not args.env_file.exists():
        print(f"No {args.env_file}: copy .env.example to .env and fill it in first.", file=sys.stderr)
        return 1
    manifest = secret(args.env_file)
    print(f"bank-env: {len(manifest['stringData'])} settings from {args.env_file.name} "
          f"(names only: {', '.join(sorted(manifest['stringData']))})", file=sys.stderr)
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
