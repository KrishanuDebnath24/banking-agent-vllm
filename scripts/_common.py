"""Shared helpers for the scripts: env loading and customer logins."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from obp.client import OBPClient  # noqa: E402

DEMO_ACCOUNTS_FILE = ROOT / "demo_accounts.json"


def load_env() -> None:
    # override=True: values in .env win over old ones set in Windows / the shell
    load_dotenv(ROOT / ".env", override=True)


def setup() -> tuple[OBPClient, str]:
    load_env()
    return OBPClient.from_env(), os.environ.get("OBP_BANK_ID", "gh.29.uk")


def login(client: OBPClient, label: str) -> tuple[str, str]:
    """Log in customer A or B. Returns (username, token)."""
    user = os.environ[f"CUSTOMER_{label}_USERNAME"]
    token = client.login(user, os.environ[f"CUSTOMER_{label}_PASSWORD"])
    return user, token


def load_demo_accounts() -> dict:
    if not DEMO_ACCOUNTS_FILE.exists():
        sys.exit(f"{DEMO_ACCOUNTS_FILE.name} not found. Run: python -m scripts.list_accounts --write")
    return json.loads(DEMO_ACCOUNTS_FILE.read_text())
