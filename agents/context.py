"""Per-session customer context.

Everything the tools need to act *as this customer* lives here: the OBP token,
the accounts this customer may see, and the bank. It is passed to the agent at
run time (LangGraph `context`), never shown to the model, and never taken from
model output. That keeps identity and authorisation out of the LLM's hands.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from obp.client import OBPClient

ROOT = Path(__file__).resolve().parent.parent
VISIBLE_ROLES = ("current", "savings")  # hidden helper accounts never reach the agent

_FILLER = re.compile(r"\b(?:my|the|our|account|acct|a/c)\b", re.I)
_ALIASES = {"saving": "savings", "savings": "savings", "current": "current"}


def account_key(name: str) -> str:
    """How customers say it -> the account's name: "my Current Account" -> "current",
    "savings acct" -> "savings". Anything else ("joint account") stays unrecognised."""
    key = " ".join(_FILLER.sub(" ", name or "").split()).lower()
    return _ALIASES.get(key, key)


@dataclass(kw_only=True)
class CustomerContext:
    label: str                     # "A" or "B"
    username: str
    token: str = field(repr=False)  # OBP Direct Login token; kept out of logs and prompts
    bank_id: str
    currency: str
    accounts: dict[str, str]       # friendly name -> OBP account id, e.g. {"current": "u8ll..."}
    client: OBPClient = field(repr=False)
    show_since: str | None = None  # ISO date; older transactions are hidden in the demo
    display_name: str = field(default="")

    def resolve(self, account: str) -> str | None:
        """Map a name ('current') or id to an allowed account id, else None."""
        key = account_key(account)
        if key in self.accounts:
            return self.accounts[key]
        return account.strip() if account.strip() in self.accounts.values() else None


def _first_seed_time(label: str) -> str | None:
    """Earliest seeded transfer for this customer, so pre-existing sandbox
    noise (other developers' test transactions) stays out of the demo."""
    log = ROOT / "seed_log.jsonl"
    if not log.exists():
        return None
    times = []
    for line in log.read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("customer") == label and entry.get("status") == "COMPLETED":
            times.append(entry["at"][:10])
    if not times:
        return None
    # Date-level with a day's margin, so clock differences can't hide seeded data;
    # the pre-existing noise in the shared sandbox is years older anyway.
    return (date.fromisoformat(min(times)) - timedelta(days=1)).isoformat()


def build_context(client: OBPClient, label: str, username: str, token: str) -> CustomerContext:
    demo = json.loads((ROOT / "demo_accounts.json").read_text())
    cust = demo["customers"][label]
    if cust["username"] != username:
        raise ValueError(f"demo_accounts.json has {cust['username']} for {label}, but you logged in as {username}")
    return CustomerContext(
        label=label,
        username=username,
        token=token,
        bank_id=demo["bank_id"],
        currency=cust["currency"],
        accounts={role: cust[role] for role in VISIBLE_ROLES},
        client=client,
        show_since=_first_seed_time(label),
        display_name=username.split(".")[0].capitalize(),
    )
