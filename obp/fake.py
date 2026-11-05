"""In-memory stand-in for the OBP sandbox, for offline tests and demos.

Enable with OBP_FAKE=1. It mirrors the OBPClient methods the MCP servers use,
keyed by the accounts in demo_accounts.json. Tokens are "fake-token-A" and
"fake-token-B".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from obp.client import OBPError


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class FakeOBP:
    users: dict[str, dict]                      # token -> {"user_id", "username"}
    balances: dict[str, float]                  # account id -> balance
    txns: dict[str, list[dict]] = field(default_factory=dict)
    customers: dict[str, dict] = field(default_factory=dict)   # username -> customer record
    messages: list[dict] = field(default_factory=list)
    attributes: dict[str, list[dict]] = field(default_factory=dict)  # user_id -> personal user attributes
    max_transfer: float = 850.0
    currency: str = "GBP"

    @classmethod
    def from_demo(cls, demo_file: Path) -> "FakeOBP":
        demo = json.loads(Path(demo_file).read_text())
        users, balances, customers = {}, {}, {}
        for label, c in demo["customers"].items():
            users[f"fake-token-{label}"] = {"user_id": f"user-{label}", "username": c["username"]}
            for role in ("current", "savings", "income", "merchants"):
                balances[c[role]] = 0.0
            customers[c["username"]] = {"customer_id": f"cust-{label}", "customer_number": f"10000{ord(label)}",
                                        "legal_name": c["username"].split(".")[0].capitalize() + " Demo",
                                        "email": c["username"], "mobile_phone_number": "+44 7700 900000",
                                        "kyc_status": True}
        fake = cls(users=users, balances=balances, customers=customers)
        # A little starting activity so reads return something.
        for label, c in demo["customers"].items():
            fake._post(c["income"], 5000.0, "Opening funds")
            fake.transfer(f"fake-token-{label}", demo["bank_id"], c["income"], c["current"], 712.50, "GBP",
                          "ACME Analytics Ltd - Weekly pay")
            fake.transfer(f"fake-token-{label}", demo["bank_id"], c["current"], c["merchants"], 41.87, "GBP",
                          "Tesco Stores 2041")
            fake.transfer(f"fake-token-{label}", demo["bank_id"], c["current"], c["merchants"], 10.99, "GBP",
                          "Netflix.com")
        return fake

    @classmethod
    def from_fixture(cls, fixture_file: str | Path) -> "FakeOBP":
        """A fixed bank with dated transactions (see evals/fixture.py), for repeatable evals."""
        data = json.loads(Path(fixture_file).read_text())
        users, balances, customers, txns, attributes = {}, {}, {}, {}, {}
        for label, c in data["customers"].items():
            users[f"fake-token-{label}"] = {"user_id": f"user-{label}", "username": c["username"]}
            customers[c["username"]] = {"customer_id": f"cust-{label}", "customer_number": c["customer_number"],
                                        "legal_name": c["legal_name"], "email": c["username"],
                                        "mobile_phone_number": c["mobile"], "kyc_status": True}
            for acc in c["accounts"].values():
                balances[acc["id"]] = acc["opening"]
                txns[acc["id"]] = []
            if c.get("address"):  # the address on file, stored like the real demo stores it
                attributes[f"user-{label}"] = [{"user_attribute_id": f"attr-{label}", "name": "ADDRESS",
                                                "type": "STRING", "value": c["address"],
                                                "insert_date": "2026-07-01T09:00:00Z"}]
            for t in sorted(c["transactions"], key=lambda t: t["date"]):
                acc = c["accounts"][t["account"]]["id"]
                balances[acc] = round(balances[acc] + t["amount"], 2)
                txns[acc].insert(0, {"details": {
                    "completed": f"{t['date']}T09:00:00Z", "description": t["description"],
                    "value": {"currency": c["currency"], "amount": f"{t['amount']:.2f}"},
                    "new_balance": {"currency": c["currency"], "amount": f"{balances[acc]:.2f}"}}})
        return cls(users=users, balances=balances, txns=txns, customers=customers, attributes=attributes)

    # ---- helpers ----
    def _user(self, token: str) -> dict:
        if token not in self.users:
            raise OBPError(401, '{"message":"OBP-20001: User not logged in"}', "fake")
        return self.users[token]

    def _post(self, acc: str, amount: float, desc: str) -> None:
        self.balances[acc] = round(self.balances.get(acc, 0.0) + amount, 2)
        self.txns.setdefault(acc, []).insert(0, {"details": {
            "completed": _now(), "description": desc,
            "value": {"currency": self.currency, "amount": f"{amount:.2f}"},
            "new_balance": {"currency": self.currency, "amount": f"{self.balances[acc]:.2f}"}}})

    # ---- OBPClient-compatible methods ----
    def current_user(self, token: str) -> dict:
        return self._user(token)

    def balance_amount(self, token: str, bank_id: str, account_id: str) -> tuple[float, str]:
        self._user(token)
        return self.balances[account_id], self.currency

    def transactions(self, token: str, bank_id: str, account_id: str, view_id: str = "owner") -> list[dict]:
        self._user(token)
        return list(self.txns.get(account_id, []))

    def transfer(self, token, bank_id, from_account, to_account, amount, currency, description, view_id="owner"):
        self._user(token)
        if amount > self.max_transfer:
            raise OBPError(409, "needs approval", "fake")
        self._post(from_account, -amount, description)
        self._post(to_account, amount, description)
        return {"id": f"tr-{len(self.messages)}-{amount}", "status": "COMPLETED"}

    def my_customers(self, token: str, bank_id: str) -> list[dict]:
        user = self._user(token)
        c = self.customers.get(user["username"])
        return [c] if c else []

    def my_user_attributes(self, token: str) -> list[dict]:
        user = self._user(token)
        return [dict(a) for a in self.attributes.get(user["user_id"], [])]

    def set_my_user_attribute(self, token: str, name: str, value: str) -> dict:
        attrs = self.attributes.setdefault(self._user(token)["user_id"], [])
        attr = next((a for a in attrs if a["name"] == name), None)
        if attr is None:
            attr = {"user_attribute_id": f"attr-{len(attrs) + 1}", "name": name, "type": "STRING",
                    "insert_date": _now()}
            attrs.append(attr)
        attr["value"] = value[:255]
        return dict(attr)

    def create_customer_message(self, token, bank_id, customer_id, message, **kw) -> dict:
        self._user(token)
        self.messages.append({"customer_id": customer_id, "message": message})
        return {"success": "Success"}
