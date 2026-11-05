"""Minimal Open Bank Project (OBP) client using Direct Login.

One client is shared by the app; each call takes the customer's token,
so the token always comes from the request context, never from the LLM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


class OBPError(Exception):
    """Raised when OBP returns a non-2xx response."""

    def __init__(self, status: int, body: str, path: str):
        self.status = status
        self.body = body
        self.path = path
        super().__init__(f"OBP {status} on {path}: {body[:300]}")


@dataclass
class OBPClient:
    base_url: str          # e.g. https://apisandbox.openbankproject.com (no trailing slash)
    consumer_key: str
    version: str = "v5.0.0"
    timeout: float = 20.0
    # Transfers above OBP's challenge threshold need a second user to approve
    # them (maker/checker), so scripts keep every single transfer below this.
    max_transfer: float = 800.0

    @classmethod
    def from_env(cls) -> "OBPClient":
        return cls(
            base_url=os.environ["OBP_BASE_URL"].rstrip("/"),
            consumer_key=os.environ["OBP_CONSUMER_KEY"],
            version=os.environ.get("OBP_API_VERSION", "v5.0.0"),
            max_transfer=float(os.environ.get("OBP_MAX_TRANSFER", "800")),
        )

    def split_amount(self, amount: float) -> list[float]:
        """Split an amount into chunks no bigger than max_transfer."""
        chunks, left = [], round(amount, 2)
        while left > 0.004:
            part = round(min(left, self.max_transfer), 2)
            chunks.append(part)
            left = round(left - part, 2)
        return chunks

    # ---------- auth ----------

    def login(self, username: str, password: str) -> str:
        """Direct Login: returns a token for this customer."""
        auth = (
            f'DirectLogin username="{username}",'
            f'password="{password}",'
            f'consumer_key="{self.consumer_key}"'
        )
        r = httpx.post(
            f"{self.base_url}/my/logins/direct",
            headers={"Authorization": auth, "Content-Type": "application/json"},
            timeout=self.timeout,
        )
        if r.status_code >= 300:
            raise OBPError(r.status_code, r.text, "/my/logins/direct")
        return r.json()["token"]

    # ---------- low-level ----------

    def _request(self, method: str, token: str, path: str, json: dict | None = None,
                 version: str | None = None) -> dict:
        url = f"{self.base_url}/obp/{version or self.version}{path}"
        r = httpx.request(
            method,
            url,
            headers={
                "Authorization": f'DirectLogin token="{token}"',
                "Content-Type": "application/json",
            },
            json=json,
            timeout=self.timeout,
        )
        if r.status_code >= 300:
            raise OBPError(r.status_code, r.text, path)
        return r.json() if r.content else {}

    def get(self, token: str, path: str) -> dict:
        return self._request("GET", token, path)

    # ---------- read operations used by the agents ----------

    def current_user(self, token: str) -> dict:
        return self.get(token, "/users/current")

    def accounts_held(self, token: str, bank_id: str) -> list[dict]:
        """Get Accounts Held: the customer's accounts at one bank."""
        data = self.get(token, f"/banks/{bank_id}/accounts-held")
        return data.get("accounts", [])

    def balances(self, token: str, bank_id: str, account_id: str) -> dict:
        """Get Account Balances."""
        return self.get(token, f"/banks/{bank_id}/accounts/{account_id}/balances")

    def transactions(self, token: str, bank_id: str, account_id: str, view_id: str = "owner") -> list[dict]:
        """Get Transactions for Account (Full) through a view."""
        data = self.get(token, f"/banks/{bank_id}/accounts/{account_id}/{view_id}/transactions")
        return data.get("transactions", [])

    def balance_amount(self, token: str, bank_id: str, account_id: str) -> tuple[float, str]:
        """First balance on the account as (amount, currency)."""
        data = self.balances(token, bank_id, account_id)
        bals = data.get("balances") or [{}]
        return float(bals[0].get("amount", 0) or 0), bals[0].get("currency", "")

    # ---------- write operations ----------

    def transfer(
        self,
        token: str,
        bank_id: str,
        from_account: str,
        to_account: str,
        amount: float,
        currency: str,
        description: str,
        view_id: str = "owner",
    ) -> dict:
        """Move money between two accounts at the same bank (ACCOUNT transaction request).

        Small transfers complete at once. Above OBP's challenge threshold the
        request waits for approval by a different user (maker/checker), which a
        single customer can't give, so this raises a clear error instead.
        """
        if amount > self.max_transfer:
            raise OBPError(400, f"Amount {amount:.2f} is above max_transfer {self.max_transfer:.2f}; "
                                "split it with split_amount()", "transfer")
        base = (
            f"/banks/{bank_id}/accounts/{from_account}/{view_id}"
            f"/transaction-request-types/ACCOUNT/transaction-requests"
        )
        body = {
            "to": {"bank_id": bank_id, "account_id": to_account},
            "value": {"currency": currency, "amount": f"{amount:.2f}"},
            "description": description,
        }
        result = self._request("POST", token, base, json=body)

        status = str(result.get("status", "")).upper()
        if status == "COMPLETED":
            return result
        raise OBPError(
            409,
            f"Transfer of {amount:.2f} was not completed (status {status or 'unknown'}); it needs "
            f"approval. Lower OBP_MAX_TRANSFER in .env (now {self.max_transfer:.2f}) and retry.",
            base,
        )

    def create_account(self, token: str, bank_id: str, account_id: str, label: str, currency: str) -> dict:
        """Open a new account owned by the logged-in user. Balance must start at zero.

        OBP requires user_id in the body; using the caller's own user_id means
        no special role is needed. The body shape changed across versions, so this
        tries the current shape first, then the v2.2 shape (CreateAccountJSONV220)
        that the v3.0.0 endpoint expects.
        """
        user_id = self.current_user(token)["user_id"]
        attempts = [
            (self.version, {
                "user_id": user_id, "label": label, "product_code": "CURRENT",
                "balance": {"currency": currency, "amount": "0"},
                "branch_id": "1234",
                "account_routings": [{"scheme": "OBP", "address": account_id}],
            }),
            ("v3.0.0", {
                "user_id": user_id, "label": label, "type": "CURRENT",
                "balance": {"currency": currency, "amount": "0"},
                "branch_id": "1234",
                "account_routing": {"scheme": "OBP", "address": account_id},
            }),
        ]
        errors: list[str] = []
        for version, body in attempts:
            url = f"{self.base_url}/obp/{version}/banks/{bank_id}/accounts/{account_id}"
            r = httpx.put(
                url,
                headers={"Authorization": f'DirectLogin token="{token}"', "Content-Type": "application/json"},
                json=body,
                timeout=self.timeout,
            )
            if r.status_code < 300:
                return r.json() if r.content else {}
            errors.append(f"[{version}] HTTP {r.status_code}: {r.text[:200]}")
        # Report every attempt, not just the last, so the right shape is easy to spot.
        raise OBPError(400, " | ".join(errors), f"/banks/{bank_id}/accounts/{account_id}")

    def my_customers(self, token: str, bank_id: str) -> list[dict]:
        """Get My Customers at Bank: the customer record(s) linked to this user."""
        return self.get(token, f"/banks/{bank_id}/my/customers").get("customers", [])

    # ---------- the customer's own profile data (no staff role needed) ----------
    # OBP keeps customer records (and their address) for bank staff only, but every
    # user can keep "personal user attributes" on their own login. The demo stores
    # the address there, so a change really is saved at the bank and read back from it.

    def my_user_attributes(self, token: str) -> list[dict]:
        """Get My Personal User Attributes: [{user_attribute_id, name, type, value, insert_date}]."""
        return self._request("GET", token, "/my/user/attributes", version="v4.0.0").get("user_attributes", [])

    def set_my_user_attribute(self, token: str, name: str, value: str) -> dict:
        """Create My Personal User Attribute, or update it if one with this name exists.
        OBP stores at most 255 characters."""
        body = {"name": name, "type": "STRING", "value": value[:255]}
        existing = next((a for a in self.my_user_attributes(token) if a.get("name") == name), None)
        if existing:
            return self._request("PUT", token, f"/my/user/attributes/{existing['user_attribute_id']}",
                                 json=body, version="v4.0.0")
        return self._request("POST", token, "/my/user/attributes", json=body, version="v4.0.0")

    def create_customer_message(
        self, token: str, bank_id: str, customer_id: str, message: str,
        from_department: str = "Digital Assistant", from_person: str = "Banking Agent",
    ) -> dict:
        """Create Customer Message: used for service requests (address, KYC, cheque book)."""
        return self._request(
            "POST",
            token,
            f"/banks/{bank_id}/customer/{customer_id}/messages",
            json={"message": message, "from_department": from_department, "from_person": from_person},
        )
