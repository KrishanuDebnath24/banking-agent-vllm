"""What a pending action looks like on the confirmation slip: a title, a few rows, a button label."""

from __future__ import annotations


def _money(value) -> str:
    try:
        return f"£{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _name(value) -> str:
    text = str(value or "").strip()
    return text[:1].upper() + text[1:]


def slip(action: dict) -> dict:
    name, a = action.get("name", ""), action.get("args", {}) or {}
    if name == "transfer_between_my_accounts":
        rows = [["From", _name(a.get("from_account"))], ["To", _name(a.get("to_account"))],
                ["Amount", _money(a.get("amount"))]]
        if a.get("reference"):
            rows.append(["Reference", str(a["reference"])])
        return {"title": "Transfer between your accounts", "rows": rows, "approve_label": "Approve transfer"}
    if name == "change_address":
        return {"title": "Change your address", "rows": [["New address", str(a.get("new_address", ""))]],
                "approve_label": "Change address"}
    if name == "request_cheque_book":
        return {"title": "Order a cheque book", "rows": [["Account", _name(a.get("account"))]],
                "approve_label": "Order cheque book"}
    if name == "request_kyc_update":
        return {"title": "Update your identity details", "rows": [["Change", str(a.get("update", ""))]],
                "approve_label": "Send update"}
    return {"title": name.replace("_", " ").capitalize(), "rows": [[k, str(v)] for k, v in a.items()],
            "approve_label": "Approve"}


SPECIALISTS = {"accounts-agent": "Accounts specialist", "transactions-agent": "Transactions specialist",
               "service-agent": "Service specialist", "chart-agent": "Chart specialist"}
