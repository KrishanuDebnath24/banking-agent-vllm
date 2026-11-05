"""Read-only banking tools for the phase 1 agent.

Each tool:
  - gets the customer's token and allowed accounts from the run-time context,
  - refuses any account that isn't the customer's (authorisation layer 2,
    on top of OBP's own checks),
  - does arithmetic in code, so totals never depend on the model's maths.

In phase 2 these move behind the Accounts and Transactions MCP servers.
"""

from __future__ import annotations

import json
from collections import defaultdict

from langchain.tools import ToolRuntime, tool

from agents.context import CustomerContext
from obp.client import OBPError

DENIED = "Access denied: that account does not belong to this customer. Allowed accounts: {names}."


def _deny(ctx: CustomerContext) -> str:
    return DENIED.format(names=", ".join(ctx.accounts))


def _fetch(ctx: CustomerContext, account_id: str) -> list[dict]:
    """Transactions as simple rows, newest first, hiding pre-demo noise."""
    rows = []
    for t in ctx.client.transactions(ctx.token, ctx.bank_id, account_id):
        d = t.get("details", {})
        completed = d.get("completed") or d.get("posted") or ""
        if ctx.show_since and completed and completed[:10] < ctx.show_since:
            continue
        rows.append({
            "date": completed[:10],
            "amount": float(d.get("value", {}).get("amount", 0) or 0),
            "description": d.get("description") or "",
            "balance_after": d.get("new_balance", {}).get("amount"),
        })
    rows.sort(key=lambda r: r["date"], reverse=True)
    return rows


def _matches(desc: str, search: str | None) -> bool:
    if not search:
        return True
    terms = [s.strip().lower() for s in search.split(",") if s.strip()]
    return any(term in desc.lower() for term in terms)


@tool
def list_my_accounts(runtime: ToolRuntime[CustomerContext]) -> str:
    """List the customer's accounts with their current balances."""
    ctx = runtime.context
    out = []
    for name, acc_id in ctx.accounts.items():
        try:
            amount, currency = ctx.client.balance_amount(ctx.token, ctx.bank_id, acc_id)
        except OBPError as e:
            return f"Bank error while reading {name}: HTTP {e.status}"
        out.append({"account": name, "balance": round(amount, 2), "currency": currency or ctx.currency})
    return json.dumps(out)


@tool
def get_balance(account: str, runtime: ToolRuntime[CustomerContext]) -> str:
    """Get the balance of one of the customer's accounts.

    Args:
        account: account name, e.g. "current" or "savings".
    """
    ctx = runtime.context
    acc_id = ctx.resolve(account)
    if acc_id is None:
        return _deny(ctx)
    amount, currency = ctx.client.balance_amount(ctx.token, ctx.bank_id, acc_id)
    return json.dumps({"account": account, "balance": round(amount, 2), "currency": currency or ctx.currency})


@tool
def get_transactions(account: str, runtime: ToolRuntime[CustomerContext],
                     search: str | None = None, limit: int = 30) -> str:
    """List recent transactions on one of the customer's accounts, newest first.
    Negative amounts are money out, positive are money in.

    Args:
        account: account name, e.g. "current" or "savings".
        search: optional comma-separated words to match in the description, e.g. "tesco,lidl".
        limit: maximum number of transactions to return (default 30).
    """
    ctx = runtime.context
    acc_id = ctx.resolve(account)
    if acc_id is None:
        return _deny(ctx)
    rows = [r for r in _fetch(ctx, acc_id) if _matches(r["description"], search)]
    return json.dumps({"account": account, "currency": ctx.currency,
                       "count": len(rows), "transactions": rows[: max(1, min(limit, 100))]})


@tool
def summarize_spending(account: str, runtime: ToolRuntime[CustomerContext], search: str | None = None) -> str:
    """Totals for one account: money in, money out, and spending per payee.
    Use this for any "how much did I spend / earn" question instead of adding numbers yourself.

    Args:
        account: account name, e.g. "current" or "savings".
        search: optional comma-separated words to restrict to matching payees,
            e.g. "tesco,sainsbury,lidl" for groceries.
    """
    ctx = runtime.context
    acc_id = ctx.resolve(account)
    if acc_id is None:
        return _deny(ctx)
    rows = [r for r in _fetch(ctx, acc_id) if _matches(r["description"], search)]
    money_in = sum(r["amount"] for r in rows if r["amount"] > 0)
    money_out = -sum(r["amount"] for r in rows if r["amount"] < 0)
    by_payee: dict[str, float] = defaultdict(float)
    for r in rows:
        if r["amount"] < 0:
            by_payee[r["description"]] += -r["amount"]
    top = sorted(by_payee.items(), key=lambda kv: kv[1], reverse=True)
    return json.dumps({
        "account": account, "currency": ctx.currency, "filter": search or "none",
        "transactions": len(rows),
        "money_in": round(money_in, 2), "money_out": round(money_out, 2),
        "spending_by_payee": [{"payee": p, "total": round(v, 2)} for p, v in top],
    })


ALL_TOOLS = [list_my_accounts, get_balance, get_transactions, summarize_spending]
