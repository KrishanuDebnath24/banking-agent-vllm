"""Accounts MCP server: balances and transfers between the customer's own accounts.

Transfers from OTP_TRANSFER_THRESHOLD up need a security code the customer
confirmed for exactly this transfer (see mcp_servers/verification.py).
"""

from __future__ import annotations

import asyncio
import json

from mcp.server.fastmcp import Context, FastMCP

from mcp_servers.common import PORTS, OBPError, bank, customer, error, transport_security
from mcp_servers.verification import TRANSFER, VERIFICATION_REQUIRED, consume, needs_code

mcp = FastMCP("accounts", port=PORTS["accounts"], stateless_http=True, json_response=True,
              transport_security=transport_security(),
              instructions="Balances and transfers between the logged-in customer's own accounts.")


@mcp.tool()
async def list_my_accounts(ctx: Context) -> str:
    """List the customer's accounts ("current", "savings") with their balances."""
    try:
        cust = await customer(ctx)
        out = []
        for name, acc_id in cust.accounts.items():
            amount, cur = await bank(cust.obp.balance_amount, cust.token, cust.bank_id, acc_id)
            out.append({"account": name, "balance": round(amount, 2), "currency": cur or cust.currency})
        return json.dumps(out)
    except PermissionError as e:
        return error("not_authorised", detail=str(e))
    except OBPError as e:
        return error("bank_error", status=e.status)


@mcp.tool()
async def get_balance(account: str, ctx: Context) -> str:
    """Balance of one account.

    Args:
        account: "current" or "savings".
    """
    try:
        cust = await customer(ctx)
        acc_id = cust.resolve(account)
        if acc_id is None:
            return cust.denied()
        amount, cur = await bank(cust.obp.balance_amount, cust.token, cust.bank_id, acc_id)
        return json.dumps({"account": cust.name_of(acc_id), "balance": round(amount, 2), "currency": cur or cust.currency})
    except PermissionError as e:
        return error("not_authorised", detail=str(e))
    except OBPError as e:
        return error("bank_error", status=e.status)


@mcp.tool()
async def transfer_between_my_accounts(from_account: str, to_account: str, amount: float, ctx: Context,
                                       reference: str = "Transfer between my accounts") -> str:
    """Move money between the customer's own accounts, e.g. current -> savings.
    The customer is asked to confirm before this runs (and for larger amounts, to enter
    a security code emailed to them).

    Args:
        from_account: "current" or "savings".
        to_account: "current" or "savings" (must differ from from_account).
        amount: amount in GBP, greater than 0.
        reference: short reference shown on both statements.
    """
    try:
        cust = await customer(ctx)
        src, dst = cust.resolve(from_account), cust.resolve(to_account)
        if src is None or dst is None:
            return cust.denied()
        if src == dst:
            return error("same_account", message="Choose two different accounts.")
        amount = round(float(amount), 2)
        if amount <= 0:
            return error("invalid_amount", message="Amount must be greater than 0.")
        if amount > cust.obp.max_transfer:
            return error("over_limit", message=f"Online transfers are limited to £{cust.obp.max_transfer:.2f}.")
        balance, _ = await bank(cust.obp.balance_amount, cust.token, cust.bank_id, src)
        if amount > balance:
            return error("insufficient_funds", message=f"The {cust.name_of(src)} account has £{balance:.2f}.")
        action = {"from_account": from_account, "to_account": to_account, "amount": amount}
        if needs_code(TRANSFER, action, cust.obp.max_transfer) and not await asyncio.to_thread(consume, cust, TRANSFER, action):
            return json.dumps(VERIFICATION_REQUIRED)
        res = await bank(cust.obp.transfer, cust.token, cust.bank_id, src, dst, amount, cust.currency, reference[:100])
        new_src, _ = await bank(cust.obp.balance_amount, cust.token, cust.bank_id, src)
        new_dst, _ = await bank(cust.obp.balance_amount, cust.token, cust.bank_id, dst)
        return json.dumps({"status": res.get("status", "COMPLETED"), "amount": amount, "currency": cust.currency,
                           "from": cust.name_of(src), "to": cust.name_of(dst), "reference": reference[:100],
                           "new_balances": {cust.name_of(src): round(new_src, 2), cust.name_of(dst): round(new_dst, 2)}})
    except PermissionError as e:
        return error("not_authorised", detail=str(e))
    except OBPError as e:
        return error("bank_error", status=e.status, message="The bank did not complete the transfer.")


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
