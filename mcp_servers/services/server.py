"""Service MCP server: customer profile, address changes and service requests.

Address changes take effect straight away: once the customer has confirmed the
change and entered the security code emailed to them, the new address is saved
at the bank as a personal attribute on their own login (OBP keeps the customer
record itself for staff only), and get_my_profile reads it back from there.

Cheque books and identity (KYC) updates need bank staff, so they are logged as
service requests (with a reference number) for the back office, like a real
bank's case system. The bank is also sent a customer message for each one.

Requests are kept in the service_requests table (Postgres if DATABASE_URL is set, else a SQLite
file). Requests logged before that, in service_requests.jsonl, are copied in once on first use.
Visitors' requests are filed under guest:<id> and deleted along with their pretend bank.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP

from mcp_servers.common import PORTS, ROOT, OBPError, bank, customer, error, transport_security
from mcp_servers.verification import ADDRESS, ADDRESS_MAX, VERIFICATION_REQUIRED, consume
from storage import REQUESTS

mcp = FastMCP("service", port=PORTS["service"], stateless_http=True, json_response=True,
              transport_security=transport_security(),
              instructions="Customer profile, address changes and service requests (cheque book, KYC update).")

LEGACY_FILE = Path(os.environ.get("SERVICE_REQUESTS_FILE", ROOT / "service_requests.jsonl"))  # before phase 4
ADDRESS_ATTRIBUTE = "ADDRESS"   # name of the personal user attribute that holds the address
NEXT_STEPS = {
    "cheque_book": "A new cheque book will be posted to your registered address within 7 working days.",
    "kyc_update": "Our team will review this and contact you if we need documents.",
}


def _new_record(cust, kind: str, details: dict, status: str) -> dict:
    return {"reference": "SR-" + secrets.token_hex(3).upper(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "customer": cust.label, "username": cust.records_key, "type": kind, "details": details,
            "status": status}


COLUMNS = ("reference", "created_at", "created_ts", "customer", "username", "type", "details", "status",
           "error", "bank_message")
_INSERT = (f"INSERT INTO service_requests ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})"
           " ON CONFLICT (reference) DO NOTHING")
_legacy = {"checked": False, "lock": threading.Lock()}


def _row(record: dict, created_ts: float) -> tuple:
    return (record["reference"], record["created_at"], created_ts, record["customer"], record["username"],
            record["type"], json.dumps(record["details"]), record["status"], record.get("error"),
            record.get("bank_message"))


def _import_legacy(db) -> None:
    """Copy requests from service_requests.jsonl into an empty table, once, so history isn't lost."""
    with _legacy["lock"]:
        if _legacy["checked"]:
            return
        _legacy["checked"] = True
        if not LEGACY_FILE.exists() or db.execute("SELECT COUNT(*) FROM service_requests").fetchone()[0]:
            return
        rows = []
        for n, line in enumerate(LEGACY_FILE.read_text(encoding="utf-8").splitlines()):
            try:
                r = json.loads(line)
                ts = datetime.fromisoformat(r["created_at"]).timestamp() + n / 1000  # keep the file's order
                rows.append(_row(r, ts))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        db.executemany(_INSERT, rows)


def _save(record: dict) -> None:
    with REQUESTS.connect() as db:
        _import_legacy(db)
        db.execute(_INSERT, _row(record, time.time()))


def requests_of(username: str) -> list[dict]:
    """This customer's requests, newest first."""
    with REQUESTS.connect() as db:
        _import_legacy(db)
        rows = db.execute("SELECT reference, created_at, type, details, status FROM service_requests"
                          " WHERE username=? ORDER BY created_ts DESC", (username,)).fetchall()
    return [{"reference": ref, "created_at": at, "type": kind, "details": json.loads(details), "status": status}
            for ref, at, kind, details, status in rows]


async def _log(record: dict) -> None:
    await asyncio.to_thread(_save, record)  # the database call mustn't hold up the other servers


async def _customer_id(cust) -> str | None:
    try:
        records = await bank(cust.obp.my_customers, cust.token, cust.bank_id)
        return records[0].get("customer_id") if records else None
    except OBPError:
        return None


async def _message_bank(cust, text: str) -> str:
    """Send a customer message to the bank; returns what happened, for the log."""
    cid = await _customer_id(cust)
    if not cid:
        return "not sent (no customer record at the bank)"
    try:
        await bank(cust.obp.create_customer_message, cust.token, cust.bank_id, cid, text)
        return "sent"
    except OBPError as e:
        return f"not sent (HTTP {e.status})"


async def _create(ctx, kind: str, details: dict) -> str:
    cust = await customer(ctx)
    record = _new_record(cust, kind, details, "received")
    record["bank_message"] = await _message_bank(
        cust, f"[{record['reference']}] Service request: {kind}: {json.dumps(details)}")
    await _log(record)
    return json.dumps({"reference": record["reference"], "type": kind, "status": "received",
                       "next_steps": NEXT_STEPS[kind]})


def _guard(fn):
    """Turn identity and bank errors into JSON errors the agent can explain."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except PermissionError as e:
            return error("not_authorised", detail=str(e))
        except OBPError as e:
            return error("bank_error", status=e.status)
    return wrapper


@mcp.tool()
@_guard
async def get_my_profile(ctx: Context) -> str:
    """The customer's profile held by the bank: name, email, mobile, customer number, KYC status,
    and the address on file."""
    cust = await customer(ctx)
    try:
        records = await bank(cust.obp.my_customers, cust.token, cust.bank_id)
    except OBPError:
        records = []
    try:
        attrs = await bank(cust.obp.my_user_attributes, cust.token)
        saved = next((a for a in attrs if a.get("name") == ADDRESS_ATTRIBUTE), None)
        address = saved.get("value") if saved else None
    except OBPError:
        address = None
    # The address is saved on the customer's own login at the bank, so it doesn't depend on
    # the staff-only customer record (which the sandbox doesn't link to every demo login).
    out = {"username": cust.username, "address": address or "no address saved yet"}
    if records:
        keys = ("legal_name", "email", "mobile_phone_number", "customer_number", "kyc_status")
        out["details"] = {k: records[0].get(k) for k in keys}
    else:
        out["details"] = "not available for this demo login (this doesn't affect the address)"
    return json.dumps(out)


@mcp.tool()
@_guard
async def change_address(new_address: str, ctx: Context) -> str:
    """Change the customer's address on their bank profile. Takes effect straight away.
    The customer is asked to confirm and to enter a security code emailed to them first.

    Args:
        new_address: the full new address including postcode.
    """
    address = " ".join(new_address.split())
    if len(address) < 10:
        return error("invalid_input", message="Please give the full address including postcode.")
    if len(address) > ADDRESS_MAX:
        return error("invalid_input", message=f"Please shorten the address to {ADDRESS_MAX} characters.")
    cust = await customer(ctx)
    if not await asyncio.to_thread(consume, cust, ADDRESS, {"new_address": new_address}):
        return json.dumps(VERIFICATION_REQUIRED)

    record = _new_record(cust, "address_change", {"new_address": address}, "completed")
    try:
        await bank(cust.obp.set_my_user_attribute, cust.token, ADDRESS_ATTRIBUTE, address)
    except OBPError as e:
        record.update(status="failed", error=f"HTTP {e.status}")
        await _log(record)
        return error("bank_error", status=e.status, message="The bank did not save the new address; nothing changed.")
    # Read it back, so "saved" means the bank really holds it.
    try:
        attrs = await bank(cust.obp.my_user_attributes, cust.token)
    except OBPError:
        attrs = []
    on_file = next((a.get("value") for a in attrs if a.get("name") == ADDRESS_ATTRIBUTE), None)
    if on_file != address:
        record.update(status="failed", error="not found after saving")
        await _log(record)
        return error("bank_error", message="The bank did not confirm the new address; please try again.")
    # Banks tell the customer whenever contact details change, in case it wasn't them.
    record["bank_message"] = await _message_bank(
        cust, f"[{record['reference']}] Your address was changed to: {address}. If this wasn't you, contact us now.")
    await _log(record)
    return json.dumps({"reference": record["reference"], "status": "completed",
                       "address_on_file": on_file, "saved_to": "the customer's profile at the bank"})


@mcp.tool()
@_guard
async def request_cheque_book(account: str, ctx: Context) -> str:
    """Order a new cheque book for one of the customer's accounts. The customer is asked to confirm first.

    Args:
        account: "current" or "savings".
    """
    cust = await customer(ctx)
    acc_id = cust.resolve(account)
    if acc_id is None:
        return cust.denied()
    return await _create(ctx, "cheque_book", {"account": cust.name_of(acc_id)})


@mcp.tool()
@_guard
async def request_kyc_update(update: str, ctx: Context) -> str:
    """Tell the bank about a change to identity details (e.g. new surname, new passport).
    The customer is asked to confirm first.

    Args:
        update: what has changed, in the customer's words.
    """
    if len(update.strip()) < 5:
        return error("invalid_input", message="Please describe what has changed.")
    return await _create(ctx, "kyc_update", {"update": update.strip()[:500]})


@mcp.tool()
@_guard
async def list_my_service_requests(ctx: Context) -> str:
    """The customer's address changes and service requests so far, newest first, with reference and status."""
    cust = await customer(ctx)
    return json.dumps({"requests": await asyncio.to_thread(requests_of, cust.records_key)})


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
