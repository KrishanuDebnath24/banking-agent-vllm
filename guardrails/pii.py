"""Deterministic PII detection and masking (no network, no model).

Used in three places, each with its own policy:
  INPUT_BLOCK  never let these reach the model, the session store or the traces
               (card numbers, CVVs, PINs, passwords, one-time security codes)
  INPUT_MASK   not needed for any banking operation here, so the model gets a placeholder
  OUTPUT_MASK  never shown in a reply (full card/IBAN/account numbers, internal ids, tokens)
  TRACE        everything above plus postcodes; the trace anonymizer also hides address fields

The customer's own address is deliberately NOT masked on its way to the model: the
change-of-address tool needs it. It is hidden from traces instead.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

CARD, CVV, PIN, PASSWORD, SECURITY_CODE = "card_number", "cvv", "pin", "password", "security_code"
EMAIL, PHONE, IBAN, SORT_CODE, ACCOUNT_NUMBER = "email", "phone", "iban", "sort_code", "account_number"
TOKEN, ACCOUNT_ID, POSTCODE, STREET = "token", "account_id", "postcode", "street"

_PATTERNS: dict[str, re.Pattern] = {
    # Not part of a longer number or an amount like 1,234.56 (a separator followed by a digit).
    CARD: re.compile(r"(?<!\d)(?<!\d[.,])(?:\d[ -]?){12,18}\d(?!\d|[.,]\d)"),
    CVV: re.compile(r"(?i)\b(?:cvv2?|cvc2?|card security (?:code|number))\b\D{0,12}\d{3,4}\b"),
    PIN: re.compile(r"(?i)\bpin\b(?:\s*(?:number|code))?\s*(?:is|was|:|=|-)?\s*\d{4,6}\b"),
    PASSWORD: re.compile(r"(?i)\b(?:password|passcode|passwd|pwd)\b\s*(?:is|was|:|=)\s*\S+"),
    SECURITY_CODE: re.compile(r"(?i)\b(?:security\s+code|one[- ]time\s+(?:code|password)|otp|code)\b\D{0,20}\b\d{6}\b"),
    EMAIL: re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    PHONE: re.compile(r"(?<![\w+])(?:\+44|\+353|0)\s?\d(?:[\s-]?\d){7,10}\b"),
    IBAN: re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]){11,30}\b"),
    SORT_CODE: re.compile(r"\b\d{2}-\d{2}-\d{2}\b"),
    # Only the digits after "account number"/"account no", so the words stay readable.
    ACCOUNT_NUMBER: re.compile(r"(?i)\baccount\s+(?:number|no\.?)\s*[:#]?\s*(?P<v>\d{8})\b"),
    TOKEN: re.compile(r"\beyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}\b|\bfake-token-[A-Z]\b"),
    # OBP account ids look like 4f9wyhc8k6soyyjfft6gsymfb: long, lowercase, letters and digits mixed.
    ACCOUNT_ID: re.compile(r"\b(?=[a-z0-9]*\d)(?=[a-z0-9]*[a-z])[a-z0-9]{20,40}\b"),
    POSTCODE: re.compile(r"\b(?:[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}|[AC-FHKNPRTV-Y]\d{2}\s?[AC-FHKNPRTV-Y\d]{4})\b"),
    STREET: re.compile(r"\b\d{1,4}[A-Za-z]?,?\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|"
                       r"Close|Court|Ct|Park|Terrace|Way|Place|Square|Crescent|Grove|Gardens|Hill|Row|View)\b\.?"),
}

INPUT_BLOCK = (CARD, CVV, PIN, PASSWORD, SECURITY_CODE)
INPUT_MASK = (EMAIL, PHONE, IBAN, SORT_CODE, ACCOUNT_NUMBER, TOKEN)
OUTPUT_MASK = (CARD, IBAN, SORT_CODE, ACCOUNT_NUMBER, TOKEN, ACCOUNT_ID)
CONTACT = (EMAIL, PHONE)
TRACE = (CARD, CVV, PIN, PASSWORD, SECURITY_CODE, EMAIL, PHONE, IBAN, SORT_CODE, ACCOUNT_NUMBER, TOKEN, POSTCODE,
         STREET)


@dataclass(frozen=True)
class Match:
    kind: str
    start: int
    end: int
    value: str


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _iban_ok(value: str) -> bool:
    s = value.replace(" ", "").upper()
    if not 15 <= len(s) <= 34:
        return False
    rearranged = s[4:] + s[:4]
    try:
        return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1
    except ValueError:
        return False


def _valid(kind: str, value: str) -> bool:
    if kind == CARD:
        digits = re.sub(r"\D", "", value)
        return 13 <= len(digits) <= 19 and _luhn_ok(digits)
    if kind == IBAN:
        return _iban_ok(value)
    return True


def find(text: str, kinds=TRACE) -> list[Match]:
    """All non-overlapping matches of the given kinds, earliest and longest first."""
    found: list[Match] = []
    for kind in kinds:
        pattern = _PATTERNS[kind]
        for m in pattern.finditer(text or ""):
            start, end = m.span("v") if "v" in pattern.groupindex else m.span()
            if _valid(kind, text[start:end]):
                found.append(Match(kind, start, end, text[start:end]))
    found.sort(key=lambda m: (m.start, -(m.end - m.start)))
    out: list[Match] = []
    for m in found:
        if not out or m.start >= out[-1].end:
            out.append(m)
    return out


def _last4(value: str) -> str:
    chars = re.sub(r"[\s-]", "", value)
    return chars[-4:]


def placeholder(m: Match, style: str) -> str:
    """style "label": [email] for the model and traces; "partial": •••• 1234 for customers."""
    if style == "partial":
        if m.kind == EMAIL:
            name, _, domain = m.value.partition("@")
            return f"{name[:1]}***@{domain}"
        if m.kind == PHONE:
            return "••• " + _last4(m.value)[-3:]
        if m.kind in (CARD, IBAN, ACCOUNT_ID, ACCOUNT_NUMBER):
            return "••••" + _last4(m.value)
        if m.kind == SORT_CODE:
            return "••-••-" + m.value[-2:]
    return f"[{m.kind.replace('_', ' ')}]"


def redact(text: str, kinds, style: str = "label") -> tuple[str, list[Match]]:
    """Replace every match; returns the new text and what was found."""
    matches = find(text, kinds)
    if not matches:
        return text, []
    parts, pos = [], 0
    for m in matches:
        parts.append(text[pos:m.start])
        parts.append(placeholder(m, style))
        pos = m.end
    parts.append(text[pos:])
    return "".join(parts), matches


def _hide_addresses(data):
    if isinstance(data, dict):
        return {k: ("[address]" if "address" in str(k).lower() and isinstance(v, str) and len(v) > 6
                    else _hide_addresses(v)) for k, v in data.items()}
    if isinstance(data, list):
        return [_hide_addresses(v) for v in data]
    return data


def trace_replacer(text: str, path: list) -> str:
    """For the LangSmith anonymizer: hide address fields entirely and mask PII everywhere else.

    Tool results arrive as JSON strings, so address fields inside them are hidden too.
    Addresses typed in free text are caught by the street and postcode patterns.
    """
    keys = [str(p).lower() for p in path if isinstance(p, str)]
    if any("address" in k for k in keys) and len(text.strip()) > 6:
        return "[address]"
    stripped = text.lstrip()
    if stripped[:1] in "{[" and "address" in text:
        try:
            text = json.dumps(_hide_addresses(json.loads(text)))
        except ValueError:
            pass
    return redact(text, TRACE)[0]
