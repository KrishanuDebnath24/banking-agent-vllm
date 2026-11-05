"""Sending email: security codes (the bank servers) and sign-in codes (the web chat).

    EMAIL_BACKEND=console   print the email in the server's window or log (no setup; the default)
    EMAIL_BACKEND=smtp      really send it, through SMTP_HOST (Gmail: an app password in SMTP_PASSWORD)

A personal Gmail account sends at most 500 emails in 24 hours, so the web chat caps sign-in codes
per address, per internet address and per day (see api/signin.py).
"""

from __future__ import annotations

import json
import os
import smtplib
import ssl
from email.message import EmailMessage


def mask(email: str) -> str:
    """k***@gmail.com: enough for someone to recognise their own address."""
    name, _, domain = email.partition("@")
    return f"{name[:1]}***@{domain}" if domain else "***"


def backend() -> str:
    return os.environ.get("EMAIL_BACKEND", "console").strip().lower()


def send_email(to: str, subject: str, body: str) -> None:
    if backend() == "smtp":
        user = os.environ.get("SMTP_USER", "").strip()
        password = os.environ.get("SMTP_PASSWORD", "").replace(" ", "")  # Gmail shows app passwords in groups of 4
        host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        port = int(os.environ.get("SMTP_PORT", "587"))
        msg = EmailMessage()
        msg["From"] = os.environ.get("EMAIL_FROM") or user
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)
        context = ssl.create_default_context()
        if port == 465:
            with smtplib.SMTP_SSL(host, port, context=context, timeout=20) as s:
                s.login(user, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as s:
                s.starttls(context=context)
                s.login(user, password)
                s.send_message(msg)
        return
    # console: show the email in the server's window (no setup needed)
    print(f"\n----- email to {to} -----\nSubject: {subject}\n\n{body}\n--------------------------", flush=True)
    outbox = os.environ.get("OTP_OUTBOX")  # tests read codes from here
    if outbox:
        with open(outbox, "a", encoding="utf-8") as f:
            f.write(json.dumps({"to": to, "subject": subject, "body": body}) + "\n")
