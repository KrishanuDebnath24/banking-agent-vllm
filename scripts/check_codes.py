"""Check the security code email and the bank profile storage before chatting.

    python -m scripts.check_codes            # settings + a test email + profile storage
    python -m scripts.check_codes --no-email # skip the test email

Checks each piece in turn:
  1. the email settings in .env (EMAIL_BACKEND, OTP_EMAIL_TO, SMTP_*)
  2. a test email really arrives (or is printed, with EMAIL_BACKEND=console)
  3. customer A can save and read data on their own profile at OBP
     (where the address is stored), using a harmless DEMO_CHECK entry
"""

from __future__ import annotations

import argparse
import os
import smtplib
import sys
from datetime import datetime, timezone

from scripts._common import login, setup


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-email", action="store_true", help="skip sending the test email")
    args = ap.parse_args()
    client, _ = setup()

    from mcp_servers.verification import mask, send_email, transfer_threshold

    backend = os.environ.get("EMAIL_BACKEND", "console").strip().lower()
    to = os.environ.get("OTP_EMAIL_TO", "").strip()
    threshold = transfer_threshold()
    print("1. Settings")
    print(f"   Codes needed for: address changes"
          + (f" and transfers from £{threshold:.2f}" if threshold is not None else " (transfers: off)"))
    print(f"   EMAIL_BACKEND = {backend}")
    if backend not in ("console", "smtp"):
        print("   Use EMAIL_BACKEND=console (codes printed in the server window) or smtp (real email).")
        return 1
    if backend == "smtp":
        user = os.environ.get("SMTP_USER", "").strip()
        password = os.environ.get("SMTP_PASSWORD", "").replace(" ", "")
        host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        missing = [name for name, value in (("OTP_EMAIL_TO", to), ("SMTP_USER", user),
                                            ("SMTP_PASSWORD", password)) if not value]
        if missing:
            print(f"   MISSING in .env: {', '.join(missing)}")
            return 1
        print(f"   Codes go to {mask(to)}, sent by {mask(user)} via {host}:{os.environ.get('SMTP_PORT', '587')}")
        if "gmail" in host and len(password) != 16:
            print(f"   SMTP_PASSWORD has {len(password)} characters; a Gmail app password has 16. "
                  "Check you pasted the app password, not your normal password.")
    else:
        print("   Codes will be printed in the MCP server window (no email). For real email set EMAIL_BACKEND=smtp.")

    if not args.no_email:
        target = to or "(console)"
        print(f"\n2. Test email to {mask(target) if to else target} ...", end=" ", flush=True)
        try:
            send_email(to or "you@console", "Banking demo: test email",
                       "This is a test from scripts/check_codes.py. Security codes will arrive like this.")
            print("SENT" + (". Check your inbox (and spam folder)." if backend == "smtp" else ""))
        except smtplib.SMTPAuthenticationError:
            print("FAILED: the email server refused the login.\n"
                  "   For Gmail: turn on 2-Step Verification for the account, create an app password at\n"
                  "   https://myaccount.google.com/apppasswords and put it in SMTP_PASSWORD (not your normal password).")
            return 1
        except Exception as e:  # noqa: BLE001
            print(f"FAILED: {type(e).__name__}: {str(e)[:200]}")
            return 1

    print("\n3. Profile storage at the bank (customer A) ...", end=" ", flush=True)
    try:
        _, token = login(client, "A")
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        client.set_my_user_attribute(token, "DEMO_CHECK", stamp)
        attrs = {a.get("name"): a.get("value") for a in client.my_user_attributes(token)}
    except Exception as e:  # noqa: BLE001
        print(f"FAILED: {str(e)[:300]}\n   Paste this output: the address can't be saved this way.")
        return 1
    if attrs.get("DEMO_CHECK") != stamp:
        print("FAILED: saved, but not read back. Paste this output.")
        return 1
    print("OK (saved and read back)")
    print(f"   Address on file: {attrs.get('ADDRESS') or 'none yet'}")
    print("\nAll set. Restart  python -m mcp_servers.run  and then  python -m scripts.chat_multi")
    return 0


if __name__ == "__main__":
    sys.exit(main())
