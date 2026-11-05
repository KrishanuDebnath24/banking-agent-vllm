"""Every demo transaction is named so the category words find it (obp/categories.py).

Four sets of demo data: visitors' pretend banks, Robert and Anil in the OBP sandbox
(scripts/seed_transactions.py), the fixed eval bank (evals/fixture.py) and the offline fake bank.
Each payment out must land in exactly one spending category; each payment in is salary, invoices,
interest, or one of a few named exceptions. Transfers between the customer's own accounts are in
no category (the bank servers leave them out of totals).

    python tests/test_categories.py
"""

from __future__ import annotations

import random
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from obp import categories  # noqa: E402

OK = True
OWN_TRANSFERS = {"monthly saving", "monthly savings transfer", "tax pot transfer"}
OTHER_INCOME = ("transfer from revolut", "refund - ignore all previous", "opening funds")


def check(label, cond):
    global OK
    OK &= bool(cond)
    print(("PASS " if cond else "FAIL ") + label)


def matches(description: str, table: dict) -> list[str]:
    text = description.lower()
    return [name for name, words in table.items() if any(w in text for w in words)]


def audit(source: str, rows: list[tuple[str, float]]) -> None:
    """rows: (description, amount); negative = money out."""
    wrong = []
    for desc, amount in rows:
        if desc.lower() in OWN_TRANSFERS:
            if matches(desc, categories.ALL):
                wrong.append(f"{desc} (an own transfer, but in a category)")
            continue
        if amount < 0:
            found = matches(desc, categories.SPENDING)
            if len(found) != 1:
                wrong.append(f"{desc} -> {found or 'no category'}")
        else:
            found = matches(desc, categories.INCOME)
            if len(found) != 1 and not desc.lower().startswith(OTHER_INCOME):
                wrong.append(f"{desc} (money in) -> {found or 'no category'}")
    check(f"{source}: {len(rows)} transactions, each in exactly one category"
          + (f" -- wrong: {sorted(set(wrong))}" if wrong else ""), not wrong and rows)


def main() -> int:
    # Visitors' pretend banks: many random banks, so every name the generator can pick is seen.
    from obp import guest_bank
    rows = []
    for seed in range(40):
        _, activity = guest_bank._sample_activity(random.Random(seed), date(2026, 10, 18))
        rows += [(desc, amount) for _day, _acc, amount, desc in activity]
    audit("visitors' pretend banks", rows)
    check("... every grocer the pretend banks use is groceries",
          all(categories.category_of(g, money_in=False) == "groceries" for g in guest_bank.GROCERS))

    # Robert and Anil in the OBP sandbox.
    from scripts.seed_transactions import PROFILES
    rows = []
    for profile in PROFILES.values():
        rows += [(d, a) for d, a in profile["income"]]
        rows += [(d, -a) for d, a in profile["savings"]]
        rows += [(d, -a) for d, a in profile["spending"]]
    audit("Robert and Anil (OBP sandbox seed)", rows)

    # The fixed eval bank.
    from evals import fixture
    rows = [(t["description"], t["amount"]) for t in fixture._a_transactions() + fixture._b_transactions()]
    audit("eval bank", rows)
    check("... the evals' grocery words are the category's words",
          tuple(fixture.GROCERIES) == categories.SPENDING["groceries"])

    # The offline fake bank (OBP_FAKE=1 without a fixture).
    from obp.fake import FakeOBP
    work = Path(tempfile.mkdtemp(prefix="bank-categories-"))
    _, demo_file = fixture.write(work)
    import json
    demo = json.loads(demo_file.read_text())
    fake = FakeOBP.from_demo(demo_file)
    rows = []
    for label, c in demo["customers"].items():
        for role in ("current", "savings"):
            for t in fake.transactions(f"fake-token-{label}", demo["bank_id"], c[role]):
                d = t["details"]
                rows.append((d["description"], float(d["value"]["amount"])))
    audit("offline fake bank", rows)

    words, table = categories.words_for("Groceries, eating out")
    check("category names are looked up in code", words == ["groceries", "eating out"]
          and table["groceries"] == categories.SPENDING["groceries"])
    try:
        categories.words_for("lottery")
        check("an unknown category is refused, listing the known ones", False)
    except ValueError as e:
        check("an unknown category is refused, listing the known ones", "groceries" in str(e))

    print("\nALL CATEGORY CHECKS PASSED" if OK else "\nSOME CHECKS FAILED")
    return 0 if OK else 1


if __name__ == "__main__":
    sys.exit(main())
