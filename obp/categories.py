"""Spending categories, and the payee words that put a transaction in each one.

One list for everything: the AI's instructions name these categories, the bank servers look the
words up here (so the AI never has to type them), and every demo transaction is named so that it
lands in exactly one category (tests/test_categories.py checks all four sets of demo data).

A transaction is in a category when its description contains one of the category's words,
ignoring capitals: "Tesco Stores 2041" contains "tesco", so it's groceries. Real banks use the
card network's merchant category codes (5411 = grocery store); the OBP sandbox and the pretend
banks don't have those, so names it is. Words are chosen so they can't clash: "tfl travel", not
"tfl" (inside "netflix"); "ee - mobile", not "ee" (inside "coffee"); "rent - ", not "rent" (inside
"current").
"""

from __future__ import annotations

# Money out.
SPENDING: dict[str, tuple[str, ...]] = {
    "groceries": ("tesco", "sainsbury", "lidl", "aldi", "co-op", "waitrose"),
    "eating out": ("deliveroo", "nando", "dishoom", "wagamama"),
    "coffee": ("costa", "pret a manger", "caffe nero"),
    "transport": ("tfl travel", "uber", "leap card", "national rail", "lime bike"),
    "subscriptions": ("netflix", "spotify", "adobe", "notion"),
    "bills": ("energy", "water", "vodafone", "ee - mobile", "broadband"),
    "gym": ("puregym", "flyefit"),
    "rent": ("rent - ",),
    "shopping": ("amazon", "boots", "argos"),
}

# Money in.
INCOME: dict[str, tuple[str, ...]] = {
    "salary": ("salary", "weekly pay"),
    "invoices": ("invoice",),
    "interest": ("interest",),
}

ALL = {**SPENDING, **INCOME}


def words_for(names: str) -> tuple[list[str], dict[str, tuple[str, ...]]]:
    """ "groceries, eating out" -> (["groceries", "eating out"], {name: words}). Unknown names raise
    ValueError listing the known ones."""
    wanted = [n.strip().lower() for n in (names or "").split(",") if n.strip()]
    unknown = [n for n in wanted if n not in ALL]
    if unknown:
        raise ValueError(f"unknown category {', '.join(unknown)}; known: {', '.join(ALL)}")
    return wanted, {n: ALL[n] for n in wanted}


def category_of(description: str, money_in: bool) -> str | None:
    """The category a transaction belongs to, or None."""
    text = description.lower()
    for name, words in (INCOME if money_in else SPENDING).items():
        if any(w in text for w in words):
            return name
    return None


def for_prompt() -> str:
    """The category names, for the AI's instructions (the words stay in code)."""
    return ("spending: " + ", ".join(SPENDING) + "; income: " + ", ".join(INCOME))
