"""Matching a payee the customer named against transaction descriptions.

The customer (and so the model) writes payees their own way: "Leap Card top-ups" for the bank's
"Leap Card top-up", "Sainsburys" for "Sainsbury's 0612". A plain substring search finds nothing and
the answer says £0.00, so a search term also matches when every word of it is in the description,
ignoring case, punctuation, apostrophes and a plural "s". A word of four letters or more also
matches the start of a longer word ("spot" finds "Spotify").
"""

from __future__ import annotations

import re


def _words(text: str) -> list[str]:
    text = re.sub(r"['’]s\b", "s", text.lower())          # sainsbury's -> sainsburys
    return [w[:-1] if len(w) > 2 and w.endswith("s") and not w.endswith("ss") else w
            for w in re.findall(r"[a-z0-9]+", text)]


def payee_matches(description: str, term: str) -> bool:
    """True if `term` (what the customer called the payee) matches this description."""
    term = term.strip().lower()
    if not term:
        return True
    if term in description.lower():
        return True
    wanted, have = _words(term), _words(description)
    return bool(wanted) and all(any(h == w or (len(w) >= 4 and h.startswith(w)) for h in have) for w in wanted)
