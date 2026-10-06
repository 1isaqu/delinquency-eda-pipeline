"""Text normalization shared by the diagnostics and the cleaning rules."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(value: str) -> str:
    """Case, accents and spacing folded away, for comparing values that differ only in form."""
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped.strip().casefold())
