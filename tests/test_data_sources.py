"""docs/data-sources.md lists the demo generator's constants. These tests keep the list true."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from demo_loader import module as generator

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "docs" / "data-sources.md").read_text(encoding="utf-8")
FLAT = " ".join(PAGE.split())  # sentences wrap across lines


def test_tiers_and_links():
    for heading in ("## Tier 0", "## Tier 1", "## Tier 2"):
        assert heading in PAGE, heading
    assert "(data-contract.md)" in PAGE
    assert "No fixture file holds data" in FLAT


def test_line_constants():
    for spec in generator.LINES:
        assert f"`{spec.name}`: {spec.slips} slips" in PAGE, spec.name
        assert f"ticket scale {spec.ticket_scale}" in PAGE, spec.name
        assert f"zero-balance share {spec.zero_balance_rate}" in PAGE, spec.name
        assert f"status ids from {spec.status_base + 1}" in PAGE, spec.name


def test_world_constants():
    assert f"grace period `{generator.GRACE}` days" in PAGE
    assert "seed `7`" in PAGE and "reference date `2026-03-31`" in PAGE


def test_outreach_weights():
    listed = ", ".join(f"{w:.2f}" for w in generator.RESULT_WEIGHTS)
    assert listed in PAGE


def test_payment_coefficients():
    source = inspect.getsource(generator.LineBuilder._outcome)
    line = next(x for x in source.splitlines() if x.strip().startswith("logit ="))
    coefficients = re.findall(r"\d+\.\d+", line)
    assert len(coefficients) == 4
    for number in coefficients:
        assert number in PAGE, number


def test_no_fitting_claim():
    assert "None comes from a public source" in FLAT
    assert "fitted to, tuned against or checked against any real dataset" in FLAT
