"""Inline builders for the offline stages: a dataset from a few compact, invented rows.

Every value is made up. Rows go through ``normalize_slip`` exactly as extracted rows do, so the
derived columns (bucket, days overdue, days to due) come from the production rule.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any

from delinquency_eda.dataset import Dataset, DatasetMeta, normalize_slip

AS_OF = date(2026, 3, 31)
GRACE = 3

# A dataset is only written with an admission record, and only the ``base`` stage earns a real
# one. The offline stages are tested on rows built by hand (some deliberately dirty, to give the
# diagnostics something to find), so those datasets carry this record. It says so in the file.
FIXTURE_ADMISSION: dict[str, Any] = {
    "contract_version": 1,
    "package_version": "fixture",
    "passed": True,
    "rules": [],
    "ledger": [],
    "queries": [],
    "note": "hand-built test fixture, not judged against the entry contract",
}


def raw(i: int = 1, **overrides: Any) -> dict[str, Any]:
    """One raw slip. Defaults: unpaid, due 2026-03-10, a 100.00 slip on a 500.00 debt."""
    row: dict[str, Any] = {
        "record_id": i,
        "agreement_id": i,
        "portfolio_id": 1,
        "debtor_id": i,
        "person_key": f"pk_{i}",
        "status_id": 21,
        "status_label": "ACTIVE",
        "first_installment_value": 100.0,
        "total_value": 100.0,
        "total_installments": 1,
        "plan_installments": 1,
        "original_debt": 500.0,
        "agent_id": "a1",
        "agent_name": "Agent 01",
        "agent_login": "agent_01",
        "portfolio_label": "Portfolio A-01",
        "agreement_date": "2026-03-01",
        "due_date": "2026-03-10",
        "paid_amount": None,
        "paid_date": None,
    }
    row.update(overrides)
    return row


def dataset(
    rows: Iterable[Mapping[str, Any]],
    *,
    as_of: date = AS_OF,
    grace: int = GRACE,
    debtors: Iterable[Mapping[str, Any]] = (),
    contacts: Iterable[Mapping[str, Any]] = (),
    breaks: Iterable[Mapping[str, Any]] = (),
    lines: tuple[str, ...] = ("auto",),
) -> Dataset:
    """A dataset whose base rows carry an optional ``product_line`` override."""
    base = []
    for r in rows:
        row = dict(r)
        line = row.pop("product_line", lines[0])
        base.append(normalize_slip(row, line, as_of=as_of, grace_days=grace))
    meta = DatasetMeta(
        period_from="2026-01-01",
        period_to=as_of.isoformat(),
        as_of=as_of.isoformat(),
        grace_days=grace,
        product_lines=list(lines),
        generated_status_labels=["ACTIVE", "BROKEN", "PAID"],
        broken_status_labels=["BROKEN", "AUTO_BROKEN"],
        manual_break_label="BROKEN",
        excluded_agent_patterns=["system%"],
    )
    return Dataset(
        meta=meta,
        base=base,
        debtors=[dict(d) for d in debtors],
        contacts=[dict(c) for c in contacts],
        breaks=[dict(b) for b in breaks],
        admission=dict(FIXTURE_ADMISSION),
    )


def paid_on_time(i: int, **overrides: Any) -> dict[str, Any]:
    """A slip due 2026-03-10 and paid the next day, inside the grace period."""
    return raw(i, paid_amount=100.0, paid_date="2026-03-11", **overrides)
