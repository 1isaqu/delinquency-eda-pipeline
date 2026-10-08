"""Row-level features built from the dataset bundle.

Everything here is known at (or before) the moment the agreement was made, except the follow-up
counts, which describe what happened *between* the agreement and its due date. Follow-up columns
are explanatory (stage 03) and are never model features (stage 10): the operation calls the people
who are already slipping, so follow-up and non-payment are entangled by reverse causality.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from delinquency_eda.config import Settings
from delinquency_eda.dataset import Dataset
from delinquency_eda.derive import OVERDUE_UNPAID

UNKNOWN = "(unknown)"
NO_PORTFOLIO = "(no portfolio)"

FOLLOWUP_NONE = "0 - no follow-up after the agreement"
FOLLOWUP_CPC = "CPC - right party reached"
FOLLOWUP_ANSWERED = "Answered, not the right party"
FOLLOWUP_ATTEMPTS = "Attempts only, nobody answered"
FOLLOWUP_LEVELS = (FOLLOWUP_NONE, FOLLOWUP_CPC, FOLLOWUP_ANSWERED, FOLLOWUP_ATTEMPTS)

RATIO_OK = "ratio"
RATIO_ZERO = "zero_balance"
RATIO_NO_LINK = "no_debt_link"
RATIO_NEGATIVE = "negative_balance"


def followup_level(contacts_after: int, cpc_after: int, answered_after: int) -> str:
    """Strongest contact the debtor had after the agreement day."""
    if contacts_after == 0:
        return FOLLOWUP_NONE
    if cpc_after > 0:
        return FOLLOWUP_CPC
    if answered_after > 0:
        return FOLLOWUP_ANSWERED
    return FOLLOWUP_ATTEMPTS


def ratio_state(original_debt: float | None) -> str:
    """Why a first-installment / original-debt ratio is, or is not, defined.

    ``zero_balance`` is a real value (the debt is linked but its balance is zero), not a missing
    one. Testing ``if original_debt:`` conflates the two, so the states are kept apart.
    """
    if original_debt is None or (isinstance(original_debt, float) and np.isnan(original_debt)):
        return RATIO_NO_LINK
    if original_debt == 0:
        return RATIO_ZERO
    if original_debt < 0:
        return RATIO_NEGATIVE
    return RATIO_OK


def portfolio_group(label: str | None, pattern: str) -> str:
    """The portfolio label without its trailing sequence token (``Bank A II`` -> ``Bank A``)."""
    if label is None or (isinstance(label, float) and np.isnan(label)) or not str(label).strip():
        return NO_PORTFOLIO
    stripped = re.sub(pattern, "", str(label).strip())
    return stripped or str(label).strip()


def enrich(dataset: Dataset) -> pd.DataFrame:
    """The base table plus follow-up, repeat-break and ratio columns.

    Debtors are linked by ``(product_line, debtor_id)`` and grouped into people by ``person_key``,
    so one person's history is gathered across portfolios and product lines without ever mixing
    two unrelated debtors that happen to share a numeric id.
    """
    df = dataset.base_frame()
    person_debtors: dict[str, set[tuple[Any, Any]]] = defaultdict(set)
    for r in dataset.debtors:
        if r["person_key"]:
            person_debtors[r["person_key"]].add((r["product_line"], r["debtor_id"]))
    events: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for c in dataset.contacts:
        events[(c["product_line"], c["debtor_id"])].append(c)
    breaks: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for b in dataset.breaks:
        breaks[(b["product_line"], b["debtor_id"])].append(b)

    manual = dataset.meta.manual_break_label
    out: dict[str, list[Any]] = {
        name: []
        for name in (
            "contacts_total",
            "contacts_after",
            "answered_after",
            "cpc_after",
            "known_person",
            "prior_breaks",
            "prior_manual_breaks",
        )
    }
    for row in df.itertuples(index=False):
        owners = {(row.product_line, row.debtor_id)}
        key = row.person_key if isinstance(row.person_key, str) else None
        if key:
            owners |= person_debtors.get(key, set())
        start, end = row.agreement_date, row.due_date
        inside = [
            e
            for o in owners
            for e in events.get(o, ())
            if isinstance(start, str) and start <= e["contact_date"] <= end
        ]
        after = [e for e in inside if e["contact_date"] > start]
        earlier = [
            b
            for o in owners
            for b in breaks.get(o, ())
            if isinstance(start, str)
            and b["agreement_date"] < start
            and not (
                b["product_line"] == row.product_line
                and b["agreement_id"] == row.agreement_id
                and b["portfolio_id"] == row.portfolio_id
            )
        ]
        out["contacts_total"].append(len(inside))
        out["contacts_after"].append(len(after))
        out["answered_after"].append(sum(1 for e in after if e["answered"]))
        out["cpc_after"].append(sum(1 for e in after if e["answered"] and e["right_party"]))
        out["known_person"].append(bool(key) and key in person_debtors)
        out["prior_breaks"].append(len(earlier))
        out["prior_manual_breaks"].append(sum(1 for b in earlier if b["status_label"] == manual))
    for name, values in out.items():
        df[name] = values
    df["followup_level"] = [
        followup_level(a, c, n)
        for a, c, n in zip(df["contacts_after"], df["cpc_after"], df["answered_after"], strict=True)
    ]
    df["ratio_state"] = [ratio_state(v if pd.notna(v) else None) for v in df["original_debt"]]
    usable = df["ratio_state"] == RATIO_OK
    safe_debt = df["original_debt"].where(usable)
    df["first_installment_ratio"] = df["first_installment_value"] / safe_debt
    return df


def with_bands(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Add ordered band columns: term, first-installment value, installment count, ratio."""
    out = df.copy()
    for column, source, bands in (
        ("term_band", "days_to_due", settings.term_bands()),
        ("value_band", "first_installment_value", settings.value_bands()),
        ("installment_band", "total_installments", settings.installment_bands()),
    ):
        out[column] = _banded(out[source], bands.labels, bands.assign(out[source].tolist()))
    ratio_bands = settings.ratio_bands()
    ratio_labels = ratio_bands.assign(out["first_installment_ratio"].tolist())
    out["ratio_band"] = _banded(out["first_installment_ratio"], ratio_bands.labels, ratio_labels)
    out.loc[out["ratio_state"] != RATIO_OK, "ratio_band"] = pd.NA
    return out


def _banded(source: pd.Series, labels: list[str], assigned: list[str | None]) -> pd.Series:
    return pd.Series(pd.Categorical(assigned, categories=labels, ordered=True), index=source.index)


def is_unpaid(df: pd.DataFrame) -> pd.Series:
    """Rows in the overdue-unpaid bucket."""
    return df["bucket"] == OVERDUE_UNPAID


def mature_mask(df: pd.DataFrame, grace_days: int) -> pd.Series:
    """Rows whose outcome is final. Same rule as ``derive.is_mature``, evaluated on the column.

    ``days_overdue`` was computed against the dataset's reference date, so the mask depends on the
    due date alone, never on the outcome.
    """
    return df["days_overdue"] > grace_days


# Measured after the agreement, so they must never feed a model that predicts the outcome. They
# are kept in the cleaned table for audit (``paid_amount`` is what proves the outcome).
LEAKAGE_COLUMNS = ("paid_amount", "paid_date", "days_overdue", "status_id", "status_label")
# Describe what happened between agreement and due date. Useful for explanation, but outreach
# targets people who are already slipping, so using them to predict non-payment confuses cause
# and effect.
EXPLANATORY_ONLY_COLUMNS = (
    "contacts_total",
    "contacts_after",
    "answered_after",
    "cpc_after",
    "followup_level",
)

REPEAT_NONE = "First time - no earlier break"


def repeat_level(prior_breaks: int, chronic_threshold: int) -> str:
    """Repeat-break group of one agreement: none, a few, or chronic."""
    if prior_breaks <= 0:
        return REPEAT_NONE
    if prior_breaks >= chronic_threshold:
        return f"Chronic - {chronic_threshold}+ earlier breaks"
    return f"Repeat - {prior_breaks} earlier break{'' if prior_breaks == 1 else 's'}"


RATIO_STATE_LABELS = {
    RATIO_ZERO: "(zero balance - ratio undefined)",
    RATIO_NO_LINK: "(no debt link - ratio unknown)",
    RATIO_NEGATIVE: "(negative balance)",
}


def ratio_group(df: pd.DataFrame) -> pd.Series:
    """Ratio band where the ratio is defined, otherwise the reason it is not."""
    labels = df["ratio_band"].astype(object)
    reason = df["ratio_state"].map(RATIO_STATE_LABELS)
    return labels.where(df["ratio_state"] == RATIO_OK, reason)


def analysis_frame(dataset: Dataset, settings: Settings) -> pd.DataFrame:
    """The base table plus everything the offline stages share.

    Adds follow-up and repeat-break columns, bands, the portfolio group and the maturity flag. The
    maturity cut uses the grace period the dataset was built with, because that is what produced
    its ``bucket`` column.
    """
    df = with_bands(enrich(dataset), settings)
    df["portfolio_group"] = [
        portfolio_group(v if isinstance(v, str) else None, settings.portfolio_group_pattern)
        for v in df["portfolio_label"]
    ]
    df["is_mature"] = mature_mask(df, dataset.meta.grace_days)
    return df
