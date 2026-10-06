"""Group tables shared by the analysis, EDA and diagnostics stages.

Two questions are asked of every factor, and they need different populations:

* *concentration*: among the slips that are overdue and unpaid, who holds the value?
* *discrimination*: among the slips whose outcome is final (mature), does the group fail more or
  less often than the baseline?

Concentration describes where the money is. Only discrimination says whether a factor separates
payers from non-payers, so it uses the mature cohort alone: counting slips that can still be
paid on time as failures would mix loss with censoring.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from delinquency_eda.config import Settings
from delinquency_eda.derive import OVERDUE_UNPAID
from delinquency_eda.features import UNKNOWN

RELIABLE = "reliable"
WEAK = "weak"
NOT_RELIABLE = "not reliable"

CONCENTRATION_COLUMNS = ["group", "slips", "share_slips", "value", "share_value", "avg_ticket"]
DISCRIMINATION_COLUMNS = ["group", "mature", "unpaid", "rate", "lift", "reliability"]


def reliability(n: int, settings: Settings) -> str:
    """How far a group's size supports a reading: a rate on a handful of slips is noise."""
    if n >= settings.min_n_reliable:
        return RELIABLE
    return WEAK if n >= settings.min_n_weak else NOT_RELIABLE


def unpaid_rate(frame: pd.DataFrame) -> float:
    """Share of rows in the overdue-unpaid bucket; ``nan`` for an empty frame."""
    if frame.empty:
        return float("nan")
    return float((frame["bucket"] == OVERDUE_UNPAID).mean())


def label_series(labels: pd.Series) -> pd.Series:
    """Plain-object labels with missing values named, so a group-by never drops them."""
    out = labels.astype(object)
    return out.where(out.notna(), UNKNOWN).astype(str)


def sorted_groups(
    groups: Iterable[str], order: Sequence[str] | None, fallback: pd.Series
) -> list[str]:
    """Natural order when one is known, else largest ``fallback`` first."""
    present = list(groups)
    if order:
        known = [g for g in order if g in present]
        rest = [g for g in present if g not in known]
        return known + sorted(rest, key=lambda g: -float(fallback.get(g, 0.0)))
    return sorted(present, key=lambda g: -float(fallback.get(g, 0.0)))


def concentration(
    frame: pd.DataFrame,
    labels: pd.Series,
    *,
    value_col: str = "first_installment_value",
    order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Slips and value per group, with each group's share of the whole."""
    keyed = pd.DataFrame({"group": label_series(labels), "value": frame[value_col].fillna(0.0)})
    agg = keyed.groupby("group", sort=False).agg(slips=("value", "size"), value=("value", "sum"))
    total_slips, total_value = int(agg["slips"].sum()), float(agg["value"].sum())
    agg["share_slips"] = agg["slips"] / total_slips if total_slips else 0.0
    agg["share_value"] = agg["value"] / total_value if total_value else 0.0
    agg["avg_ticket"] = agg["value"] / agg["slips"]
    groups = sorted_groups(agg.index, order, agg["value"])
    return agg.loc[groups].reset_index()[CONCENTRATION_COLUMNS]


def discrimination(
    mature: pd.DataFrame,
    labels: pd.Series,
    baseline: float,
    settings: Settings,
    *,
    order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Unpaid rate per group over the mature cohort, with lift against ``baseline``."""
    keyed = pd.DataFrame(
        {
            "group": label_series(labels),
            "unpaid": (mature["bucket"] == OVERDUE_UNPAID).astype(int),
        }
    )
    agg = keyed.groupby("group", sort=False).agg(
        mature=("unpaid", "size"), unpaid=("unpaid", "sum")
    )
    agg["rate"] = agg["unpaid"] / agg["mature"]
    usable = bool(baseline) and not np.isnan(baseline) and baseline > 0
    agg["lift"] = agg["rate"] / baseline if usable else np.nan
    agg["reliability"] = [reliability(int(n), settings) for n in agg["mature"]]
    groups = sorted_groups(agg.index, order, agg["rate"])
    return agg.loc[groups].reset_index()[DISCRIMINATION_COLUMNS]
