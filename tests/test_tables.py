"""The shared group tables: concentration over the unpaid bucket, discrimination over the mature."""

from __future__ import annotations

import numpy as np
import pandas as pd

from delinquency_eda.config import Settings
from delinquency_eda.tables import (
    NOT_RELIABLE,
    RELIABLE,
    WEAK,
    concentration,
    discrimination,
    label_series,
    reliability,
    sorted_groups,
    unpaid_rate,
)

SETTINGS = Settings(min_n_reliable=4, min_n_weak=2)


def test_reliability_thresholds_are_inclusive():
    assert reliability(4, SETTINGS) == RELIABLE
    assert reliability(3, SETTINGS) == WEAK
    assert reliability(2, SETTINGS) == WEAK
    assert reliability(1, SETTINGS) == NOT_RELIABLE


def test_unpaid_rate_and_the_empty_frame():
    buckets = ["overdue_unpaid", "paid_on_time", "paid_on_time", "paid_on_time"]
    frame = pd.DataFrame({"bucket": buckets})
    assert unpaid_rate(frame) == 0.25
    assert np.isnan(unpaid_rate(frame.iloc[0:0]))


def test_labels_are_text_and_missing_values_are_named():
    labels = label_series(pd.Series([1, None, "x"], dtype=object))
    assert labels.tolist() == ["1", "(unknown)", "x"]


def test_sorted_groups_prefers_the_natural_order_then_the_largest():
    fallback = pd.Series({"a": 1.0, "b": 5.0, "c": 3.0, "d": 9.0})
    assert sorted_groups(["a", "b", "c", "d"], ["c", "a"], fallback) == ["c", "a", "d", "b"]
    assert sorted_groups(["a", "b", "c", "d"], None, fallback) == ["d", "b", "c", "a"]


def _unpaid_frame():
    return pd.DataFrame(
        {"first_installment_value": [100.0, 300.0, 600.0, np.nan]}, index=[10, 11, 12, 13]
    )


def test_concentration_totals_shares_and_order():
    frame = _unpaid_frame()
    labels = pd.Series(["g1", "g1", "g2", "g1"], index=frame.index)
    table = concentration(frame, labels)
    assert table["group"].tolist() == ["g2", "g1"]  # largest value first
    g1 = table.set_index("group").loc["g1"]
    assert (g1["slips"], g1["value"]) == (3, 400.0)  # a missing value counts as zero
    assert round(g1["share_slips"], 3) == 0.75
    assert round(g1["share_value"], 3) == 0.4
    assert g1["avg_ticket"] == 400.0 / 3
    assert table["share_value"].sum() == 1.0


def test_concentration_follows_a_natural_order_when_given():
    frame = _unpaid_frame()
    labels = pd.Series(["g1", "g1", "g2", "g1"], index=frame.index)
    assert concentration(frame, labels, order=["g1", "g2"])["group"].tolist() == ["g1", "g2"]


def test_concentration_of_nothing_is_an_empty_table():
    frame = _unpaid_frame().iloc[0:0]
    table = concentration(frame, pd.Series([], dtype=object, index=frame.index))
    assert table.empty
    assert next(iter(table.columns)) == "group"


def _mature_frame():
    buckets = ["overdue_unpaid", "paid_on_time", "overdue_unpaid", "overdue_unpaid"]
    return pd.DataFrame({"bucket": buckets}, index=[1, 2, 3, 4])


def test_discrimination_rate_lift_and_sample_size():
    mature = _mature_frame()
    labels = pd.Series(["g1", "g1", "g2", "g2"], index=mature.index)
    baseline = unpaid_rate(mature)
    table = discrimination(mature, labels, baseline, SETTINGS).set_index("group")
    assert baseline == 0.75
    assert (table.loc["g2", "mature"], table.loc["g2", "unpaid"]) == (2, 2)
    assert table.loc["g2", "rate"] == 1.0
    assert round(table.loc["g2", "lift"], 4) == round(1.0 / 0.75, 4)
    assert table.loc["g1", "lift"] == 0.5 / 0.75
    assert table.loc["g1", "reliability"] == WEAK
    ordered = discrimination(mature, labels, baseline, SETTINGS)
    assert ordered["group"].tolist() == ["g2", "g1"]


def test_lift_is_undefined_when_the_baseline_is_zero_or_missing():
    mature = _mature_frame()
    labels = pd.Series(["g1", "g1", "g2", "g2"], index=mature.index)
    for baseline in (0.0, float("nan")):
        assert discrimination(mature, labels, baseline, SETTINGS)["lift"].isna().all()
