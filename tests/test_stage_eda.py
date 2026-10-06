"""Stage 06: the EDA report generator. Computation first, then the assembled report."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from delinquency_eda import features as F
from delinquency_eda.config import Settings
from delinquency_eda.stages import s06_eda as E

from clirun import run_cli
from frames import dataset, paid_on_time, raw

SETTINGS = Settings(min_n_reliable=2, min_n_weak=1, rare_category_min_n=3)


def frame(rows, **kwargs):
    return F.analysis_frame(dataset(rows, **kwargs), SETTINGS)


def checks_by_name(df):
    return {c.name: c for c in E.quality_checks(df)}


def test_quality_checks_count_each_condition_once_on_rows_built_to_trigger_it():
    rows = [
        raw(1, original_debt=0.0),
        raw(2, original_debt=None, portfolio_label=None, person_key=None),
        raw(3, original_debt=-5.0),
        raw(4, first_installment_value=0.0),
        raw(5, total_value=10.0),  # below its own first installment (100)
        raw(6, agreement_date="2026-03-20", due_date="2026-03-10"),  # due before the agreement
        raw(7, plan_installments=3, total_installments=1),
        raw(8, first_installment_value=900.0, original_debt=500.0, total_value=900.0),
        raw(9, paid_amount=100.0, paid_date="2026-03-25"),  # paid after the grace period
        raw(9, record_id=90),  # same agreement again: grain violated
    ]
    c = checks_by_name(frame(rows))
    assert c["original_debt equals zero"].rows == 1
    assert c["original_debt missing"].rows == 1
    assert c["original_debt negative"].rows == 1
    assert c["portfolio_label missing"].rows == 1
    assert c["person_key missing"].rows == 1
    assert c["first_installment_value <= 0"].rows == 1
    assert c["total_value below first_installment_value"].rows == 1
    assert c["days_to_due negative"].rows == 1
    assert c["plan_installments differs from generated installments"].rows == 1
    assert c["first installment above original debt"].rows == 1
    assert c["money received on an unpaid slip"].rows == 1
    assert c["duplicate (product_line, agreement_id, portfolio_id)"].rows == 1
    assert c["duplicate (product_line, record_id)"].rows == 0


def test_a_clean_frame_triggers_no_high_severity_check():
    rows = [raw(1), paid_on_time(2), raw(3, due_date="2026-03-20")]
    flagged = [c for c in E.quality_checks(frame(rows)) if c.severity == E.HIGH and c.rows]
    assert flagged == []


def test_record_ids_repeating_across_product_lines_are_informational():
    rows = [raw(1), raw(1, product_line="consumer")]
    c = checks_by_name(frame(rows, lines=("auto", "consumer")))
    shared = c["record_id shared by two product lines"]
    assert (shared.rows, shared.severity) == (2, E.INFO)
    assert c["duplicate (product_line, record_id)"].rows == 0


def test_profile_has_no_example_values():
    prof = E.profile(frame([raw(1, person_key="never_print_me")]).iloc[:, :24])
    assert set(prof.columns) == {"column", "dtype", "nulls", "% null", "distinct"}
    assert "never_print_me" not in prof.to_string()


def test_nullness_classifies_the_outcome_when_money_only_exists_if_paid():
    mature = frame([raw(1), raw(2), paid_on_time(3), paid_on_time(4)])
    assert E.nullness_accuracy(mature) == 1.0
    assert np.isnan(E.nullness_accuracy(mature.iloc[0:0]))


def test_orphans_and_rare_categories():
    left = pd.DataFrame({"k": ["a", "b", "z"]})
    right = pd.DataFrame({"k": ["a", "b", "c"]})
    assert E.orphans(left, right, ["k"]) == (2, 1)
    assert E.orphans(left.iloc[0:0], right, ["k"]) == (0, 0)
    total, small, share = E.rare_categories(pd.Series(["a"] * 5 + ["b", "c"]), 3)
    assert (total, small) == (3, 2) and share == pytest.approx(2 / 7)


def test_monthly_table_counts_mature_and_unpaid_per_due_month():
    feb = {"due_date": "2026-02-10", "agreement_date": "2026-02-01"}
    jan = {"due_date": "2026-01-10", "agreement_date": "2026-01-01"}
    paid = {"paid_amount": 100.0, "paid_date": "2026-01-11"}
    rows = [raw(i, **feb) for i in range(1, 5)] + [raw(i, **jan, **paid) for i in range(5, 9)]
    months = E.monthly(frame(rows)).set_index("month")
    assert months.index.tolist() == ["2026-01", "2026-02"]
    assert months.loc["2026-02", "unpaid"] == 4 and months.loc["2026-02", "rate"] == 1.0
    assert months.loc["2026-01", "unpaid"] == 0 and months.loc["2026-01", "rate"] == 0.0


def test_stability_range_ignores_months_without_enough_mature_slips():
    two = pd.DataFrame({"mature": [10, 10, 1], "rate": [0.2, 0.5, 0.9]})
    assert E.stability_range(two, 5) == pytest.approx(30.0)  # the 1-slip month is ignored
    assert E.stability_range(two.iloc[:1], 5) is None


def test_correlation_marks_leaking_columns_and_finds_the_term():
    rows = []
    for i in range(1, 41):
        term = i % 10
        due = f"2026-03-{10 + term:02d}"
        base = {"agreement_date": "2026-03-01", "due_date": due}
        rows.append(raw(i, **base) if term >= 5 else paid_on_time(i, **base))
    table = E.correlations(frame(rows)).set_index("column")
    assert table.loc["days_to_due", "spearman"] > 0.5
    assert bool(table.loc["paid_amount", "leaks"]) and bool(table.loc["days_overdue", "leaks"])
    assert not bool(table.loc["days_to_due", "leaks"])


def test_collinear_pairs_exclude_leakage_columns():
    rows = [
        raw(i, first_installment_value=float(i), total_value=float(2 * i)) for i in range(1, 12)
    ]
    pairs = E.collinear_pairs(frame(rows))
    assert ("first_installment_value", "total_value") in set(
        zip(pairs["column a"], pairs["column b"], strict=True)
    )
    assert "paid_amount" not in set(pairs["column a"]) | set(pairs["column b"])


def test_the_assembled_report_has_ordered_sections_and_no_person_key():
    rows = [raw(i, person_key=f"pk_secret_{i}") for i in range(1, 6)] + [paid_on_time(6)]
    ds = dataset(rows)
    df = F.analysis_frame(ds, SETTINGS)
    text = E.render(ds, df, SETTINGS, file_size=1234).render()
    numbers = [int(m) for m in re.findall(r"^## (\d+)\. ", text, flags=re.MULTILINE)]
    assert numbers == list(range(1, 15))
    assert "pk_secret_" not in text
    assert "1,234 bytes" in text
    assert "6.2 Target leakage" in text


def test_a_recommendation_only_appears_when_its_trigger_fired():
    no_zero = frame([raw(1), paid_on_time(2)])
    ds = dataset([raw(1), paid_on_time(2)])
    text = E.render(ds, no_zero, SETTINGS).render()
    assert "zero original debt" not in text.split("## 13. Recommendations")[1]
    with_zero = dataset([raw(1, original_debt=0.0), paid_on_time(2)])
    text = E.render(with_zero, F.analysis_frame(with_zero, SETTINGS), SETTINGS).render()
    assert "zero original debt" in text.split("## 13. Recommendations")[1]


def test_cli_writes_the_report(tmp_path):
    path = dataset([raw(1), paid_on_time(2), raw(3, due_date="2026-03-20")]).save(
        tmp_path / "ds.json"
    )
    code, out = run_cli("eda", "--dataset", str(path), "--out-dir", str(tmp_path / "r"))
    assert code == 0 and "slips=3" in out
    assert (tmp_path / "r" / "eda.md").read_text(encoding="utf-8").startswith("# Exploratory")
