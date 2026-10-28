"""Stage 05: how much of the overdue-unpaid bucket is still open."""

from __future__ import annotations

import math

import pytest

from delinquency_eda.stages.s05_censoring import FINAL, OPEN, measure, render

from clirun import run_cli
from frames import dataset, paid_on_time, raw

# Reference date 2026-03-31, grace 3: a slip due 03-27 is 4 days overdue (definitive), one due
# 03-28 is exactly 3 days overdue (still inside the grace window), 03-31 is not overdue yet.
ROWS = [
    paid_on_time(1),  # mature, paid
    raw(2, due_date="2026-03-27"),  # unpaid, definitive
    raw(3, due_date="2026-03-28"),  # unpaid, still open (boundary)
    raw(4, due_date="2026-03-30"),  # unpaid, still open
    raw(5, due_date="2026-03-31"),  # due today: not overdue yet
    raw(6, due_date="2026-03-27", first_installment_value=50.0, product_line="consumer"),
]


@pytest.fixture
def result():
    ds = dataset(ROWS, lines=("auto", "consumer"))
    return measure(ds.base_frame(), 3)


def test_bucket_split(result):
    assert (result.unpaid, result.definitive, result.still_open) == (4, 2, 2)


def test_boundary_day(result):
    states = dict(zip(result.by_due_date["due_date"], result.by_due_date["state"], strict=True))
    assert states["2026-03-27"] == FINAL
    assert states["2026-03-28"] == OPEN


def test_inflation(result):
    assert result.inflation == pytest.approx(4 / 2)


def test_value_split(result):
    assert result.value_definitive == pytest.approx(150.0)  # 100 + 50
    assert result.value_open == pytest.approx(200.0)


def test_mature_rate(result):
    # all six slips: row 1 paid on time. mature = rows 1, 2, 6 (rows 3-5 are not past grace).
    assert result.naive_on_time_rate == pytest.approx(1 / 6)
    assert result.mature == 3
    assert result.mature_on_time_rate == pytest.approx(1 / 3)


def test_undefined_inflation():
    ds = dataset([raw(1, due_date="2026-03-30")])
    assert math.isnan(measure(ds.base_frame(), 3).inflation)


def test_by_line_table(result):
    by_line = result.by_line.set_index("product_line")
    assert (by_line.loc["auto", "unpaid"], by_line.loc["auto", "still_open"]) == (3, 2)
    assert by_line.loc["consumer", "definitive"] == 1


def test_old_due_dates():
    rows = [raw(i, due_date=f"2026-02-{i:02d}") for i in range(1, 11)] + [
        raw(20, due_date="2026-03-29")
    ]
    ds = dataset(rows)
    text = render(measure(ds.base_frame(), 3), len(rows)).render()
    assert "older due date(s) are all definitive" in text
    assert "2026-02-01" not in text  # summarized, not listed
    assert "2026-03-29" in text


def test_cli_writes_the_report(tmp_path):
    path = dataset(ROWS, lines=("auto", "consumer")).save(tmp_path / "ds.json")
    code, out = run_cli("censoring", "--dataset", str(path), "--out-dir", str(tmp_path / "r"))
    assert code == 0
    assert "inflation=2.00x" in out
    assert (tmp_path / "r" / "censoring.md").exists()
