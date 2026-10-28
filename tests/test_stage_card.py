"""Stage 09: the summary card with and without a maturity cut."""

from __future__ import annotations

import math
from datetime import date

import pytest

from delinquency_eda.stages.s00_period import Window
from delinquency_eda.stages.s09_card import card, card_row, default_periods, render

from clirun import run_cli, sqlite_args
from frames import dataset, paid_on_time, raw

MARCH = Window("march", date(2026, 3, 1), date(2026, 3, 31))
FEBRUARY = Window("february", date(2026, 2, 1), date(2026, 2, 28))

# Reference date 2026-03-31, grace 3 days.
ROWS = [
    paid_on_time(1, total_value=100.0),  # due 03-10, paid on time
    raw(2, due_date="2026-03-27", total_value=300.0),  # 4 days overdue: lost
    raw(3, due_date="2026-03-28", total_value=200.0),  # 3 days overdue: still in grace
    raw(4, due_date="2026-03-30", total_value=100.0),  # still in grace
    raw(5, due_date="2026-04-05", total_value=100.0),  # not due yet (outside both windows)
    raw(6, due_date="2026-02-10", agreement_date="2026-02-01", total_value=900.0),  # lost
]


@pytest.fixture
def slips():
    return dataset(ROWS).base


def test_open_month(slips):
    r = card_row(slips, MARCH, 3)
    assert (r.generated, r.paid_on_time, r.to_mature) == (4, 1, 0)
    assert (r.as_counted, r.in_grace, r.lost) == (3, 2, 1)
    assert r.as_counted == r.in_grace + r.lost


def test_value_measures(slips):
    r = card_row(slips, MARCH, 3)
    assert r.value_in_grace == pytest.approx(200.0)
    assert r.value_lost == pytest.approx(100.0)
    assert r.value_as_counted_whole == pytest.approx(600.0)  # 300 + 200 + 100
    assert r.value_lost_whole == pytest.approx(300.0)


def test_inflation_and_rates(slips):
    r = card_row(slips, MARCH, 3)
    assert r.inflation == pytest.approx(3.0)
    assert r.rate_as_card == pytest.approx(1 / 4)  # over every slip, matured or not
    assert (r.mature, r.mature_paid) == (2, 1)  # slips 1 and 2 are past their grace window
    assert r.rate_mature == pytest.approx(1 / 2)


def test_closed_period(slips):
    r = card_row(slips, FEBRUARY, 3)
    assert (r.as_counted, r.lost, r.in_grace) == (1, 1, 0)
    assert r.inflation == pytest.approx(1.0)


def test_empty_periods(slips):
    r = card_row(slips, Window("none", date(2020, 1, 1), date(2020, 1, 31)), 3)
    assert r.generated == 0
    assert math.isnan(r.inflation) and math.isnan(r.rate_as_card) and math.isnan(r.rate_mature)


def test_boundary_day(slips):
    assert card_row(slips, MARCH, 4).lost == 0  # with 4 days of grace row 2 is still open
    assert card_row(slips, MARCH, 2).lost == 2  # with 2 days, row 3 is lost too


def test_default_periods():
    periods = default_periods(date(2026, 3, 15))
    assert [p.name for p in periods] == ["month to date", "previous month", "two months back"]
    assert periods[0].start == date(2026, 3, 1) and periods[0].end == date(2026, 3, 15)
    assert (periods[1].start, periods[1].end) == (date(2026, 2, 1), date(2026, 2, 28))
    assert (periods[2].start, periods[2].end) == (date(2026, 1, 1), date(2026, 1, 31))


def test_report_columns(slips):
    rows = card(slips, [MARCH, FEBRUARY], 3)
    text = render(rows, date(2026, 3, 31), 3).render()
    header = next(line for line in text.splitlines() if line.startswith("| indicator"))
    assert header == "| indicator | march | february |"
    assert "inflation of the card count" in text


def test_cli_from_dataset(tmp_path):
    path = dataset(ROWS).save(tmp_path / "ds.json")
    base = ["card", "--dataset", str(path), "--out-dir", str(tmp_path / "r"), "--from-dataset"]
    code, out = run_cli(*base, "--period", "march=2026-03-01:2026-03-31")
    assert code == 0 and "march" in out
    code, _ = run_cli(*base, "--as-of", "2026-03-31")
    assert code == 2


def test_routes_agree(world, tmp_path):
    dataset_path = tmp_path / "ds.json"
    out_dir = str(tmp_path / "r")
    sources = sqlite_args(("auto", world))
    period = ["--period", "march=2026-03-01:2026-03-31"]
    code, _ = run_cli(
        *sources, "--dataset", str(dataset_path), "--out-dir", out_dir,
        "base", "--from", "2026-03-01", "--to", "2026-03-31", "--as-of", "2026-03-31",
    )  # fmt: skip
    assert code == 0
    code_db, via_db = run_cli(
        *sources, "--out-dir", out_dir, "card", "--as-of", "2026-03-31", *period
    )
    code_ds, via_ds = run_cli(
        "--dataset", str(dataset_path), "--out-dir", out_dir, "card", "--from-dataset", *period
    )
    assert code_db == code_ds == 0
    line = lambda text: next(x for x in text.splitlines() if x.startswith("march"))  # noqa: E731
    assert line(via_db) == line(via_ds)
