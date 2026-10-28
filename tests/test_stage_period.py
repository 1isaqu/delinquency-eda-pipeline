"""Stage 00: which window reproduces a funnel."""

from __future__ import annotations

from datetime import date

import pytest

from delinquency_eda import derive
from delinquency_eda.errors import ConfigError
from delinquency_eda.stages.s00_period import (
    FUNNEL_KEYS,
    Window,
    default_windows,
    distance,
    funnel,
    parse_reference,
    parse_window,
    rank,
)

from clirun import run_cli, sqlite_args


def slip(due: str, bucket: str, value: float = 100.0, paid: float | None = None) -> dict:
    return {
        "due_date": due,
        "bucket": bucket,
        "first_installment_value": value,
        "paid_amount": paid,
        "paid_date": "2026-03-01" if paid else None,
    }


SLIPS = [
    slip("2026-03-01", derive.PAID_ON_TIME, paid=100.0),
    slip("2026-03-05", derive.OVERDUE_UNPAID, value=250.0),
    slip("2026-03-10", derive.OVERDUE_UNPAID, value=50.0),
    slip("2026-03-31", derive.TO_MATURE),
    slip("2026-02-28", derive.PAID_ON_TIME, paid=100.0),
]


def test_parse_window():
    w = parse_window("march=2026-03-01:2026-03-31")
    assert (w.name, w.start, w.end) == ("march", date(2026, 3, 1), date(2026, 3, 31))


@pytest.mark.parametrize(
    "bad", ["", "march", "march=2026-03-01", "=2026-03-01:2026-03-02", "m=x:y"]
)
def test_malformed_window(bad):
    with pytest.raises(ConfigError):
        parse_window(bad)


def test_reversed_window():
    with pytest.raises(ConfigError, match="ends before"):
        Window("w", date(2026, 3, 2), date(2026, 3, 1))


def test_parse_reference():
    assert parse_reference(["generated=12", "paid_on_time=3"]) == {
        "generated": 12,
        "paid_on_time": 3,
    }
    for bad in (["generated"], ["nonsense=1"], ["generated=many"]):
        with pytest.raises(ConfigError):
            parse_reference(bad)


def test_default_windows():
    names = {w.name: w for w in default_windows(date(2026, 3, 15))}
    assert names["month to date"].start == date(2026, 3, 1)
    assert names["month to yesterday"].end == date(2026, 3, 14)
    assert (names["previous month"].start, names["previous month"].end) == (
        date(2026, 2, 1),
        date(2026, 2, 28),
    )


def test_first_of_month():
    windows = default_windows(date(2026, 3, 1))
    spans = [(w.start, w.end) for w in windows]
    assert len(spans) == len(set(spans))
    assert all(w.end >= w.start for w in windows)


def test_funnel_partition():
    march = Window("march", date(2026, 3, 1), date(2026, 3, 31))
    f = funnel(SLIPS, march)
    assert f.generated == 4  # the 02-28 slip is outside
    assert f.generated == f.paid_on_time + f.overdue_unpaid + f.to_mature
    assert (f.paid_on_time, f.overdue_unpaid, f.to_mature) == (1, 2, 1)
    assert f.overdue_unpaid_value == 300.0
    assert f.amount_due == 100 + 250 + 50 + 100
    assert f.amount_received == 100.0


def test_inclusive_window():
    one_day = Window("d", date(2026, 3, 5), date(2026, 3, 5))
    assert funnel(SLIPS, one_day).generated == 1


def test_distance_ranking():
    march = funnel(SLIPS, Window("march", date(2026, 3, 1), date(2026, 3, 31)))
    day = funnel(SLIPS, Window("day", date(2026, 3, 5), date(2026, 3, 5)))
    assert distance(march, {"generated": 4}) == 0
    assert distance(day, {"generated": 4, "overdue_unpaid": 1}) == 3
    ranked = rank({"march": march, "day": day}, {"generated": 4})
    assert ranked == [("march", 0), ("day", 3)]


def test_funnel_keys():
    assert FUNNEL_KEYS == ("generated", "to_mature", "overdue_unpaid", "paid_on_time")


def test_cli_windows(world, tmp_path):
    out_dir = tmp_path / "reports"
    code, out = run_cli(
        *sqlite_args(("auto", world)),
        "--out-dir", str(out_dir),
        "period",
        "--as-of", "2026-03-31",
        "--window", "march=2026-03-01:2026-03-31",
        "--window", "first week=2026-03-01:2026-03-07",
        "--reference", "generated=5",
    )  # fmt: skip
    assert code == 0
    assert "march" in out and "closest window: march (distance 0)" in out
    text = (out_dir / "period.md").read_text(encoding="utf-8")
    assert "Closest window: **march**" in text


def test_cli_without_sources(tmp_path):
    code, _ = run_cli("period", "--out-dir", str(tmp_path), "--backend", "sqlite")
    assert code == 2


def test_window_names_must_be_unique(world, tmp_path):
    code, _ = run_cli(
        *sqlite_args(("auto", world)),
        "--out-dir", str(tmp_path / "r"),
        "period",
        "--window", "w=2026-03-01:2026-03-05",
        "--window", "w=2026-03-06:2026-03-10",
    )  # fmt: skip
    assert code == 2
