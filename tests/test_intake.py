"""Intake: the row-level entry rules, judged on values as the driver returns them."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import pytest

from delinquency_eda.db import SQLiteDatabase
from delinquency_eda.intake import (
    PAYMENT,
    READABLE,
    REQUIRED,
    Intake,
    fingerprint,
    is_missing,
    reads_as_count,
    reads_as_date,
    reads_as_flag,
    reads_as_number,
)
from delinquency_eda.queries import Query

from frames import raw

# The columns of a slip as the extraction query returns them, with invented values.
SLIP = raw(1) | {"plan_installments": 1}


def slip(**overrides):
    return {**SLIP, **overrides}


def test_clean_slip():
    intake = Intake()
    assert intake.accept("slips", [slip()]) == [slip()]
    for rule in (REQUIRED, READABLE, PAYMENT):
        assert (intake.tallies[rule].checked, intake.tallies[rule].violations) == (1, 0)


def test_paid_slip():
    row = slip(paid_amount=100.0, paid_date="2026-03-11")
    assert Intake().accept("slips", [row]) == [row]


@pytest.mark.parametrize("column", ["agent_id", "due_date", "agreement_date", "debtor_id"])
@pytest.mark.parametrize("empty", [None, "", "   ", float("nan")])
def test_missing_required(column, empty):
    intake = Intake()
    assert intake.accept("slips", [slip(**{column: empty}), slip()]) == [slip()]
    tally = intake.tallies[REQUIRED]
    assert (tally.checked, tally.violations) == (2, 1)
    assert tally.detail == {f"slips.{column}": 1}


def test_two_flaws_one_row():
    intake = Intake()
    intake.accept("slips", [slip(agent_id=None, due_date=None)])
    tally = intake.tallies[REQUIRED]
    assert tally.violations == 1
    assert tally.detail == {"slips.agent_id": 1, "slips.due_date": 1}


def test_optional_nulls():
    row = slip(person_key=None, original_debt=None, portfolio_label=None, paid_amount=None)
    assert Intake().accept("slips", [row]) == [row]


@pytest.mark.parametrize(
    ("column", "bad"),
    [
        ("due_date", "05/03/2026"),
        ("due_date", "2026-3-5"),
        ("due_date", 20260305),
        ("agreement_date", "yesterday"),
        ("first_installment_value", "abc"),
        ("first_installment_value", float("inf")),
        ("original_debt", "12,5"),
        ("total_installments", "many"),
        ("total_installments", 2.5),
    ],
)
def test_unreadable_value(column, bad):
    intake = Intake()
    assert intake.accept("slips", [slip(**{column: bad})]) == []
    assert intake.tallies[READABLE].detail == {f"slips.{column}": 1}


@pytest.mark.parametrize(
    "good",
    [
        ("due_date", "2026-03-05"),
        ("due_date", "2026-03-05 10:30:00"),
        ("due_date", date(2026, 3, 5)),
        ("due_date", datetime(2026, 3, 5, 10, 30)),
        ("first_installment_value", Decimal("12.50")),
        ("first_installment_value", "12.5"),
        ("first_installment_value", 12),
        ("total_installments", 3.0),
        ("total_installments", Decimal("3")),
    ],
)
def test_driver_forms(good):
    column, value = good
    assert Intake().accept("slips", [slip(**{column: value})]) == [slip(**{column: value})]


@pytest.mark.parametrize(
    "flaw",
    [
        {"paid_amount": 0.0, "paid_date": "2026-03-11"},
        {"paid_amount": -5.0, "paid_date": "2026-03-11"},
        {"paid_amount": 100.0, "paid_date": None},
        {"paid_amount": None, "paid_date": "2026-03-11"},
        {"paid_amount": 0.0, "paid_date": None},
    ],
)
def test_payment_flaws(flaw):
    intake = Intake()
    assert intake.accept("slips", [slip(**flaw)]) == []
    tally = intake.tallies[PAYMENT]
    assert (tally.checked, tally.violations) == (1, 1)


def test_payment_detail():
    intake = Intake()
    intake.accept("slips", [slip(paid_amount=0.0, paid_date=None)])
    assert intake.tallies[PAYMENT].detail == {
        "slips.paid_amount is not positive": 1,
        "slips.paid_amount without paid_date": 1,
    }


def test_payment_only_slips():
    intake = Intake()
    row = CONTACT | {"answered": 1, "right_party": 0}
    assert intake.accept("contacts", [row]) == [row]
    assert intake.tallies[PAYMENT].checked == 0


CONTACT = {"debtor_id": 1, "contact_date": "2026-03-02", "result_id": 3}


@pytest.mark.parametrize("flag", ["Y", "N", "yes", "0", 2, -1, 0.5])
def test_bad_flag(flag):
    # bool('N') is True, so a text flag has to be refused, not coerced.
    intake = Intake()
    assert intake.accept("contacts", [CONTACT | {"answered": flag, "right_party": 1}]) == []
    assert intake.tallies[READABLE].detail == {"contacts.answered": 1}


@pytest.mark.parametrize("empty", [None, "", "  "])
def test_missing_flag(empty):
    # bool(None) is False, so a NULL flag must not slip through as 'no'.
    intake = Intake()
    assert intake.accept("contacts", [CONTACT | {"answered": 1, "right_party": empty}]) == []
    assert intake.tallies[REQUIRED].detail == {"contacts.right_party": 1}


@pytest.mark.parametrize("flag", [0, 1, True, False, 1.0, Decimal(0)])
def test_flag_values(flag):
    assert reads_as_flag(flag)


def test_breaks_and_debtors():
    intake = Intake()
    broken = {"debtor_id": 1, "agreement_id": 9, "portfolio_id": 1, "status_id": 2}
    assert intake.accept("breaks", [broken | {"agreement_date": "2026-01-01"}])
    assert not intake.accept("breaks", [broken | {"agreement_date": None}])
    assert not intake.accept("debtors", [{"debtor_id": None, "person_key": "k"}])
    assert intake.accept("debtors", [{"debtor_id": 1, "person_key": None}])


def test_helpers():
    assert is_missing(None) and is_missing("") and is_missing(float("nan"))
    assert not is_missing(0) and not is_missing(0.0) and not is_missing("0")
    assert reads_as_date("2026-03-05") and not reads_as_date("2026/03/05")
    assert reads_as_number(Decimal("1.5")) and not reads_as_number(True)
    assert reads_as_count(3) and not reads_as_count(3.5)


# ---- the query log -------------------------------------------------------------------------


def test_fingerprint():
    short = "SELECT 1 WHERE x IN (?, ?) AND y = ?"
    long = "SELECT 1 WHERE x IN (?, ?, ?, ?, ?) AND y = ?"
    assert fingerprint(short) == fingerprint(long)
    assert fingerprint(short) != fingerprint("SELECT 2 WHERE x IN (?, ?) AND y = ?")
    assert len(fingerprint(short)) == 64


def test_query_log(world):
    intake = Intake()
    with SQLiteDatabase.open(world) as db:
        query = Query("SELECT agent_id FROM agents WHERE agent_id IN (?, ?)", ("a1", "a2"))
        assert len(intake.run(db, "auto", "agents", query)) == 2
        longer = Query(
            "SELECT agent_id FROM agents WHERE agent_id IN (?, ?, ?)", ("a1", "a2", "sys")
        )
        intake.run(db, "auto", "agents", longer)
        intake.run(db, "consumer", "agents", query)
    log = intake.queries()
    assert [(q["product_line"], q["name"], q["runs"], q["rows"]) for q in log] == [
        ("auto", "agents", 2, 5),
        ("consumer", "agents", 1, 2),
    ]
    assert log[0]["sha256"] == fingerprint(query.sql)


def test_query_log_order(world):
    first, second = Intake(), Intake()
    with SQLiteDatabase.open(world) as db:
        for intake, order in ((first, ("b", "a")), (second, ("a", "b"))):
            for name in order:
                intake.run(db, "auto", name, Query(f"SELECT '{name}' AS n", ()))
    assert first.queries() == second.queries()
