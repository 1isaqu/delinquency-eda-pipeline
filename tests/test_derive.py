"""Outcome classification. The boundaries are the whole point, so each one has a test."""

from __future__ import annotations

from datetime import date

import pytest

from delinquency_eda import derive

DUE = date(2026, 3, 10)
GRACE = 3


def classify(paid_date, paid_amount, as_of):
    return derive.classify(DUE, paid_date, paid_amount, as_of=as_of, grace_days=GRACE)


def test_paid_on_the_last_day_of_grace_counts_as_on_time():
    assert classify(date(2026, 3, 13), 10.0, date(2026, 3, 31)) == derive.PAID_ON_TIME


def test_paid_one_day_after_grace_is_unpaid():
    assert classify(date(2026, 3, 14), 10.0, date(2026, 3, 31)) == derive.OVERDUE_UNPAID


def test_paid_early_counts_as_on_time():
    assert classify(date(2026, 3, 1), 10.0, date(2026, 3, 31)) == derive.PAID_ON_TIME


@pytest.mark.parametrize("amount", [0.0, None, -5.0])
def test_a_payment_without_money_is_not_a_payment(amount):
    assert classify(date(2026, 3, 11), amount, date(2026, 3, 31)) == derive.OVERDUE_UNPAID


def test_no_payment_date_means_unpaid():
    assert classify(None, 10.0, date(2026, 3, 31)) == derive.OVERDUE_UNPAID


def test_the_due_date_itself_is_not_yet_overdue():
    assert classify(None, None, DUE) == derive.TO_MATURE
    assert classify(None, None, date(2026, 3, 11)) == derive.OVERDUE_UNPAID


def test_a_slip_inside_its_grace_window_is_overdue_but_not_mature():
    as_of = date(2026, 3, 12)
    assert classify(None, None, as_of) == derive.OVERDUE_UNPAID
    assert not derive.is_mature(DUE, as_of=as_of, grace_days=GRACE)


def test_maturity_starts_the_day_after_the_grace_window_ends():
    assert not derive.is_mature(DUE, as_of=date(2026, 3, 13), grace_days=GRACE)
    assert derive.is_mature(DUE, as_of=date(2026, 3, 14), grace_days=GRACE)


def test_a_payment_after_the_reference_date_did_not_exist_yet():
    # Paid on the 12th, looked at on the 11th: still overdue then.
    assert classify(date(2026, 3, 12), 10.0, date(2026, 3, 11)) == derive.OVERDUE_UNPAID
    assert classify(date(2026, 3, 12), 10.0, date(2026, 3, 12)) == derive.PAID_ON_TIME


def test_zero_grace_period():
    assert (
        derive.classify(DUE, date(2026, 3, 11), 1.0, as_of=date(2026, 3, 31), grace_days=0)
        == derive.OVERDUE_UNPAID
    )
    assert (
        derive.classify(DUE, DUE, 1.0, as_of=date(2026, 3, 31), grace_days=0) == derive.PAID_ON_TIME
    )


def test_day_counts():
    assert derive.days_to_due(date(2026, 3, 1), date(2026, 3, 1)) == 0
    assert derive.days_to_due(date(2026, 3, 1), date(2026, 3, 8)) == 7
    assert derive.days_overdue(DUE, date(2026, 3, 8)) == -2
