"""Outcome classification for one first-installment payment slip.

The database supplies raw facts (due date, payment date, amount paid). Everything derived from
them is computed here, against an explicit ``as_of`` date, so that:

* a dataset can be re-derived at any earlier date (stage 04 replays the past this way),
* the maturity rule lives in one place and is unit-tested without a database.

A slip is *paid on time* when money was received on or before ``due + grace_days``. It is
*overdue unpaid* when it fell due before ``as_of`` and was not paid on time. It is *to mature*
when it has not fallen due yet. A slip is *mature* once ``as_of`` is past ``due + grace_days``:
from then on its outcome can no longer change.
"""

from __future__ import annotations

from datetime import date, timedelta

PAID_ON_TIME = "paid_on_time"
OVERDUE_UNPAID = "overdue_unpaid"
TO_MATURE = "to_mature"
BUCKETS = (PAID_ON_TIME, OVERDUE_UNPAID, TO_MATURE)


def days_to_due(agreement_date: date, due_date: date) -> int:
    """The term the agent granted: days from the agreement to the first due date."""
    return (due_date - agreement_date).days


def days_overdue(due_date: date, as_of: date) -> int:
    """Days past the due date at ``as_of``; zero or negative while the slip has not fallen due."""
    return (as_of - due_date).days


def is_paid_on_time(
    due_date: date,
    paid_date: date | None,
    paid_amount: float | None,
    *,
    as_of: date,
    grace_days: int,
) -> bool:
    """Money received by ``as_of`` and no later than ``due + grace_days``.

    A payment dated after ``as_of`` did not exist yet at ``as_of`` and is ignored.
    """
    if paid_date is None or paid_amount is None or paid_amount <= 0:
        return False
    if paid_date > as_of:
        return False
    return paid_date <= due_date + timedelta(days=grace_days)


def classify(
    due_date: date,
    paid_date: date | None,
    paid_amount: float | None,
    *,
    as_of: date,
    grace_days: int,
) -> str:
    """One of ``BUCKETS`` for the slip as it stood at ``as_of``."""
    if is_paid_on_time(due_date, paid_date, paid_amount, as_of=as_of, grace_days=grace_days):
        return PAID_ON_TIME
    if due_date < as_of:
        return OVERDUE_UNPAID
    return TO_MATURE


def is_mature(due_date: date, *, as_of: date, grace_days: int) -> bool:
    """True once the grace window has fully elapsed, so the outcome is final.

    Maturity depends on the due date only, never on the outcome. Selecting mature slips by their
    outcome (for example "paid, or unpaid and old") would bias any rate computed on them.
    """
    return days_overdue(due_date, as_of) > grace_days
