"""Stage 09 - the corrected summary card: the same indicator with a maturity cut.

A card that shows "overdue unpaid" counts every slip past its due date and not yet paid on time.
Slips still inside the grace period are in that number although they can still be paid on time
(stage 05 explains why). This stage recomputes the card for any number of periods, side by side:

* **as the card counts it**: overdue and not paid on time,
* **still in grace**: part of the above that is not lost yet,
* **lost**: overdue, unpaid and past the grace period, the only part that is definitive,
* the on-time rate **as the card computes it** (over all slips) against the rate **over mature
  slips**, whose denominator cannot move any more.

Closed periods have everything matured, so the two definitions agree there. The distortion lives
in the open period (month to date), which is the view people look at every day. Value is shown
both as the first installment (the cash that did not arrive) and as the whole agreement (what a
card summing agreement value displays); stage 04 shows how to tell which a headline uses.

All periods are evaluated from one row fetch over the widest span. ``--from-dataset`` computes the
same card offline from the dataset bundle.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from delinquency_eda.context import Context
from delinquency_eda.dataset import fetch_slips
from delinquency_eda.dates import date_argument, to_date
from delinquency_eda.derive import OVERDUE_UNPAID, PAID_ON_TIME, TO_MATURE
from delinquency_eda.errors import ConfigError
from delinquency_eda.report import Doc, fmt_int, fmt_money, fmt_rate
from delinquency_eda.stages.s00_period import Window, parse_window


@dataclass(frozen=True)
class CardRow:
    period: Window
    generated: int
    paid_on_time: int
    to_mature: int
    as_counted: int
    in_grace: int
    lost: int
    value_in_grace: float
    value_lost: float
    value_as_counted_whole: float
    value_lost_whole: float
    mature: int
    mature_paid: int

    @property
    def inflation(self) -> float:
        return self.as_counted / self.lost if self.lost else float("nan")

    @property
    def rate_as_card(self) -> float:
        return self.paid_on_time / self.generated if self.generated else float("nan")

    @property
    def rate_mature(self) -> float:
        return self.mature_paid / self.mature if self.mature else float("nan")


def default_periods(as_of: date) -> list[Window]:
    """The open month, then the two closed months before it."""
    first = as_of.replace(day=1)
    last_end = first - timedelta(days=1)
    last_start = last_end.replace(day=1)
    older_end = last_start - timedelta(days=1)
    return [
        Window("month to date", first, as_of),
        Window("previous month", last_start, last_end),
        Window("two months back", older_end.replace(day=1), older_end),
    ]


def _inside(slip: Mapping[str, Any], period: Window) -> bool:
    due = to_date(slip["due_date"])
    return due is not None and period.start <= due <= period.end


def card_row(slips: Sequence[Mapping[str, Any]], period: Window, grace_days: int) -> CardRow:
    """One period of the card. Slips need ``bucket``, ``days_overdue`` and the value fields."""
    rows = [s for s in slips if _inside(s, period)]
    unpaid = [s for s in rows if s["bucket"] == OVERDUE_UNPAID]
    lost = [s for s in unpaid if s["days_overdue"] > grace_days]
    grace = [s for s in unpaid if s["days_overdue"] <= grace_days]
    mature = [s for s in rows if s["days_overdue"] > grace_days]

    def money(group: Sequence[Mapping[str, Any]], field: str) -> float:
        return float(sum(s[field] or 0.0 for s in group))

    return CardRow(
        period=period,
        generated=len(rows),
        paid_on_time=sum(1 for s in rows if s["bucket"] == PAID_ON_TIME),
        to_mature=sum(1 for s in rows if s["bucket"] == TO_MATURE),
        as_counted=len(unpaid),
        in_grace=len(grace),
        lost=len(lost),
        value_in_grace=money(grace, "first_installment_value"),
        value_lost=money(lost, "first_installment_value"),
        value_as_counted_whole=money(unpaid, "total_value"),
        value_lost_whole=money(lost, "total_value"),
        mature=len(mature),
        mature_paid=sum(1 for s in mature if s["bucket"] == PAID_ON_TIME),
    )


def card(
    slips: Sequence[Mapping[str, Any]], periods: Sequence[Window], grace_days: int
) -> list[CardRow]:
    return [card_row(slips, p, grace_days) for p in periods]


def render(rows: Sequence[CardRow], as_of: date, grace_days: int) -> Doc:
    doc = Doc().h1("Summary card with a maturity cut")
    doc.p(
        f"Reference date {as_of.isoformat()}, grace period {grace_days} day(s). First-installment "
        "slips grouped by due date."
    )
    names = [r.period.name for r in rows]
    spans = [f"{r.period.start} to {r.period.end}" for r in rows]
    table = [
        ["window", *spans],
        ["generated", *[fmt_int(r.generated) for r in rows]],
        ["paid on time", *[fmt_int(r.paid_on_time) for r in rows]],
        ["to mature (not due yet)", *[fmt_int(r.to_mature) for r in rows]],
        ["overdue unpaid, as the card counts it", *[fmt_int(r.as_counted) for r in rows]],
        ["  of which still in grace", *[fmt_int(r.in_grace) for r in rows]],
        ["  of which lost (grace over)", *[fmt_int(r.lost) for r in rows]],
        ["inflation of the card count", *[_ratio(r.inflation) for r in rows]],
        ["first installment still in grace", *[fmt_money(r.value_in_grace) for r in rows]],
        ["first installment lost", *[fmt_money(r.value_lost) for r in rows]],
        [
            "whole agreement, as the card counts it",
            *[fmt_money(r.value_as_counted_whole) for r in rows],
        ],
        ["whole agreement lost", *[fmt_money(r.value_lost_whole) for r in rows]],
        ["on-time rate, as the card computes it", *[fmt_rate(r.rate_as_card) for r in rows]],
        ["on-time rate over mature slips", *[fmt_rate(r.rate_mature) for r in rows]],
        ["mature slips (denominator)", *[fmt_int(r.mature) for r in rows]],
    ]
    doc.table(table, ["indicator", *names])
    doc.p(
        "Where the inflation is 1.00x, everything has matured and both definitions agree. Any "
        "gap is the share of the bucket that is not lost yet."
    )
    return doc


def _ratio(value: float) -> str:
    return "-" if math.isnan(value) else f"{value:.2f}x"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--as-of", type=date_argument, help="reference date (default: today)")
    parser.add_argument(
        "--period",
        action="append",
        default=[],
        metavar="NAME=FROM:TO",
        help="due-date period, inclusive (repeatable; default: month to date and the two before)",
    )
    parser.add_argument(
        "--from-dataset",
        action="store_true",
        help="compute from the dataset bundle instead of querying the database",
    )


def _acquire(
    ctx: Context, args: argparse.Namespace, explicit: Sequence[Window]
) -> tuple[list[dict[str, Any]], date, int, list[Window]]:
    """Slips, reference date, grace days and the periods to evaluate."""
    if args.from_dataset:
        if args.as_of:
            raise ConfigError("--as-of cannot be used with --from-dataset; the dataset fixes it")
        dataset = ctx.load_dataset()
        as_of = dataset.meta.as_of_date
        periods = list(explicit) or default_periods(as_of)
        return dataset.base, as_of, dataset.meta.grace_days, periods
    as_of = args.as_of or date.today()
    periods = list(explicit) or default_periods(as_of)
    start, end = min(p.start for p in periods), max(p.end for p in periods)
    slips: list[dict[str, Any]] = []
    with ctx.databases() as databases:
        for line, db in databases.items():
            slips += fetch_slips(db, line, ctx.settings, start, end, as_of=as_of, detail=False)
    return slips, as_of, ctx.settings.grace_days, periods


def run(ctx: Context, args: argparse.Namespace) -> int:
    explicit = [parse_window(s) for s in args.period]
    slips, as_of, grace, periods = _acquire(ctx, args, explicit)
    rows = card(slips, periods, grace)
    for r in rows:
        ctx.say(
            f"{r.period.name:<18} generated={r.generated:<6} as_counted={r.as_counted:<5} "
            f"lost={r.lost:<5} on-time(card)={fmt_rate(r.rate_as_card)} "
            f"on-time(mature)={fmt_rate(r.rate_mature)}"
        )
    ctx.write_report("card.md", render(rows, as_of, grace))
    return 0
