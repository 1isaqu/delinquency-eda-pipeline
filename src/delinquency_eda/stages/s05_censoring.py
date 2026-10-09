"""Stage 05 - maturity censoring: how much of the "overdue unpaid" bucket is not lost yet?

The bucket is defined by two clocks that disagree. A slip is *overdue* the day after its due
date, but it is only *paid on time* if the money arrives within the grace period after that date.
So a slip that fell due yesterday sits in the overdue-unpaid bucket today while still having
``grace_days - 1`` days left to be paid on time. It is right-censored: its outcome is not known
yet. Counting it as lost inflates the bucket, and only for the most recent due dates, the ones a
period that is still open (month to date) holds.

This stage splits the bucket into

* **still open**: overdue, unpaid, grace period not over (``days_overdue <= grace_days``),
* **definitive**: overdue, unpaid, grace period over,

reports the inflation factor of the naive count, and recomputes the on-time rate over mature
slips only, which is the denominator that cannot move any more.

A mature slip's outcome is final, and maturity depends on the due date alone. Selecting "paid, or
unpaid and old" would pick slips by their outcome and bias every rate computed on them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from delinquency_eda import features as F
from delinquency_eda.context import Context
from delinquency_eda.derive import OVERDUE_UNPAID, PAID_ON_TIME
from delinquency_eda.report import Doc, fmt_int, fmt_money, fmt_pct, fmt_rate

OPEN = "still open (grace period not over)"
FINAL = "definitive (grace period over)"


@dataclass
class CensoringResult:
    grace_days: int
    unpaid: int
    still_open: int
    definitive: int
    inflation: float
    naive_on_time_rate: float
    mature_on_time_rate: float
    mature: int
    value_open: float
    value_definitive: float
    by_due_date: pd.DataFrame
    by_line: pd.DataFrame


def _rate(part: int, whole: int) -> float:
    return part / whole if whole else float("nan")


def _due_table(unpaid: pd.DataFrame, grace_days: int) -> pd.DataFrame:
    grouped = unpaid.groupby("due_date").agg(
        age=("days_overdue", "first"),
        slips=("record_id", "size"),
        value=("first_installment_value", "sum"),
    )
    grouped["age"] = grouped["age"].astype(int)
    grouped["state"] = np.where(grouped["age"] > grace_days, FINAL, OPEN)
    return grouped.reset_index().sort_values("due_date")


def _line_table(unpaid: pd.DataFrame, grace_days: int) -> pd.DataFrame:
    flagged = unpaid.assign(final=unpaid["days_overdue"] > grace_days)
    out = flagged.groupby("product_line").agg(
        unpaid=("record_id", "size"), definitive=("final", "sum")
    )
    out["still_open"] = out["unpaid"] - out["definitive"]
    return out.reset_index()


def measure(df: pd.DataFrame, grace_days: int) -> CensoringResult:
    """Censoring figures for a base frame (``bucket`` and ``days_overdue`` are required)."""
    unpaid = df[df["bucket"] == OVERDUE_UNPAID]
    final = unpaid["days_overdue"] > grace_days
    mature = df[F.mature_mask(df, grace_days)]
    paid_on_time = int((df["bucket"] == PAID_ON_TIME).sum())
    mature_paid = int((mature["bucket"] == PAID_ON_TIME).sum())
    definitive = int(final.sum())
    return CensoringResult(
        grace_days=grace_days,
        unpaid=len(unpaid),
        still_open=len(unpaid) - definitive,
        definitive=definitive,
        inflation=len(unpaid) / definitive if definitive else float("nan"),
        naive_on_time_rate=_rate(paid_on_time, len(df)),
        mature_on_time_rate=_rate(mature_paid, len(mature)),
        mature=len(mature),
        value_open=float(unpaid.loc[~final, "first_installment_value"].sum()),
        value_definitive=float(unpaid.loc[final, "first_installment_value"].sum()),
        by_due_date=_due_table(unpaid, grace_days),
        by_line=_line_table(unpaid, grace_days),
    )


CONTEXT_DAYS = 7


def _due_date_table(doc: Doc, result: CensoringResult) -> None:
    """The recent due dates in full; everything older is definitive and summarized."""
    table = result.by_due_date
    recent = table[table["age"] <= result.grace_days + CONTEXT_DAYS]
    older = table[table["age"] > result.grace_days + CONTEXT_DAYS]
    doc.h2("By due date")
    doc.table(
        [
            [r.due_date, r.age, fmt_int(r.slips), fmt_money(r.value), r.state]
            for r in recent.itertuples(index=False)
        ],
        ["due date", "days overdue", "slips", "first-installment value", "state"],
    )
    if len(older):
        doc.p(
            f"{fmt_int(len(older))} older due date(s) are all definitive: "
            f"{fmt_int(older['slips'].sum())} slips, {fmt_money(older['value'].sum())}."
        )


def render(result: CensoringResult, total_slips: int) -> Doc:
    g = result.grace_days
    doc = Doc().h1("Maturity censoring")
    doc.p(
        f"The grace period is {g} day(s). A slip due yesterday is overdue today but can still be "
        f"paid on time for {max(g - 1, 0)} more day(s), so it is not lost yet."
    )
    doc.h2("The overdue-unpaid bucket, split")
    total_value = result.value_open + result.value_definitive
    doc.table(
        [
            [
                OPEN,
                fmt_int(result.still_open),
                fmt_pct(result.still_open, result.unpaid),
                fmt_money(result.value_open),
                fmt_pct(result.value_open, total_value),
            ],
            [
                FINAL,
                fmt_int(result.definitive),
                fmt_pct(result.definitive, result.unpaid),
                fmt_money(result.value_definitive),
                fmt_pct(result.value_definitive, total_value),
            ],
            [
                "**bucket as the card counts it**",
                f"**{fmt_int(result.unpaid)}**",
                "**100.0%**",
                f"**{fmt_money(total_value)}**",
                "**100.0%**",
            ],
        ],
        ["state", "slips", "% of slips", "first-installment value", "% of value"],
    )
    inflation = (
        "undefined (nothing is definitive yet)"
        if np.isnan(result.inflation)
        else (f"{result.inflation:.2f}x")
    )
    doc.p(f"Inflation of the naive count over the definitive count: {inflation}.")
    doc.h2("On-time rate with and without the cut")
    doc.table(
        [
            [
                "naive: paid on time / all slips",
                fmt_int(total_slips),
                fmt_rate(result.naive_on_time_rate),
            ],
            [
                "mature only: paid on time / mature slips",
                fmt_int(result.mature),
                fmt_rate(result.mature_on_time_rate),
            ],
        ],
        ["rate", "denominator", "value"],
    )
    doc.p(
        "The naive rate puts slips whose outcome is unknown in the denominator, so it moves "
        "every day for reasons unrelated to collection performance. The mature rate only changes "
        "when a new due date matures."
    )
    _due_date_table(doc, result)
    doc.h2("By product line")
    doc.table(
        [
            [r.product_line, fmt_int(r.unpaid), fmt_int(r.still_open), fmt_int(r.definitive)]
            for r in result.by_line.itertuples(index=False)
        ],
        ["product line", "overdue-unpaid", "still open", "definitive"],
    )
    doc.p(
        "Rule of thumb: any indicator that compares today with an outcome that has a grace "
        "period needs a maturity cut. Without it, the most recent days always look worse."
    )
    return doc


def add_arguments(parser: argparse.ArgumentParser) -> None:
    del parser  # no options of its own; reads the dataset


def run(ctx: Context, args: argparse.Namespace) -> int:
    del args
    dataset = ctx.load_dataset()
    df = dataset.base_frame()
    result = measure(df, dataset.meta.grace_days)
    ctx.say(
        f"overdue_unpaid={result.unpaid} still_open={result.still_open} "
        f"definitive={result.definitive} inflation={result.inflation:.2f}x"
    )
    ctx.write_report("censoring.md", render(result, len(df)))
    return 0
