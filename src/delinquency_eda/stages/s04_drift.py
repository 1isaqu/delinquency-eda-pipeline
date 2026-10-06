"""Stage 04 - drift: replay the buckets at an earlier date and reconcile a headline figure.

A summary card shown on one day rarely matches a re-run made later, and the first question is
always "which slips moved?". This stage answers it offline, from the dataset alone. Each slip's
outcome is re-derived as of a reference date (a payment dated after that day did not exist yet,
and an agreement made after it did not exist either), then compared with the outcome now:

* bucket counts and values at the reference date and now,
* a transition matrix (bucket then -> bucket now),
* the movers: slips that left the overdue-unpaid bucket by paying on time since,
* which value measure a headline number sums, when you give the figure you were shown.

Limits, stated plainly: the dataset holds the latest status of each agreement, so a slip that was
re-written, broken or cancelled after the reference date cannot be rewound, and slips that no
longer pass the extraction filters are absent altogether.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date

import pandas as pd

from delinquency_eda import derive
from delinquency_eda.context import Context
from delinquency_eda.dates import parse_date_arg, to_date
from delinquency_eda.errors import ConfigError, DatasetError
from delinquency_eda.report import Doc, fmt_int, fmt_money

NOT_YET = "(not yet made)"
STATES = (derive.PAID_ON_TIME, derive.OVERDUE_UNPAID, derive.TO_MATURE, NOT_YET)
MEASURES = {
    "first installment": "first_installment_value",
    "whole agreement": "total_value",
    "original debt": "original_debt",
}
MOVER_COLUMNS = [
    "product_line",
    "agreement_id",
    "portfolio_id",
    "due_date",
    "paid_date",
    "first_installment_value",
    "total_value",
]


def bucket_at(df: pd.DataFrame, as_of: date, grace_days: int) -> pd.Series:
    """Each slip's bucket as it stood at ``as_of``; ``(not yet made)`` before its agreement."""
    out = []
    for r in df.itertuples(index=False):
        made, due = to_date(r.agreement_date), to_date(r.due_date)
        if due is None:
            raise DatasetError("a slip in the dataset has no due date")
        if made is not None and made > as_of:
            out.append(NOT_YET)
            continue
        paid = None if pd.isna(r.paid_amount) else float(r.paid_amount)
        out.append(
            derive.classify(due, to_date(r.paid_date), paid, as_of=as_of, grace_days=grace_days)
        )
    return pd.Series(out, index=df.index, dtype=object)


@dataclass
class DriftResult:
    reference: date
    now: date
    then: pd.Series
    current: pd.Series
    transitions: pd.DataFrame
    movers: pd.DataFrame
    entered: int

    def unpaid_then(self) -> int:
        return int((self.then == derive.OVERDUE_UNPAID).sum())

    def unpaid_now(self) -> int:
        return int((self.current == derive.OVERDUE_UNPAID).sum())


def replay(df: pd.DataFrame, reference: date, now: date, grace_days: int) -> DriftResult:
    """Compare the buckets at ``reference`` with those at ``now`` for the same slips."""
    if reference > now:
        raise ConfigError("the reference date must not be after the dataset reference date")
    then = bucket_at(df, reference, grace_days)
    current = bucket_at(df, now, grace_days)
    matrix = pd.crosstab(then, current).reindex(index=STATES, columns=STATES, fill_value=0)
    unpaid = derive.OVERDUE_UNPAID
    left = (then == unpaid) & (current != unpaid)
    entered = int(((then != unpaid) & (current == unpaid)).sum())
    movers = df.loc[left, MOVER_COLUMNS].sort_values(["paid_date", "agreement_id"])
    return DriftResult(reference, now, then, current, matrix, movers, entered)


def measure_sums(df: pd.DataFrame, bucket: pd.Series) -> dict[str, float]:
    """Sum of each value measure over the slips in the overdue-unpaid bucket."""
    unpaid = df[bucket == derive.OVERDUE_UNPAID]
    return {name: float(unpaid[column].sum()) for name, column in MEASURES.items()}


def closest_measure(headline: float, sums: dict[str, float]) -> list[tuple[str, float, float]]:
    """``(measure, sum, relative difference)`` ordered from the closest to the headline."""
    scored = [
        (m, v, abs(v - headline) / abs(headline) if headline else float("inf"))
        for m, v in sums.items()
    ]
    return sorted(scored, key=lambda row: row[2])


def _bucket_table(doc: Doc, df: pd.DataFrame, result: DriftResult) -> None:
    rows = []
    for state in STATES[:3]:
        a, b = result.then == state, result.current == state
        rows.append(
            [
                state,
                fmt_int(a.sum()),
                fmt_money(df.loc[a, "first_installment_value"].sum()),
                fmt_int(b.sum()),
                fmt_money(df.loc[b, "first_installment_value"].sum()),
            ]
        )
    doc.table(
        rows,
        [
            "bucket",
            f"slips on {result.reference}",
            "value then",
            f"slips on {result.now}",
            "value now",
        ],
    )


def _headline_section(
    doc: Doc, df: pd.DataFrame, result: DriftResult, value: float | None, count: int | None
) -> None:
    doc.h2("Which value does a headline figure sum?")
    dated = (
        (str(result.reference), measure_sums(df, result.then)),
        (str(result.now), measure_sums(df, result.current)),
    )
    doc.table(
        [[m, *[fmt_money(sums[m]) for _, sums in dated]] for m in MEASURES],
        ["measure", *[f"overdue-unpaid on {when}" for when, _ in dated]],
    )
    if value is not None:
        rows = [
            [when, m, fmt_money(total), f"{100 * rel:.2f}%"]
            for when, sums in dated
            for m, total, rel in closest_measure(value, sums)
        ]
        doc.p(f"Headline value given: {fmt_money(value)}. Relative difference to each measure:")
        doc.table(rows, ["date", "measure", "sum", "difference"])
        best = min(
            ((when, *closest_measure(value, sums)[0]) for when, sums in dated),
            key=lambda row: row[3],
        )
        doc.p(f"Closest: **{best[1]}** on {best[0]} ({100 * best[3]:.2f}% apart).")
    if count is not None:
        doc.p(
            f"Headline count given: {count}. Overdue-unpaid slips on {result.reference}: "
            f"{result.unpaid_then()}; on {result.now}: {result.unpaid_now()}."
        )


def render(
    df: pd.DataFrame,
    result: DriftResult,
    *,
    headline_value: float | None = None,
    headline_count: int | None = None,
    max_movers: int = 20,
) -> Doc:
    doc = Doc().h1("Drift: the same slips, two dates")
    doc.p(
        f"Outcomes re-derived as of {result.reference} and as of {result.now}. Payments dated "
        "after a date are ignored for that date; agreements made after it are not counted."
    )
    doc.h2("Buckets")
    _bucket_table(doc, df, result)
    doc.h2("Transitions")
    doc.p("Rows: bucket on the reference date. Columns: bucket now.")
    doc.table(
        [[state, *result.transitions.loc[state].tolist()] for state in STATES],
        ["then \\ now", *STATES],
    )
    doc.p(
        f"{len(result.movers)} slip(s) left the overdue-unpaid bucket and {result.entered} "
        "entered it; together they account for the whole difference between the two counts."
    )
    if len(result.movers):
        doc.h2(f"Movers (up to {max_movers})")
        movers = result.movers.head(max_movers).copy()
        for column in ("agreement_id", "portfolio_id"):
            movers[column] = movers[column].astype(str)
        doc.table(movers)
    _headline_section(doc, df, result, headline_value, headline_count)
    doc.h2("Limits")
    doc.bullets(
        [
            "Only the latest status of each agreement is stored: later breaks, cancellations and "
            "re-writes cannot be rewound.",
            "Slips that fail the extraction filters today are absent on both dates.",
            "A slip counts as paid on a date only if its payment date is on or before it.",
        ]
    )
    return doc


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reference-date",
        required=True,
        type=parse_date_arg,
        help="the earlier date to replay (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--headline-value",
        type=float,
        help="a value you were shown, to identify which measure it sums",
    )
    parser.add_argument("--headline-count", type=int, help="a count you were shown")
    parser.add_argument("--max-movers", type=int, default=20, help="rows listed under Movers")


def run(ctx: Context, args: argparse.Namespace) -> int:
    dataset = ctx.load_dataset()
    df = dataset.base_frame()
    result = replay(df, args.reference_date, dataset.meta.as_of_date, dataset.meta.grace_days)
    ctx.say(
        f"overdue_unpaid on {result.reference}: {result.unpaid_then()}, now: "
        f"{result.unpaid_now()}; left: {len(result.movers)}, entered: {result.entered}"
    )
    doc = render(
        df,
        result,
        headline_value=args.headline_value,
        headline_count=args.headline_count,
        max_movers=args.max_movers,
    )
    ctx.write_report("drift.md", doc)
    return 0
