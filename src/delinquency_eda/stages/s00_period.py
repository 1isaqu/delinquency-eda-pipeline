"""Stage 00 - period selection: which date window reproduces the funnel you were shown?

Every later number depends on the window, and a summary card rarely says which window it used
("month to date" up to today or up to yesterday? the whole previous month?). This stage evaluates
the standard funnel over several candidate windows and tells you which one is closest to the
counts you already have.

The funnel partitions first-installment slips due in the window:

    generated = paid_on_time + overdue_unpaid + to_mature

The slips are fetched **once** over the widest span and every window is evaluated in Python, so
adding a candidate costs nothing and the reconciliation does not depend on how many times the
database was asked.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from delinquency_eda import derive
from delinquency_eda.context import Context
from delinquency_eda.dataset import fetch_slips
from delinquency_eda.dates import parse_date_arg, to_date
from delinquency_eda.errors import ConfigError
from delinquency_eda.report import Doc, fmt_money

FUNNEL_KEYS = ("generated", "to_mature", "overdue_unpaid", "paid_on_time")


@dataclass(frozen=True)
class Window:
    name: str
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ConfigError(f"window {self.name!r} ends before it starts")


@dataclass(frozen=True)
class Funnel:
    generated: int
    to_mature: int
    overdue_unpaid: int
    paid_on_time: int
    overdue_unpaid_value: float
    amount_due: float
    amount_received: float

    def count(self, key: str) -> int:
        return int(getattr(self, key))


def parse_window(spec: str) -> Window:
    """``NAME=YYYY-MM-DD:YYYY-MM-DD`` -> a window (both ends inclusive)."""
    name, sep, span = spec.partition("=")
    start_text, colon, end_text = span.partition(":")
    if not (sep and colon and name.strip()):
        raise ConfigError(f"a window must look like NAME=YYYY-MM-DD:YYYY-MM-DD, got {spec!r}")
    return Window(name.strip(), parse_date_arg(start_text), parse_date_arg(end_text))


def parse_reference(specs: Sequence[str]) -> dict[str, int]:
    """``generated=120`` style pairs -> counts to compare against. Any subset of ``FUNNEL_KEYS``."""
    reference: dict[str, int] = {}
    for spec in specs:
        key, sep, value = spec.partition("=")
        key = key.strip()
        if not sep or key not in FUNNEL_KEYS:
            raise ConfigError(f"a reference must look like KEY=COUNT with KEY in {FUNNEL_KEYS}")
        try:
            reference[key] = int(value)
        except ValueError:
            raise ConfigError(f"reference {key!r} needs a whole number, got {value!r}") from None
    return reference


def default_windows(as_of: date) -> list[Window]:
    """The windows a summary card usually means, relative to the reference date.

    Candidates that would be empty (month to yesterday on the first of the month) or that repeat
    another candidate's span are left out.
    """
    first = as_of.replace(day=1)
    previous_end = first - timedelta(days=1)
    yesterday = as_of - timedelta(days=1)
    spans = [
        ("month to date", first, as_of),
        ("today only", as_of, as_of),
        ("month to yesterday", first, yesterday),
        ("previous month", previous_end.replace(day=1), previous_end),
        ("last 7 days", as_of - timedelta(days=6), as_of),
        ("last 30 days", as_of - timedelta(days=29), as_of),
    ]
    seen: set[tuple[date, date]] = set()
    windows = []
    for name, start, end in spans:
        if end >= start and (start, end) not in seen:
            seen.add((start, end))
            windows.append(Window(name, start, end))
    return windows


def _due_inside(slip: Mapping[str, Any], window: Window) -> bool:
    due = to_date(slip["due_date"])
    return due is not None and window.start <= due <= window.end


def funnel(slips: Sequence[Mapping[str, Any]], window: Window) -> Funnel:
    """The funnel for the slips whose due date falls inside ``window``."""
    inside = [s for s in slips if _due_inside(s, window)]
    counts = dict.fromkeys(derive.BUCKETS, 0)
    lost_value = due = received = 0.0
    for slip in inside:
        counts[slip["bucket"]] += 1
        value = slip["first_installment_value"] or 0.0
        due += value
        if slip["bucket"] == derive.OVERDUE_UNPAID:
            lost_value += value
        paid = slip["paid_amount"]
        if paid and paid > 0 and slip["paid_date"]:
            received += paid
    return Funnel(
        generated=len(inside),
        to_mature=counts[derive.TO_MATURE],
        overdue_unpaid=counts[derive.OVERDUE_UNPAID],
        paid_on_time=counts[derive.PAID_ON_TIME],
        overdue_unpaid_value=lost_value,
        amount_due=due,
        amount_received=received,
    )


def distance(result: Funnel, reference: Mapping[str, int]) -> int:
    """Sum of absolute count differences over the reference keys that were given."""
    return sum(abs(result.count(key) - wanted) for key, wanted in reference.items())


def rank(results: Mapping[str, Funnel], reference: Mapping[str, int]) -> list[tuple[str, int]]:
    """Window names with their distance to ``reference``, closest first (stable on ties)."""
    scored = [(name, distance(result, reference)) for name, result in results.items()]
    return sorted(scored, key=lambda pair: pair[1])


def render(
    windows: Sequence[Window],
    results: Mapping[str, Funnel],
    reference: Mapping[str, int],
    as_of: date,
) -> Doc:
    doc = Doc().h1("Period selection")
    doc.p(
        f"Reference date: {as_of.isoformat()}. First-installment slips, grouped by due date. "
        "Each row is one candidate window; the funnel is evaluated on the same rows."
    )
    headers = ["window", "from", "to", *FUNNEL_KEYS, "unpaid value"]
    rows = [
        [w.name, w.start.isoformat(), w.end.isoformat()]
        + [results[w.name].count(k) for k in FUNNEL_KEYS]
        + [fmt_money(results[w.name].overdue_unpaid_value)]
        for w in windows
    ]
    doc.table(rows, headers)
    if reference:
        ordered = rank(results, reference)
        doc.h2("Closest match to the reference counts")
        doc.p("Reference: " + ", ".join(f"{k} = {v}" for k, v in reference.items()) + ".")
        doc.table(
            [
                [name, dist, *(results[name].count(k) - reference.get(k, 0) for k in reference)]
                for name, dist in ordered
            ],
            ["window", "total distance", *(f"{k} diff" for k in reference)],
        )
        best, best_distance = ordered[0]
        verdict = "an exact match" if best_distance == 0 else f"{best_distance} counts away"
        doc.p(f"Closest window: **{best}** ({verdict}).")
        doc.p(
            "A non-zero distance is expected when the reference was taken earlier than the "
            "reference date: slips paid in between leave the unpaid bucket (see the drift stage)."
        )
    return doc


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--as-of", type=parse_date_arg, help="reference date (default: today)")
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        metavar="NAME=FROM:TO",
        help="candidate window by due date, inclusive (repeatable; default: usual card windows)",
    )
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="KEY=COUNT",
        help=f"a count to reproduce; KEY is one of {', '.join(FUNNEL_KEYS)} (repeatable)",
    )


def run(ctx: Context, args: argparse.Namespace) -> int:
    as_of = args.as_of or date.today()
    windows = [parse_window(s) for s in args.window] or default_windows(as_of)
    if len({w.name for w in windows}) != len(windows):
        raise ConfigError("window names must be unique")
    reference = parse_reference(args.reference)
    span_from, span_to = min(w.start for w in windows), max(w.end for w in windows)
    slips: list[dict[str, Any]] = []
    with ctx.databases() as databases:
        for line, db in databases.items():
            slips += fetch_slips(
                db, line, ctx.settings, span_from, span_to, as_of=as_of, detail=False
            )
    results = {w.name: funnel(slips, w) for w in windows}
    doc = render(windows, results, reference, as_of)
    for window in windows:
        r = results[window.name]
        ctx.say(
            f"{window.name:<22} generated={r.generated:<6} to_mature={r.to_mature:<5} "
            f"overdue_unpaid={r.overdue_unpaid:<5} paid_on_time={r.paid_on_time}"
        )
    if reference:
        best, best_distance = rank(results, reference)[0]
        ctx.say(f"closest window: {best} (distance {best_distance})")
    ctx.write_report("period.md", doc)
    return 0
