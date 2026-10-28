"""Stage 02 - base build: extract the dataset every offline stage reads.

Four stages read a database (period, schema, base and card). This one writes the JSON bundle that
stages 03 to 08 and 10 read, so those run offline and repeatably. The bundle has four tables:

* ``base``     first-installment slips due in the period, each with its derived outcome,
* ``debtors``  every debtor row that shares an opaque person key with a base row,
* ``contacts`` outreach attempts for those debtors between the earliest agreement and last due date,
* ``breaks``   earlier first-installment agreements of those debtors that ended broken.

The outcome is derived in Python against the ``--as-of`` date (today's local date when the flag
is left out), never the database server's clock, so the same command with the same date always
gives the same dataset, and a later stage can replay the past by re-deriving at an earlier date.

This stage is also the gate. The build is judged against the entry contract (``contract.RULES``,
judged by ``admission.py``) and the dataset file is written only if every rule passes. A build that
fails writes ``admission.md`` and no dataset, prints why, and exits 1. The file already at the
dataset path, if any, is left untouched: it holds an earlier build that was admitted.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date

from delinquency_eda.admission import PASS, SKIPPED, admit
from delinquency_eda.admission import render as render_admission
from delinquency_eda.context import Context
from delinquency_eda.dataset import Dataset
from delinquency_eda.dates import date_argument
from delinquency_eda.derive import BUCKETS
from delinquency_eda.errors import DatasetError
from delinquency_eda.report import Doc, fmt_money
from delinquency_eda.stages.s01_schema import check_database


@dataclass(frozen=True)
class BaseSummary:
    slips: int
    by_line: dict[str, int]
    by_bucket: dict[str, int]
    people: int
    without_person_key: int
    debtor_rows: int
    contacts: int
    earlier_breaks: int
    first_installment_value: float


def summarize(dataset: Dataset) -> BaseSummary:
    """Counts a person can check against the source at a glance."""
    by_line = Counter(r["product_line"] for r in dataset.base)
    by_bucket = Counter(r["bucket"] for r in dataset.base)
    return BaseSummary(
        slips=len(dataset.base),
        by_line=dict(sorted(by_line.items())),
        by_bucket={b: by_bucket.get(b, 0) for b in BUCKETS},
        people=len({r["person_key"] for r in dataset.base if r["person_key"]}),
        without_person_key=sum(1 for r in dataset.base if not r["person_key"]),
        debtor_rows=len(dataset.debtors),
        contacts=len(dataset.contacts),
        earlier_breaks=len(dataset.breaks),
        first_installment_value=sum(r["first_installment_value"] or 0.0 for r in dataset.base),
    )


def render(dataset: Dataset, summary: BaseSummary) -> Doc:
    meta = dataset.meta
    doc = Doc().h1("Base build")
    parameters = [
        ["due-date window", f"{meta.period_from} to {meta.period_to}"],
        ["reference date (as of)", meta.as_of],
        ["grace days", meta.grace_days],
        ["product lines", ", ".join(meta.product_lines)],
        ["generated statuses", ", ".join(meta.generated_status_labels)],
        ["excluded agent patterns", ", ".join(meta.excluded_agent_patterns) or "(none)"],
    ]
    rules = (dataset.admission or {}).get("rules", [])
    if rules:
        passed = sum(1 for r in rules if r["status"] == PASS)
        verdict = f"{passed} of {len(rules)} rules passed (admission.md)"
        parameters.append(["entry contract", verdict])
    doc.table(parameters, ["parameter", "value"])
    doc.h2("Contents")
    doc.table(
        [
            ["slips", summary.slips],
            *[[f"  in {line}", n] for line, n in summary.by_line.items()],
            *[[f"  {bucket}", n] for bucket, n in summary.by_bucket.items()],
            ["first-installment value", fmt_money(summary.first_installment_value)],
            ["people (distinct person keys)", summary.people],
            ["slips without a person key", summary.without_person_key],
            ["debtor rows pulled in", summary.debtor_rows],
            ["contact events", summary.contacts],
            ["earlier broken agreements", summary.earlier_breaks],
        ],
        ["item", "count"],
    )
    doc.p(
        "The dataset file holds opaque keys and agent identifiers exactly as your database "
        "supplied them. Keep it out of version control (the default path is gitignored)."
    )
    return doc


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--from",
        dest="date_from",
        required=True,
        type=date_argument,
        help="first due date to include (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--to",
        dest="date_to",
        required=True,
        type=date_argument,
        help="last due date to include, inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--as-of", type=date_argument, help="reference date for outcomes (default: today)"
    )


def run(ctx: Context, args: argparse.Namespace) -> int:
    if args.date_to < args.date_from:
        raise DatasetError("the period end is before its start")
    as_of = args.as_of or date.today()
    with ctx.databases() as databases:
        schema = {name: check_database(db, name, ctx.settings) for name, db in databases.items()}
        admission = admit(
            databases,
            ctx.settings,
            schema,
            date_from=args.date_from,
            date_to=args.date_to,
            as_of=as_of,
        )
    kept = ctx.dataset_path.exists()
    dataset = admission.dataset
    if dataset is None:
        ctx.write_report(
            "admission.md",
            render_admission(admission, dataset_path=ctx.dataset_path, previous_file_kept=kept),
        )
        ctx.say(f"rejected: {len(admission.failed)} entry rule(s) failed; no dataset was written")
        for result in admission.failed:
            ctx.say(f"  {result.id}: {result.violations} violation(s)")
            for line in result.detail:
                ctx.say(f"    {line}")
        held_back = [r.id for r in admission.results if r.status == SKIPPED]
        if held_back:
            ctx.say(f"  not judged, because an earlier rule failed: {', '.join(held_back)}")
        if kept:
            ctx.say(f"  the file at {ctx.dataset_path} was left as it was")
        return 1
    summary = summarize(dataset)
    dataset.save(ctx.dataset_path)
    ctx.say(f"dataset written: {ctx.dataset_path} ({summary.slips} slips)")
    ctx.say(f"  entry contract: all {len(admission.results)} rules passed")
    for bucket, n in summary.by_bucket.items():
        ctx.say(f"  {bucket}: {n}")
    ctx.write_report("base.md", render(dataset, summary))
    ctx.write_report(
        "admission.md",
        render_admission(admission, dataset_path=ctx.dataset_path, previous_file_kept=kept),
    )
    return 0
