"""Admission: does the data meet the entry contract? If it does not, it does not enter.

The ``base`` stage is the only way rows reach the offline stages, so it is the gate. A build is
judged against the rules in ``contract.RULES`` and the dataset file is written only if every one
passes. There is no option to let a failing build through: the fix belongs in the view that maps
the database onto the contract, or in the settings.

The rules fall in four groups, run in this order:

1. *source*: the tables, columns, status labels and declared column types the contract asks for.
   Nothing is extracted from a database that fails these, because the queries would not mean what
   they say.
2. *rows*: every row as the driver returned it (``intake.py``): required values, readable values,
   a coherent payment.
3. *dataset*: the grain (one slip per agreement) and the domain of the derived columns.
4. *source again*: a second, independent route to the numbers (``queries.build_reconcile_*``)
   compared with what the extraction produced, plus the links the contract promises between
   tables. The two routes share no join and no ranking function, so a duplicate key in a
   reference table, which multiplies rows in the extraction, shows up as a difference.

Every result is a count. The report and the record stored in the dataset hold numbers and column
names only, never a key, an id or a value from a row.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from delinquency_eda import __version__
from delinquency_eda.config import Settings
from delinquency_eda.contract import COLUMN_KINDS, CONTRACT_VERSION, RULES, Rule, rule_for
from delinquency_eda.dataset import Dataset, build_dataset, debtor_ids_by_line, history_window
from delinquency_eda.dates import to_float
from delinquency_eda.db.base import Database
from delinquency_eda.derive import BUCKETS
from delinquency_eda.intake import INTAKE_RULES, Intake
from delinquency_eda.queries import (
    build_reconcile_breaks_query,
    build_reconcile_contacts_query,
    build_reconcile_slips_query,
)
from delinquency_eda.report import Doc, fmt_int, fmt_money
from delinquency_eda.sqlutil import chunked

if TYPE_CHECKING:  # named for typing only: the data layer does not import a stage
    from delinquency_eda.stages.s01_schema import LineSchema

PASS, FAIL, SKIPPED = "pass", "fail", "skipped"

# Money is summed in the database and in Python in different orders, and a currency has cents.
MONEY_TOLERANCE = 0.01


@dataclass(frozen=True)
class RuleResult:
    """How one rule came out: items examined, items that broke it, and where."""

    id: str
    status: str
    checked: int = 0
    violations: int = 0
    detail: tuple[str, ...] = ()

    @property
    def rule(self) -> Rule:
        return rule_for(self.id)

    def as_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "checked": self.checked,
            "violations": self.violations,
            "detail": list(self.detail),
        }


def judged(
    rule_id: str, checked: int, problems: Sequence[str], violations: int | None = None
) -> RuleResult:
    """A result that passes when there are no ``problems``."""
    broken = len(problems) if violations is None else violations
    return RuleResult(rule_id, FAIL if broken else PASS, checked, broken, tuple(problems))


def skipped(rule_id: str, reason: str) -> RuleResult:
    return RuleResult(rule_id, SKIPPED, detail=(reason,))


@dataclass(frozen=True)
class LedgerLine:
    """Where the rows of one product line went between the table and the dataset."""

    product_line: str
    window: int  # installment-0 rows due in the window, every status, every agent
    status: int  # after the status filter
    agents: int  # after the excluded agents
    slips: int  # after only the latest write of each installment is kept
    slips_value: float
    unknown_agent: int  # slips whose agent has no row in agents (the patterns cannot match them)


@dataclass
class Admission:
    """The verdict on one build, and the dataset when it was admitted."""

    results: list[RuleResult]
    ledger: list[LedgerLine] = field(default_factory=list)
    queries: list[dict[str, Any]] = field(default_factory=list)
    dataset: Dataset | None = None

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(r.status == PASS for r in self.results)

    @property
    def failed(self) -> list[RuleResult]:
        return [r for r in self.results if r.status == FAIL]

    def record(self) -> dict[str, Any]:
        """The part of the verdict that travels inside the dataset file."""
        return {
            "contract_version": CONTRACT_VERSION,
            "package_version": __version__,
            "passed": self.passed,
            "rules": [r.as_record() for r in self.results],
            "ledger": [asdict(line) for line in self.ledger],
            "queries": self.queries,
        }


# --------------------------------------------------------------------------------------------
# Group 1: the source
# --------------------------------------------------------------------------------------------


def source_results(schema: Mapping[str, LineSchema]) -> list[RuleResult]:
    """``source.tables``, ``source.labels`` and ``source.types`` from the schema check."""
    tables: list[str] = []
    types: list[str] = []
    labels: list[str] = []
    table_count = type_count = label_count = 0
    catalog_readable = True
    for line, check in schema.items():
        for table in check.tables:
            table_count += 1
            where = f"{line}.{table.spec.name}"
            if not table.exists:
                tables.append(f"{where}: table not found")
            elif table.missing_required:
                tables.append(f"{where}: missing {', '.join(table.missing_required)}")
            kinds = COLUMN_KINDS.get(table.spec.name, {})
            type_count += sum(1 for c in table.columns if c.name.lower() in kinds)
            types += [
                f"{where}.{m.column}: declared {m.found!r}, the contract needs {m.kind}"
                for m in table.wrong_types
            ]
        if check.labels_checked:
            label_count += check.labels_total
            labels += [f"{line}: no status labelled {x!r}" for x in check.missing_labels]
        else:
            catalog_readable = False
    return [
        judged("source.tables", table_count, tables),
        judged("source.types", type_count, types),
        judged("source.labels", label_count, labels)
        if catalog_readable
        else skipped("source.labels", "the status catalog is missing or incomplete"),
    ]


# --------------------------------------------------------------------------------------------
# Groups 2 and 3: the rows and the dataset
# --------------------------------------------------------------------------------------------


def intake_results(intake: Intake) -> list[RuleResult]:
    out = []
    for rule_id in INTAKE_RULES:
        tally = intake.tallies[rule_id]
        problems = [f"{name}: {n} row(s)" for name, n in sorted(tally.detail.items())]
        out.append(judged(rule_id, tally.checked, problems, tally.violations))
    return out


def check_grain(dataset: Dataset) -> RuleResult:
    """One slip per agreement, one per row id, one debtor row per debtor id."""
    seen: dict[str, set[Any]] = {"agreement": set(), "record": set(), "debtor": set()}
    repeats: Counter[str] = Counter()
    for row in dataset.base:
        keys = {
            "agreement": (row["product_line"], row["agreement_id"], row["portfolio_id"]),
            "record": (row["product_line"], row["record_id"]),
        }
        for name, key in keys.items():
            if key in seen[name]:
                repeats[name] += 1
            seen[name].add(key)
    for row in dataset.debtors:
        key = (row["product_line"], row["debtor_id"])
        if key in seen["debtor"]:
            repeats["debtor"] += 1
        seen["debtor"].add(key)
    labels = {
        "agreement": "slips that repeat an agreement and portfolio",
        "record": "slips that repeat a row id",
        "debtor": "debtor rows that repeat a debtor id",
    }
    problems = [f"{labels[name]}: {n}" for name, n in repeats.items()]
    return judged("key.grain", 2 * len(dataset.base) + len(dataset.debtors), problems)


def check_domain(dataset: Dataset) -> RuleResult:
    """Every slip carries what the build asked for: known bucket, status, line, due window."""
    meta = dataset.meta
    generated = {label.strip() for label in meta.generated_status_labels}
    lines = set(meta.product_lines)
    flawed: Counter[str] = Counter()
    rows_with_flaws = 0
    for row in dataset.base:
        flaws = []
        if row["bucket"] not in BUCKETS:
            flaws.append("a bucket the pipeline does not define")
        if str(row["status_label"] or "").strip() not in generated:
            flaws.append("a status label outside generated_status_labels")
        if row["product_line"] not in lines:
            flaws.append("a product line that was not requested")
        if not meta.period_from <= str(row["due_date"]) <= meta.period_to:
            flaws.append("a due date outside the requested window")
        flawed.update(flaws)
        rows_with_flaws += bool(flaws)
    problems = [f"slips with {what}: {n}" for what, n in flawed.items()]
    return judged("base.domain", len(dataset.base), problems, rows_with_flaws)


# --------------------------------------------------------------------------------------------
# Group 4: the second route
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Measure:
    n: int
    total: float


class _Findings:
    """What the second route found, added up over the product lines."""

    def __init__(self) -> None:
        self.problems: dict[str, list[str]] = {
            "row.due_date": [],
            "recon.slips": [],
            "recon.outcome": [],
            "recon.value": [],
            "recon.history": [],
        }
        self.comparisons: Counter[str] = Counter()
        self.slips = self.debtor_orphans = self.events = self.without_result = 0
        self.unreadable_due = 0

    def compare(self, rule_id: str, what: str, ours: float, theirs: float, *, money=False) -> None:
        self.comparisons[rule_id] += 1
        same = _same_money(ours, theirs) if money else ours == theirs
        if not same:
            show = fmt_money if money else fmt_int
            self.problems[rule_id].append(
                f"{what}: the dataset has {show(ours)}, the second query {show(theirs)}"
            )


def _measures(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], _Measure]:
    return {
        (str(r["kind"]), str(r["name"])): _Measure(int(r["n"] or 0), to_float(r["total"]) or 0.0)
        for r in rows
    }


def _same_money(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=MONEY_TOLERANCE)


def _compare_slips(
    line: str,
    rows: Sequence[Mapping[str, Any]],
    found: Mapping[tuple[str, str], _Measure],
    out: _Findings,
) -> LedgerLine:
    """The slips of one product line against the second query: count, buckets, values."""

    def get(kind: str, name: str) -> _Measure:
        return found.get((kind, name), _Measure(0, 0.0))

    latest, agreement = get("funnel", "latest"), get("agreement", "installments")
    out.compare("recon.slips", line, len(rows), latest.n)
    in_bucket = Counter(r["bucket"] for r in rows)
    for bucket in BUCKETS:
        out.compare("recon.outcome", f"{line} {bucket}", in_bucket[bucket], get("bucket", bucket).n)
    out.compare(
        "recon.value",
        f"{line} first-installment value",
        sum(r["first_installment_value"] or 0.0 for r in rows),
        latest.total,
        money=True,
    )
    out.compare(
        "recon.value",
        f"{line} whole-agreement value",
        sum(r["total_value"] or 0.0 for r in rows),
        agreement.total,
        money=True,
    )
    out.compare(
        "recon.value",
        f"{line} installments",
        sum(r["total_installments"] or 0 for r in rows),
        agreement.n,
    )
    out.comparisons["row.due_date"] += get("first_installments", "agreements").n
    if unreadable := get("unreadable_due", "agreements").n:
        out.problems["row.due_date"].append(
            f"{line}: first-installment rows with no readable due date: {unreadable}"
        )
        out.unreadable_due += unreadable
    out.slips += len(rows)
    out.debtor_orphans += get("debtor_orphans", "debtors").n
    return LedgerLine(
        line,
        get("funnel", "window").n,
        get("funnel", "status").n,
        get("funnel", "agents").n,
        latest.n,
        latest.total,
        get("unknown_agent", "agents").n,
    )


def _compare_history(
    line: str,
    db: Database,
    dataset: Dataset,
    intake: Intake,
    settings: Settings,
    date_from: date,
    debtor_ids: Sequence[Any],
    out: _Findings,
) -> None:
    """The contact events and earlier breaks pulled in for one product line."""
    first, last = history_window(dataset.base, date_from)
    events = missing = breaks = 0
    for part in chunked(debtor_ids, settings.id_chunk_size):
        contacts = build_reconcile_contacts_query(db.dialect, part, first, last, db.schema)
        for row in intake.run(db, line, "reconcile_contacts", contacts):
            if row["name"] == "events":
                events += int(row["n"] or 0)
            else:
                missing += int(row["n"] or 0)
        earlier = build_reconcile_breaks_query(db.dialect, settings, part, db.schema)
        breaks += sum(int(r["n"] or 0) for r in intake.run(db, line, "reconcile_breaks", earlier))
    out.events += events
    out.without_result += missing
    out.compare(
        "recon.history",
        f"{line} contact events",
        sum(1 for r in dataset.contacts if r["product_line"] == line),
        events - missing,
    )
    out.compare(
        "recon.history",
        f"{line} earlier breaks",
        sum(1 for r in dataset.breaks if r["product_line"] == line),
        breaks,
    )


def reconcile(
    databases: Mapping[str, Database],
    settings: Settings,
    dataset: Dataset,
    intake: Intake,
    *,
    date_from: date,
    date_to: date,
    as_of: date,
) -> tuple[list[RuleResult], list[LedgerLine]]:
    """Compare the dataset with the independent queries; return the results and the ledger."""
    out = _Findings()
    ledger: list[LedgerLine] = []
    debtor_ids = debtor_ids_by_line(dataset)
    for line, db in databases.items():
        query = build_reconcile_slips_query(
            db.dialect, settings, date_from, date_to, as_of, db.schema
        )
        found = _measures(intake.run(db, line, "reconcile_slips", query))
        rows = [r for r in dataset.base if r["product_line"] == line]
        ledger.append(_compare_slips(line, rows, found, out))
        if dataset.base:
            _compare_history(
                line, db, dataset, intake, settings, date_from, debtor_ids[line], out
            )

    def note(count: int, what: str) -> list[str]:
        return [f"{what}: {count}"] if count else []

    results = [
        judged(
            "ref.debtor",
            out.slips,
            note(out.debtor_orphans, "slips whose debtor has no row in debtors"),
            out.debtor_orphans,
        ),
        judged(
            "ref.contact_result",
            out.events,
            note(out.without_result, "contact events whose result is not in the catalog"),
            out.without_result,
        ),
        judged(
            "row.due_date",
            out.comparisons["row.due_date"],
            out.problems["row.due_date"],
            out.unreadable_due,
        ),
        *(
            judged(rule_id, out.comparisons[rule_id], problems)
            for rule_id, problems in out.problems.items()
            if rule_id != "row.due_date"
        ),
    ]
    return results, ledger


# --------------------------------------------------------------------------------------------
# The whole judgement
# --------------------------------------------------------------------------------------------

_AFTER_SOURCE = (*INTAKE_RULES, "key.grain", "base.domain")
_NEEDS_CLEAN_ROWS = (
    "row.due_date",
    "ref.debtor",
    "ref.contact_result",
    "recon.slips",
    "recon.outcome",
    "recon.value",
    "recon.history",
)


def admit(
    databases: Mapping[str, Database],
    settings: Settings,
    schema: Mapping[str, LineSchema],
    *,
    date_from: date,
    date_to: date,
    as_of: date,
) -> Admission:
    """Judge one build against every entry rule. The dataset is on the result only if it passed.

    ``schema`` is the schema check of each product line (``s01_schema.check_database``). When the
    source fails, nothing is extracted and the later rules are reported as skipped.
    """
    results = source_results(schema)
    if any(r.status != PASS for r in results):
        results += [skipped(rid, "the source failed its own rules") for rid in _AFTER_SOURCE]
        results += [skipped(rid, "the source failed its own rules") for rid in _NEEDS_CLEAN_ROWS]
        return Admission(_in_catalogue_order(results))

    intake = Intake()
    dataset = build_dataset(
        databases, settings, date_from=date_from, date_to=date_to, as_of=as_of, intake=intake
    )
    results += intake_results(intake)
    results += [check_grain(dataset), check_domain(dataset)]

    ledger: list[LedgerLine] = []
    if all(r.status == PASS for r in results if r.id in INTAKE_RULES):
        more, ledger = reconcile(
            databases, settings, dataset, intake, date_from=date_from, date_to=date_to, as_of=as_of
        )
        results += more
    else:
        results += [
            skipped(rid, "rows were left out at intake, so every count would differ")
            for rid in _NEEDS_CLEAN_ROWS
        ]

    admission = Admission(_in_catalogue_order(results), ledger, intake.queries())
    if admission.passed:
        dataset.admission = admission.record()
        admission.dataset = dataset
    return admission


def _in_catalogue_order(results: list[RuleResult]) -> list[RuleResult]:
    order = {rule.id: i for i, rule in enumerate(RULES)}
    return sorted(results, key=lambda r: order[r.id])


# --------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------


def _shown(status: str) -> str:
    return status if status == PASS else status.upper()


def render(admission: Admission, *, dataset_path: Path, previous_file_kept: bool) -> Doc:
    doc = Doc().h1("Admission report")
    rules = admission.results
    if admission.passed:
        doc.p(
            f"**Result: ADMITTED.** All {len(rules)} entry rules passed. "
            f"The dataset was written to `{dataset_path}`."
        )
    else:
        failed = len(admission.failed)
        text = (
            f"**Result: REJECTED.** {failed} of {len(rules)} entry rules failed. "
            "No dataset was written from this build."
        )
        if previous_file_kept:
            text += (
                f" The file at `{dataset_path}` was left as it was; it does not hold this build."
            )
        doc.p(text)
    doc.p(
        "The entry contract is the list of rules below (`docs/data-contract.md`). A build that "
        "breaks one does not enter, and no option lets it through: repair the view that maps the "
        "database onto the contract, or the settings, and run `base` again. Every number here is "
        "a count; no key, id or value from a row is printed."
    )
    doc.h2("Rules")
    doc.table(
        [
            [r.id, r.rule.scope, r.checked, r.violations, _shown(r.status)] for r in rules
        ],
        ["rule", "scope", "checked", "violations", "result"],
    )
    problems = [r for r in rules if r.status != PASS]
    if problems:
        doc.h2("What to fix")
        for r in problems:
            doc.h3(f"`{r.id}`: {r.status}")
            doc.p(r.rule.text)
            if r.detail:
                doc.bullets(r.detail)
            if r.status == FAIL:
                doc.p(f"To fix it: {r.rule.fix}")
    if admission.ledger:
        _ledger(doc, admission.ledger)
    if admission.queries:
        doc.h2("Queries run")
        doc.p(
            "Each statement is identified by a SHA-256 of its text with the length of every `IN` "
            "list ignored. The same fingerprints are stored in the dataset, so a later build can "
            "show whether the extraction changed."
        )
        doc.table(
            [
                [q["product_line"], q["name"], q["runs"], q["rows"], q["sha256"][:12]]
                for q in admission.queries
            ],
            ["product line", "query", "runs", "rows", "fingerprint"],
        )
    return doc


def _ledger(doc: Doc, ledger: Sequence[LedgerLine]) -> None:
    doc.h2("Filter ledger")
    doc.p(
        "Where the first-installment rows of the window went on the way to the slips, counted by "
        "the second route. A row is removed only by a filter the settings name; the last column "
        "counts slips whose agent has no row in `agents`, which an excluded-agent pattern cannot "
        "match."
    )
    doc.table(
        [
            [
                x.product_line,
                fmt_int(x.window),
                fmt_int(x.window - x.status),
                fmt_int(x.status - x.agents),
                fmt_int(x.agents - x.slips),
                fmt_int(x.slips),
                fmt_money(x.slips_value),
                fmt_int(x.unknown_agent),
            ]
            for x in ledger
        ],
        [
            "product line",
            "rows in window",
            "outside statuses",
            "excluded agents",
            "older writes",
            "slips",
            "slips value",
            "unknown agent",
        ],
        align_right=[
            "rows in window",
            "outside statuses",
            "excluded agents",
            "older writes",
            "slips",
            "slips value",
            "unknown agent",
        ],
    )
