"""Stage 01 - schema discovery: does the database look like the data contract?

Run this first against a new database. It never reads data rows except the small status
catalog. For every contract table it reports whether the table exists, which required columns
are missing, which optional ones are, and which extra columns the table carries. It compares the
declared type of every required column with the family the contract names (a date column must not
be declared numeric, an amount must not be declared as text). It also checks that every status
label named in the settings exists in the catalog, because the extraction resolves statuses by
label, never by number.

The ``base`` stage runs the same check before it extracts anything and refuses to build when it
fails, so a database that this stage rejects never gets as far as a dataset.

Two readiness levels, because the stages need different tables:

* ``slips``: enough for the period and card stages (agreements, statuses, agents),
* ``base``: everything the full base build reads.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field

from delinquency_eda.config import Settings
from delinquency_eda.context import Context
from delinquency_eda.contract import COLUMN_KINDS, TABLES, TableSpec, accepts
from delinquency_eda.db.base import ColumnInfo, Database
from delinquency_eda.queries import build_status_query
from delinquency_eda.report import Doc


@dataclass(frozen=True)
class TypeMismatch:
    """A required column whose declared type cannot hold what the contract names ``kind``."""

    column: str
    found: str
    kind: str


@dataclass
class TableCheck:
    spec: TableSpec
    columns: list[ColumnInfo]
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    wrong_types: list[TypeMismatch] = field(default_factory=list)

    @property
    def exists(self) -> bool:
        return bool(self.columns)

    @property
    def structure_ok(self) -> bool:
        """The table is there with every required column, whatever their types."""
        return self.exists and not self.missing_required

    @property
    def ok(self) -> bool:
        return self.structure_ok and not self.wrong_types


@dataclass
class LineSchema:
    product_line: str
    tables: list[TableCheck]
    missing_labels: list[str] = field(default_factory=list)
    labels_checked: bool = False
    labels_total: int = 0  # how many configured labels were looked up in the catalog

    def ready(self, scope: str) -> bool:
        """``scope`` is ``slips`` (period, card) or ``base`` (everything)."""
        needed = [t for t in self.tables if scope == "base" or t.spec.needed_by == "slips"]
        return all(t.ok for t in needed) and not self.missing_labels


def check_table(spec: TableSpec, columns: Sequence[ColumnInfo]) -> TableCheck:
    """Compare found columns with the spec. Names match case-insensitively."""
    present = {c.name.lower() for c in columns}
    known = {c.lower() for c in (*spec.required, *spec.optional)}
    kinds = COLUMN_KINDS.get(spec.name, {})
    return TableCheck(
        spec=spec,
        columns=list(columns),
        missing_required=[c for c in spec.required if c.lower() not in present] if columns else [],
        missing_optional=[c for c in spec.optional if c.lower() not in present] if columns else [],
        extra=[c.name for c in columns if c.name.lower() not in known],
        wrong_types=[
            TypeMismatch(c.name, c.type, kinds[c.name.lower()])
            for c in columns
            if c.name.lower() in kinds and not accepts(kinds[c.name.lower()], c.type)
        ],
    )


def configured_labels(settings: Settings) -> list[str]:
    """Every status label the extraction will look up."""
    labels = [*settings.generated_status_labels, *settings.broken_status_labels]
    labels.append(settings.manual_break_label)
    return list(dict.fromkeys(labels))


def check_database(db: Database, product_line: str, settings: Settings) -> LineSchema:
    tables = [check_table(spec, db.table_columns(spec.name)) for spec in TABLES]
    result = LineSchema(product_line, tables)
    catalog = next((t for t in tables if t.spec.name == "agreement_statuses"), None)
    if catalog is not None and catalog.ok:
        query = build_status_query(db.dialect, db.schema)
        found = {str(r["label"]).strip() for r in db.query(query.sql, query.params)}
        wanted = configured_labels(settings)
        result.labels_checked = True
        result.labels_total = len(wanted)
        result.missing_labels = [x for x in wanted if x not in found]
    return result


def render(lines: Sequence[LineSchema]) -> Doc:
    doc = Doc().h1("Schema check")
    doc.p(
        "Compared with the contract in docs/data-contract.md. Names match case-insensitively. "
        "Required columns are read by the pipeline; optional ones are checked but not read."
    )
    summary = [
        [s.product_line, "yes" if s.ready("slips") else "no", "yes" if s.ready("base") else "no"]
        for s in lines
    ]
    doc.h2("Readiness")
    doc.table(summary, ["product line", "ready for period/card", "ready for base"])
    for line in lines:
        doc.h2(f"Product line: {line.product_line}")
        rows = [
            [
                t.spec.name,
                "yes" if t.exists else "MISSING",
                ", ".join(t.missing_required) or "-",
                ", ".join(t.missing_optional) or "-",
                len(t.extra),
                ", ".join(m.column for m in t.wrong_types) or "-",
            ]
            for t in line.tables
        ]
        doc.table(
            rows,
            [
                "table",
                "present",
                "missing required",
                "missing optional",
                "extra columns",
                "wrong type",
            ],
        )
        _status_block(doc, line)
        _types_block(doc, line)
    return doc


def _status_block(doc: Doc, line: LineSchema) -> None:
    if not line.labels_checked:
        doc.p("Status labels were not checked: the status catalog is missing or incomplete.")
    elif line.missing_labels:
        doc.p(
            "Status labels named in the settings but absent from the catalog: "
            + ", ".join(f"`{x}`" for x in line.missing_labels)
            + "."
        )
    else:
        doc.p("Every configured status label exists in the catalog.")


def _types_block(doc: Doc, line: LineSchema) -> None:
    rows = []
    for table in line.tables:
        wanted = {c.lower() for c in table.spec.required}
        kinds = COLUMN_KINDS.get(table.spec.name, {})
        wrong = {m.column for m in table.wrong_types}
        rows += [
            [
                table.spec.name,
                c.name,
                c.type,
                "yes" if c.nullable else "no",
                kinds.get(c.name.lower(), "-"),
                "WRONG" if c.name in wrong else "ok",
            ]
            for c in table.columns
            if c.name.lower() in wanted
        ]
    if rows:
        doc.h3("Types of the required columns found")
        doc.table(rows, ["table", "column", "type", "nullable", "contract", "fits"])
        doc.p(
            "The contract column names what the column must hold: `integer`, `number`, `date` "
            "(a date type, or text holding ISO dates), `text`, `id` (text or integer) or `flag` "
            "(0 or 1). An engine that declares no type, as SQLite may, is not judged here: its "
            "values are checked when they are read."
        )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    del parser  # this stage has no options of its own


def run(ctx: Context, args: argparse.Namespace) -> int:
    del args
    with ctx.databases() as databases:
        lines = [check_database(db, name, ctx.settings) for name, db in databases.items()]
    for line in lines:
        ctx.say(
            f"{line.product_line}: ready for period/card = {line.ready('slips')}, "
            f"ready for base = {line.ready('base')}"
        )
        for table in line.tables:
            if not table.structure_ok:
                what = (
                    "missing" if not table.exists else f"missing columns {table.missing_required}"
                )
                ctx.say(f"  {table.spec.name}: {what}")
            for mismatch in table.wrong_types:
                ctx.say(
                    f"  {table.spec.name}.{mismatch.column}: declared {mismatch.found!r}, "
                    f"the contract needs {mismatch.kind}"
                )
        if line.missing_labels:
            ctx.say(f"  status labels not in the catalog: {line.missing_labels}")
    ctx.write_report("schema.md", render(lines))
    return 0 if all(line.ready("base") for line in lines) else 1
