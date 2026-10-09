"""Stage 01 - schema discovery: does the database look like the data contract?

Run this first against a new database. It never reads data rows except the small status
catalog. For every contract table it reports whether the table exists, which required columns
are missing, which optional ones are, and which extra columns the table carries. It also checks
that every status label named in the settings exists in the catalog, because the extraction
resolves statuses by label, never by number.

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
from delinquency_eda.contract import TABLES, TableSpec
from delinquency_eda.db.base import ColumnInfo, Database
from delinquency_eda.queries import build_status_query
from delinquency_eda.report import Doc


@dataclass
class TableCheck:
    spec: TableSpec
    columns: list[ColumnInfo]
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)

    @property
    def exists(self) -> bool:
        return bool(self.columns)

    @property
    def ok(self) -> bool:
        return self.exists and not self.missing_required


@dataclass
class LineSchema:
    product_line: str
    tables: list[TableCheck]
    missing_labels: list[str] = field(default_factory=list)
    labels_checked: bool = False

    def ready(self, scope: str) -> bool:
        """``scope`` is ``slips`` (period, card) or ``base`` (everything)."""
        needed = [t for t in self.tables if scope == "base" or t.spec.needed_by == "slips"]
        return all(t.ok for t in needed) and not self.missing_labels


def check_table(spec: TableSpec, columns: Sequence[ColumnInfo]) -> TableCheck:
    """Compare found columns with the spec. Names match case-insensitively."""
    present = {c.name.lower() for c in columns}
    known = {c.lower() for c in (*spec.required, *spec.optional)}
    return TableCheck(
        spec=spec,
        columns=list(columns),
        missing_required=[c for c in spec.required if c.lower() not in present] if columns else [],
        missing_optional=[c for c in spec.optional if c.lower() not in present] if columns else [],
        extra=[c.name for c in columns if c.name.lower() not in known],
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
        result.labels_checked = True
        result.missing_labels = [x for x in configured_labels(settings) if x not in found]
    return result


def render(lines: Sequence[LineSchema]) -> Doc:
    doc = Doc().h1("Schema check")
    doc.p(
        "Compared with the contract in docs/data-contract.md. Names match case-insensitively. "
        "Required columns are read by the pipeline; optional ones belong to the shared contract "
        "but are not read here."
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
            ]
            for t in line.tables
        ]
        doc.table(
            rows, ["table", "present", "missing required", "missing optional", "extra columns"]
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
        rows += [
            [table.spec.name, c.name, c.type, "yes" if c.nullable else "no"]
            for c in table.columns
            if c.name.lower() in wanted
        ]
    if rows:
        doc.h3("Types of the required columns found")
        doc.table(rows, ["table", "column", "type", "nullable"])


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
            if not table.ok:
                what = (
                    "missing" if not table.exists else f"missing columns {table.missing_required}"
                )
                ctx.say(f"  {table.spec.name}: {what}")
        if line.missing_labels:
            ctx.say(f"  status labels not in the catalog: {line.missing_labels}")
    ctx.write_report("schema.md", render(lines))
    return 0 if all(line.ready("base") for line in lines) else 1
