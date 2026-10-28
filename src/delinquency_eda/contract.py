"""The data contract: tables and columns the extraction expects. See docs/data-contract.md.

Names are generic. A real database is mapped onto them with views (the documented route) or by
renaming. ``required`` columns are read by this pipeline; ``optional`` columns are checked by the
schema stage but not read, so a missing one is only informational.

The second half of the module is the *entry contract*: the rules a build must satisfy before its
dataset is written. The ``base`` stage enforces them (``admission.py``). Data that breaks a rule
does not enter; there is no switch to let it through. The catalogue below is the single list the
admission code, the report, the docs and the tests read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TableSpec:
    name: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    needed_by: str = "base"
    meaning: str = ""


# ``slips``  = stages that fetch first-installment rows only (period, card).
# ``base``   = the full base build (everything).
TABLES: tuple[TableSpec, ...] = (
    TableSpec(
        "agreements",
        required=(
            "row_id",
            "agreement_id",
            "portfolio_id",
            "debtor_id",
            "installment_no",
            "amount",
            "plan_installments",
            "paid_amount",
            "issued_at",
            "due_at",
            "paid_at",
            "status_id",
            "agent_id",
        ),
        needed_by="slips",
        meaning="one row per installment of an agreement",
    ),
    TableSpec(
        "agreement_statuses",
        required=("status_id", "label"),
        needed_by="slips",
        meaning="status catalog",
    ),
    TableSpec(
        "agents",
        required=("agent_id", "display_name", "login"),
        optional=(),
        needed_by="slips",
        meaning="users who own agreements",
    ),
    TableSpec(
        "agreement_debts",
        required=("agreement_id", "portfolio_id", "debt_id"),
        meaning="links agreements to debts",
    ),
    TableSpec(
        "debts",
        required=("debt_id", "balance"),
        optional=("portfolio_id", "contract_no", "due_at", "days_overdue"),
        meaning="debt (contract) detail",
    ),
    TableSpec(
        "debt_attributes",
        required=("debt_id", "portfolio_label"),
        meaning="extra debt attributes; holds the portfolio label",
    ),
    TableSpec(
        "debtors",
        required=("debtor_id", "person_key"),
        meaning="debtor rows with an opaque person key (same person, same key)",
    ),
    TableSpec(
        "contact_events",
        required=("event_id", "debtor_id", "result_id", "occurred_at"),
        optional=("agent_id",),
        meaning="one row per outreach attempt",
    ),
    TableSpec(
        "contact_results",
        required=("result_id", "label", "answered", "right_party"),
        meaning="result catalog; right-party contact = answered AND right_party",
    ),
)

TABLE_NAMES = tuple(t.name for t in TABLES)


def spec_for(name: str) -> TableSpec:
    for spec in TABLES:
        if spec.name == name:
            return spec
    raise KeyError(name)


# --------------------------------------------------------------------------------------------
# Column kinds: what each required column holds
# --------------------------------------------------------------------------------------------

INTEGER = "integer"
NUMBER = "number"
TEXT = "text"
ID = "id"  # text or integer: an opaque identifier
DATE = "date"
FLAG = "flag"

# The families a declared type can belong to, and the families each kind accepts. A date may be
# declared as text because SQLite stores ISO text; the values are read and checked either way.
NUMBER_FAMILY = "number"
TEXT_FAMILY = "text"
DATE_FAMILY = "date"
ACCEPTED_FAMILIES: dict[str, frozenset[str]] = {
    INTEGER: frozenset({NUMBER_FAMILY}),
    NUMBER: frozenset({NUMBER_FAMILY}),
    TEXT: frozenset({TEXT_FAMILY}),
    ID: frozenset({NUMBER_FAMILY, TEXT_FAMILY}),
    DATE: frozenset({DATE_FAMILY, TEXT_FAMILY}),
    FLAG: frozenset({NUMBER_FAMILY}),
}

COLUMN_KINDS: dict[str, dict[str, str]] = {
    "agreements": {
        "row_id": INTEGER,
        "agreement_id": INTEGER,
        "portfolio_id": INTEGER,
        "debtor_id": INTEGER,
        "installment_no": INTEGER,
        "amount": NUMBER,
        "plan_installments": INTEGER,
        "paid_amount": NUMBER,
        "issued_at": DATE,
        "due_at": DATE,
        "paid_at": DATE,
        "status_id": INTEGER,
        "agent_id": ID,
    },
    "agreement_statuses": {"status_id": INTEGER, "label": TEXT},
    "agents": {"agent_id": ID, "display_name": TEXT, "login": TEXT},
    "agreement_debts": {"agreement_id": INTEGER, "portfolio_id": INTEGER, "debt_id": INTEGER},
    "debts": {"debt_id": INTEGER, "balance": NUMBER},
    "debt_attributes": {"debt_id": INTEGER, "portfolio_label": TEXT},
    "debtors": {"debtor_id": INTEGER, "person_key": TEXT},
    "contact_events": {
        "event_id": INTEGER,
        "debtor_id": INTEGER,
        "result_id": INTEGER,
        "occurred_at": DATE,
    },
    "contact_results": {
        "result_id": INTEGER,
        "label": TEXT,
        "answered": FLAG,
        "right_party": FLAG,
    },
}

_NUMBER_TYPE = re.compile(
    r"^(tinyint|smallint|mediumint|int\d*|integer|bigint|decimal|numeric|dec|real|float\d*"
    r"|double|money|smallmoney|number|bit|bool|boolean)\b",
    re.IGNORECASE,
)
_TEXT_TYPE = re.compile(
    r"^(n?(var)?char|character|n?text|clob|string|uniqueidentifier|sysname)\b", re.IGNORECASE
)
_DATE_TYPE = re.compile(
    r"^(date|datetime\d*|smalldatetime|datetimeoffset|timestamp)\b", re.IGNORECASE
)


def family_of(declared_type: str) -> str | None:
    """``number``, ``text`` or ``date`` for a declared type such as ``decimal(12,2)``.

    ``None`` means the type is not one of the three (binary, xml, ...). An empty string means the
    engine declares no type at all, which SQLite allows; ``accepts`` lets that through, because
    the values are checked when they are read.
    """
    text = declared_type.strip()
    for family, pattern in (
        (NUMBER_FAMILY, _NUMBER_TYPE),
        (TEXT_FAMILY, _TEXT_TYPE),
        (DATE_FAMILY, _DATE_TYPE),
    ):
        if pattern.match(text):
            return family
    return None


def accepts(kind: str, declared_type: str) -> bool:
    """Whether a column declared as ``declared_type`` can hold what the contract names ``kind``."""
    if not declared_type.strip():
        return True
    return family_of(declared_type) in ACCEPTED_FAMILIES[kind]


# --------------------------------------------------------------------------------------------
# The columns of what the extraction returns
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedTable:
    """The columns of one table of the dataset, by what the entry rules ask of them.

    ``required`` columns are never null (the contract's Null column says "no"). ``dates``,
    ``numbers`` and ``counts`` must read as ISO dates, numbers and whole numbers when they are
    present. ``flags`` must be 0 or 1.
    """

    name: str
    required: tuple[str, ...]
    dates: tuple[str, ...] = ()
    numbers: tuple[str, ...] = ()
    counts: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()


EXTRACTED: tuple[ExtractedTable, ...] = (
    ExtractedTable(
        "slips",
        required=(
            "record_id",
            "agreement_id",
            "portfolio_id",
            "debtor_id",
            "status_id",
            "agent_id",
            "first_installment_value",
            "agreement_date",
            "due_date",
        ),
        dates=("agreement_date", "due_date", "paid_date"),
        numbers=("first_installment_value", "total_value", "paid_amount", "original_debt"),
        counts=("plan_installments", "total_installments"),
    ),
    ExtractedTable(
        "contacts",
        required=("debtor_id", "contact_date", "result_id", "answered", "right_party"),
        dates=("contact_date",),
        flags=("answered", "right_party"),
    ),
    ExtractedTable(
        "breaks",
        required=("debtor_id", "agreement_id", "portfolio_id", "status_id", "agreement_date"),
        dates=("agreement_date",),
    ),
    ExtractedTable("debtors", required=("debtor_id",)),
)


# --------------------------------------------------------------------------------------------
# The entry rules
# --------------------------------------------------------------------------------------------

CONTRACT_VERSION = 1

SOURCE = "source"  # judged against the live database
ROWS = "rows"  # judged on every row as the driver returned it
DATASET = "dataset"  # judged on the assembled dataset


@dataclass(frozen=True)
class Rule:
    id: str
    scope: str
    text: str
    fix: str


RULES: tuple[Rule, ...] = (
    Rule(
        "source.tables",
        SOURCE,
        "Every table the base build reads exists and has all its required columns.",
        "Add the table or column to the view that maps your database onto the contract.",
    ),
    Rule(
        "source.labels",
        SOURCE,
        "Every status label named in the settings exists in agreement_statuses.",
        "Correct the label in the settings file, or add the status to the catalog.",
    ),
    Rule(
        "source.types",
        SOURCE,
        "The declared type of every required column is in the family the contract names.",
        "Cast the column in the view; docs/data-contract.md lists the family of each column.",
    ),
    Rule(
        "row.required",
        ROWS,
        "A column the contract marks not null is never null in an extracted row.",
        "Fill the value in the view, or leave the row out of it. The pipeline never imputes.",
    ),
    Rule(
        "row.readable",
        ROWS,
        "Dates read as ISO dates, amounts as numbers, counts as whole numbers, flags as 0 or 1.",
        "Cast the column in the view. A flag must be 0 or 1, never text such as 'Y' or 'N'.",
    ),
    Rule(
        "row.payment",
        ROWS,
        "paid_amount is NULL or positive, and paid_amount and paid_at are both present or both "
        "NULL.",
        "Return NULL in both columns for a slip that was not paid, never 0 and never just one.",
    ),
    Rule(
        "row.due_date",
        SOURCE,
        "Every first-installment row of agreements has a due date that reads as a date.",
        "Fix the date in the view. A row with no readable due date drops out of every window "
        "without a trace, so the pipeline cannot say what it lost.",
    ),
    Rule(
        "key.grain",
        DATASET,
        "There is one slip per product line, agreement and portfolio; row ids and debtor ids are "
        "unique within a product line.",
        "Remove the duplicate keys from agents, debtors, agreement_statuses or agreements. A "
        "duplicate key in a reference table multiplies the slips that join to it.",
    ),
    Rule(
        "base.domain",
        DATASET,
        "Every slip has a known bucket, a generated status label, a configured product line and a "
        "due date inside the requested window.",
        "This points at the extraction, not at the data. Report it with the dataset's meta block.",
    ),
    Rule(
        "ref.debtor",
        SOURCE,
        "The debtor_id of every extracted slip has a row in debtors.",
        "Add the missing debtor rows to the debtors view.",
    ),
    Rule(
        "ref.contact_result",
        SOURCE,
        "The result_id of every contact event in the window has a row in contact_results.",
        "Add the missing results to the catalog. The extraction would otherwise drop those events.",
    ),
    Rule(
        "recon.slips",
        SOURCE,
        "The number of slips per product line equals the count of a second, independent query.",
        "Compare the two counts in the report; duplicate keys in a reference table are the "
        "usual cause.",
    ),
    Rule(
        "recon.outcome",
        SOURCE,
        "The slips in each bucket, per product line, equal the count of the outcome rule "
        "re-derived in SQL.",
        "Look at the dates and amounts of the slips in the bucket that differs.",
    ),
    Rule(
        "recon.value",
        SOURCE,
        "The first-installment value and the whole-agreement value per product line equal the "
        "sums of the independent queries.",
        "Compare the two sums in the report; duplicate keys are the usual cause.",
    ),
    Rule(
        "recon.history",
        SOURCE,
        "The contact events and earlier broken agreements pulled in per product line equal the "
        "counts of the independent queries.",
        "Duplicate keys in contact_results or agreement_statuses multiply these rows.",
    ),
)

RULE_IDS = tuple(r.id for r in RULES)


def rule_for(rule_id: str) -> Rule:
    for rule in RULES:
        if rule.id == rule_id:
            return rule
    raise KeyError(rule_id)
