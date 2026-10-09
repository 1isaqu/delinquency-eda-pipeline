"""The data contract: tables and columns the extraction expects. See docs/data-contract.md.

Names are generic. A real database is mapped onto them with views (the documented route) or by
renaming. ``required`` columns are read by this pipeline; ``optional`` columns are checked by the
schema stage but not read, so a missing one is only informational.
"""

from __future__ import annotations

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
        optional=("debt_owner_id", "agent_id"),
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
