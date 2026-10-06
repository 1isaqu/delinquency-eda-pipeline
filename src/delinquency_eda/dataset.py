"""The dataset bundle: what stage 02 extracts and every offline stage reads.

One JSON file holding the meta (period, reference date, rules used) and four tables:

* ``base``     one row per first-installment slip, with its derived outcome,
* ``debtors``  every debtor row that shares an opaque ``person_key`` with a base row,
* ``contacts`` outreach attempts for those debtors between the earliest agreement and the last due
               date,
* ``breaks``   earlier first-installment agreements of those debtors that ended broken.

Keys always include the product line: row and debtor ids repeat across databases.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from delinquency_eda import derive
from delinquency_eda.config import Settings
from delinquency_eda.dates import iso, parse_date_arg, to_date, to_float, to_id
from delinquency_eda.db.base import Database
from delinquency_eda.errors import DatasetError
from delinquency_eda.queries import (
    build_breaks_query,
    build_contacts_query,
    build_person_debtors_query,
    build_slip_query,
)
from delinquency_eda.sqlutil import chunked

SCHEMA_VERSION = 1

BASE_COLUMNS = [
    "product_line",
    "record_id",
    "agreement_id",
    "portfolio_id",
    "debtor_id",
    "person_key",
    "status_id",
    "status_label",
    "first_installment_value",
    "total_value",
    "total_installments",
    "plan_installments",
    "original_debt",
    "agent_id",
    "agent_name",
    "agent_login",
    "portfolio_label",
    "agreement_date",
    "due_date",
    "paid_amount",
    "paid_date",
    "days_to_due",
    "days_overdue",
    "bucket",
]
DEBTOR_COLUMNS = ["product_line", "debtor_id", "person_key"]
CONTACT_COLUMNS = [
    "product_line",
    "debtor_id",
    "contact_date",
    "result_id",
    "result_label",
    "answered",
    "right_party",
]
BREAK_COLUMNS = [
    "product_line",
    "debtor_id",
    "agreement_id",
    "portfolio_id",
    "status_id",
    "status_label",
    "agreement_date",
]
NUMERIC_BASE = [
    "first_installment_value",
    "total_value",
    "total_installments",
    "plan_installments",
    "original_debt",
    "paid_amount",
    "days_to_due",
    "days_overdue",
]


@dataclass
class DatasetMeta:
    period_from: str
    period_to: str
    as_of: str
    grace_days: int
    product_lines: list[str]
    generated_status_labels: list[str]
    broken_status_labels: list[str]
    manual_break_label: str
    excluded_agent_patterns: list[str]
    schema_version: int = SCHEMA_VERSION

    @property
    def as_of_date(self) -> date:
        return parse_date_arg(self.as_of)


@dataclass
class Dataset:
    meta: DatasetMeta
    base: list[dict[str, Any]] = field(default_factory=list)
    debtors: list[dict[str, Any]] = field(default_factory=list)
    contacts: list[dict[str, Any]] = field(default_factory=list)
    breaks: list[dict[str, Any]] = field(default_factory=list)

    def base_frame(self) -> pd.DataFrame:
        return _frame(self.base, BASE_COLUMNS, NUMERIC_BASE)

    def debtors_frame(self) -> pd.DataFrame:
        return _frame(self.debtors, DEBTOR_COLUMNS)

    def contacts_frame(self) -> pd.DataFrame:
        return _frame(self.contacts, CONTACT_COLUMNS)

    def breaks_frame(self) -> pd.DataFrame:
        return _frame(self.breaks, BREAK_COLUMNS)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": asdict(self.meta),
            "base": self.base,
            "debtors": self.debtors,
            "contacts": self.contacts,
            "breaks": self.breaks,
        }
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")
        return target

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        target = Path(path)
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except OSError as exc:
            raise DatasetError(f"cannot read dataset {target}: {exc.strerror or exc}") from None
        except json.JSONDecodeError as exc:
            raise DatasetError(f"dataset {target} is not valid JSON: {exc}") from None
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Dataset:
        try:
            meta_raw = dict(raw["meta"])
            tables = {name: list(raw[name]) for name in ("base", "debtors", "contacts", "breaks")}
        except (KeyError, TypeError) as exc:
            raise DatasetError(f"dataset is missing a section: {exc}") from None
        if meta_raw.get("schema_version") != SCHEMA_VERSION:
            raise DatasetError(
                f"dataset schema version {meta_raw.get('schema_version')!r} is not "
                f"{SCHEMA_VERSION}; rebuild it with the base command"
            )
        try:
            meta = DatasetMeta(**meta_raw)
        except TypeError as exc:
            raise DatasetError(f"dataset meta is malformed: {exc}") from None
        return cls(meta=meta, **tables)


def _frame(rows: Sequence[dict[str, Any]], columns: list[str], numeric: Sequence[str] = ()):
    frame = pd.DataFrame(list(rows), columns=columns)
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


# --------------------------------------------------------------------------------------------
# Row normalisation
# --------------------------------------------------------------------------------------------


def normalize_slip(
    row: Mapping[str, Any], product_line: str, *, as_of: date, grace_days: int
) -> dict[str, Any]:
    """Driver values -> JSON-safe values, plus the derived outcome columns."""
    agreement = to_date(row.get("agreement_date"))
    due = to_date(row.get("due_date"))
    if due is None:
        raise DatasetError("a slip came back without a due date")
    paid_on = to_date(row.get("paid_date"))
    paid_amount = to_float(row.get("paid_amount"))
    plan = row.get("plan_installments")
    total_installments = row.get("total_installments")
    return {
        "product_line": product_line,
        "record_id": to_id(row["record_id"]),
        "agreement_id": to_id(row["agreement_id"]),
        "portfolio_id": to_id(row["portfolio_id"]),
        "debtor_id": to_id(row["debtor_id"]),
        "person_key": _text(row.get("person_key")),
        "status_id": to_id(row["status_id"]),
        "status_label": _text(row.get("status_label")),
        "first_installment_value": to_float(row.get("first_installment_value")),
        "total_value": to_float(row.get("total_value")),
        "total_installments": None if total_installments is None else int(total_installments),
        "plan_installments": None if plan is None else int(plan),
        "original_debt": to_float(row.get("original_debt")),
        "agent_id": to_id(row.get("agent_id")),
        "agent_name": _text(row.get("agent_name")),
        "agent_login": _text(row.get("agent_login")),
        "portfolio_label": _text(row.get("portfolio_label")),
        "agreement_date": iso(agreement),
        "due_date": iso(due),
        "paid_amount": paid_amount,
        "paid_date": iso(paid_on),
        "days_to_due": derive.days_to_due(agreement, due) if agreement else None,
        "days_overdue": derive.days_overdue(due, as_of),
        "bucket": derive.classify(due, paid_on, paid_amount, as_of=as_of, grace_days=grace_days),
    }


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text.strip() else None


# --------------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------------


def fetch_slips(
    db: Database,
    product_line: str,
    settings: Settings,
    date_from: date,
    date_to: date,
    *,
    as_of: date,
    detail: bool,
) -> list[dict[str, Any]]:
    """Slips due in ``[date_from, date_to]`` for one product line, normalised and classified."""
    query = build_slip_query(
        db.dialect, settings, date_from, date_to, detail=detail, schema=db.schema
    )
    rows = db.query(query.sql, query.params)
    return [
        normalize_slip(r, product_line, as_of=as_of, grace_days=settings.grace_days) for r in rows
    ]


def _ordered(ids: set[Any]) -> list[Any]:
    return sorted(ids, key=lambda v: (isinstance(v, str), v))


def build_dataset(
    databases: Mapping[str, Database],
    settings: Settings,
    *,
    date_from: date,
    date_to: date,
    as_of: date,
) -> Dataset:
    """Run the extraction against every product line and assemble the dataset bundle."""
    if date_to < date_from:
        raise DatasetError("the period end is before its start")
    base: list[dict[str, Any]] = []
    for line, db in databases.items():
        base += fetch_slips(db, line, settings, date_from, date_to, as_of=as_of, detail=True)
    meta = DatasetMeta(
        period_from=date_from.isoformat(),
        period_to=date_to.isoformat(),
        as_of=as_of.isoformat(),
        grace_days=settings.grace_days,
        product_lines=list(databases),
        generated_status_labels=list(settings.generated_status_labels),
        broken_status_labels=list(settings.broken_status_labels),
        manual_break_label=settings.manual_break_label,
        excluded_agent_patterns=list(settings.excluded_agent_patterns),
    )
    dataset = Dataset(meta=meta, base=base)
    if not base:
        return dataset

    keys = sorted({r["person_key"] for r in base if r["person_key"]})
    own_ids: dict[str, set[Any]] = {
        line: {
            r["debtor_id"]
            for r in base
            if r["product_line"] == line and r["debtor_id"] is not None
        }
        for line in databases
    }
    for line, db in databases.items():
        debtor_ids = set(own_ids[line])
        for part in chunked(keys, settings.id_chunk_size):
            q = build_person_debtors_query(db.dialect, part, db.schema)
            for r in db.query(q.sql, q.params):
                dataset.debtors.append(
                    {
                        "product_line": line,
                        "debtor_id": to_id(r["debtor_id"]),
                        "person_key": _text(r["person_key"]),
                    }
                )
                debtor_ids.add(to_id(r["debtor_id"]))
        own_ids[line] = debtor_ids

    window_from = min(to_date(r["agreement_date"]) or date_from for r in base)
    window_to = max(to_date(r["due_date"]) for r in base)  # type: ignore[type-var]
    for line, db in databases.items():
        ordered = _ordered(own_ids[line])
        for part in chunked(ordered, settings.id_chunk_size):
            q = build_contacts_query(db.dialect, part, window_from, window_to, db.schema)
            for r in db.query(q.sql, q.params):
                dataset.contacts.append(_contact(line, r))
            b = build_breaks_query(db.dialect, settings, part, db.schema)
            for r in db.query(b.sql, b.params):
                dataset.breaks.append(_break(line, r))
    return dataset


def _contact(line: str, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "product_line": line,
        "debtor_id": to_id(row["debtor_id"]),
        "contact_date": iso(row["contact_date"]),
        "result_id": to_id(row["result_id"]),
        "result_label": _text(row.get("result_label")),
        "answered": bool(row["answered"]),
        "right_party": bool(row["right_party"]),
    }


def _break(line: str, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "product_line": line,
        "debtor_id": to_id(row["debtor_id"]),
        "agreement_id": to_id(row["agreement_id"]),
        "portfolio_id": to_id(row["portfolio_id"]),
        "status_id": to_id(row["status_id"]),
        "status_label": _text(row.get("status_label")),
        "agreement_date": iso(row["agreement_date"]),
    }
