"""The dataset bundle: what stage 02 extracts and every offline stage reads.

One JSON file holding the meta (period, reference date, rules used) and four tables:

* ``base``     one row per first-installment slip, with its derived outcome,
* ``debtors``  every debtor row that shares an opaque ``person_key`` with a base row,
* ``contacts`` outreach attempts for those debtors between the earliest agreement and the last due
               date,
* ``breaks``   earlier first-installment agreements of those debtors that ended broken.

Keys always include the product line: row and debtor ids repeat across databases.

The file also carries the *admission record* (which entry rules the build was judged against and
how each came out) and a *seal*: a SHA-256 of everything else in the file and the row count of
each table. ``save`` refuses a dataset that was not admitted, and ``load`` refuses a file whose
content no longer matches its seal, so a stage never reads rows that did not pass the contract or
that were cut or edited after the ``base`` stage wrote them. The seal detects accidents; anyone
who can edit the file can also recompute it, so it is not a defence against tampering.
"""

from __future__ import annotations

import hashlib
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
from delinquency_eda.intake import Intake
from delinquency_eda.queries import (
    build_breaks_query,
    build_contacts_query,
    build_person_debtors_query,
    build_slip_query,
)
from delinquency_eda.sqlutil import chunked

SCHEMA_VERSION = 2
TABLE_NAMES = ("base", "debtors", "contacts", "breaks")

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
    admission: dict[str, Any] | None = None

    def base_frame(self) -> pd.DataFrame:
        return _frame(self.base, BASE_COLUMNS, NUMERIC_BASE)

    def debtors_frame(self) -> pd.DataFrame:
        return _frame(self.debtors, DEBTOR_COLUMNS)

    def contacts_frame(self) -> pd.DataFrame:
        return _frame(self.contacts, CONTACT_COLUMNS)

    def breaks_frame(self) -> pd.DataFrame:
        return _frame(self.breaks, BREAK_COLUMNS)

    def save(self, path: str | Path) -> Path:
        """Write the dataset with its admission record and seal. Only an admitted one is written."""
        if not is_admitted(self.admission):
            raise DatasetError(
                "refusing to write a dataset that did not pass the entry contract; "
                "only the base command can admit one"
            )
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": asdict(self.meta),
            "base": self.base,
            "debtors": self.debtors,
            "contacts": self.contacts,
            "breaks": self.breaks,
            "admission": self.admission,
        }
        sealed = {**payload, "seal": make_seal(payload)}
        target.write_text(json.dumps(sealed, ensure_ascii=False), encoding="utf-8", newline="\n")
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
            tables = {name: list(raw[name]) for name in TABLE_NAMES}
        except (KeyError, TypeError) as exc:
            raise DatasetError(f"dataset is missing a section: {exc}") from None
        if meta_raw.get("schema_version") != SCHEMA_VERSION:
            raise DatasetError(
                f"dataset schema version {meta_raw.get('schema_version')!r} is not "
                f"{SCHEMA_VERSION}; rebuild it with the base command"
            )
        verify_seal({k: v for k, v in raw.items() if k != "seal"}, raw.get("seal"))
        try:
            meta = DatasetMeta(**meta_raw)
        except TypeError as exc:
            raise DatasetError(f"dataset meta is malformed: {exc}") from None
        return cls(meta=meta, admission=raw["admission"], **tables)


def is_admitted(admission: Mapping[str, Any] | None) -> bool:
    """Whether an admission record says the build passed every entry rule."""
    return isinstance(admission, Mapping) and admission.get("passed") is True


def content_digest(body: Mapping[str, Any]) -> str:
    """SHA-256 of a dataset body in a canonical form (sorted keys, no spaces)."""
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_seal(body: Mapping[str, Any]) -> dict[str, Any]:
    """The seal of a body: its digest and the row count of each table."""
    return {
        "algorithm": "sha256",
        "digest": content_digest(body),
        "rows": {name: len(body[name]) for name in TABLE_NAMES},
    }


def verify_seal(body: Mapping[str, Any], seal: Any) -> None:
    """Raise ``DatasetError`` unless ``body`` matches ``seal`` and carries a passed admission."""
    if not isinstance(seal, Mapping) or seal.get("algorithm") != "sha256":
        raise DatasetError(
            "dataset has no seal, so it was not written by the base command; rebuild it with "
            "the base command"
        )
    expected = seal.get("rows") if isinstance(seal.get("rows"), Mapping) else {}
    for name in TABLE_NAMES:
        if expected.get(name) != len(body[name]):
            raise DatasetError(
                f"dataset table {name!r} has {len(body[name])} rows and its seal says "
                f"{expected.get(name)}: the file was cut or edited after the base command wrote "
                "it; rebuild it with the base command"
            )
    if seal.get("digest") != content_digest(body):
        raise DatasetError(
            "dataset content does not match its seal: the file was edited after the base "
            "command wrote it; rebuild it with the base command"
        )
    if not is_admitted(body.get("admission")):
        raise DatasetError(
            "dataset did not pass the entry contract; rebuild it with the base command"
        )


def _frame(rows: Sequence[dict[str, Any]], columns: list[str], numeric: Sequence[str] = ()):
    frame = pd.DataFrame(list(rows), columns=columns)
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


# --------------------------------------------------------------------------------------------
# Row normalization
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
    intake: Intake | None = None,
) -> list[dict[str, Any]]:
    """Slips due in ``[date_from, date_to]`` for one product line, normalized and classified.

    With an ``intake`` the rows are judged against the entry rules first and only the clean ones
    are normalized; without one, a row that cannot be read stops the build with an error.
    """
    query = build_slip_query(
        db.dialect, settings, date_from, date_to, detail=detail, schema=db.schema
    )
    rows = _read(db, product_line, "slips" if detail else "slips_lean", query, intake)
    if intake is not None:
        rows = intake.accept("slips", rows)
    return [
        normalize_slip(r, product_line, as_of=as_of, grace_days=settings.grace_days) for r in rows
    ]


def _ordered(ids: set[Any]) -> list[Any]:
    return sorted(ids, key=lambda v: (isinstance(v, str), v))


def _read(db: Database, line: str, name: str, query: Any, intake: Intake | None) -> list[Any]:
    """Run ``query``; through the intake when there is one, so it is logged."""
    if intake is None:
        return db.query(query.sql, query.params)
    return intake.run(db, line, name, query)


def _accepted(intake: Intake | None, table: str, rows: list[Any]) -> list[Any]:
    return rows if intake is None else intake.accept(table, rows)


def history_window(base: Sequence[Mapping[str, Any]], date_from: date) -> tuple[date, date]:
    """Earliest agreement date and last due date of the slips: the span outreach is read for."""
    first = min(to_date(r["agreement_date"]) or date_from for r in base)
    last = max(to_date(r["due_date"]) for r in base)  # type: ignore[type-var]
    return first, last


def debtor_ids_by_line(dataset: Dataset) -> dict[str, list[Any]]:
    """Every debtor id the build read outreach and break history for, per product line."""
    ids: dict[str, set[Any]] = {line: set() for line in dataset.meta.product_lines}
    for row in (*dataset.base, *dataset.debtors):
        if row["debtor_id"] is not None:
            ids[row["product_line"]].add(row["debtor_id"])
    return {line: _ordered(found) for line, found in ids.items()}


def build_dataset(
    databases: Mapping[str, Database],
    settings: Settings,
    *,
    date_from: date,
    date_to: date,
    as_of: date,
    intake: Intake | None = None,
) -> Dataset:
    """Run the extraction against every product line and assemble the dataset bundle.

    Passing an ``intake`` judges every row against the entry rules and logs every query. The
    dataset that comes back still has to be admitted (``admission.admit``) before it can be saved.
    """
    if date_to < date_from:
        raise DatasetError("the period end is before its start")
    base: list[dict[str, Any]] = []
    for line, db in databases.items():
        base += fetch_slips(
            db, line, settings, date_from, date_to, as_of=as_of, detail=True, intake=intake
        )
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
            r["debtor_id"] for r in base if r["product_line"] == line and r["debtor_id"] is not None
        }
        for line in databases
    }
    for line, db in databases.items():
        debtor_ids = set(own_ids[line])
        for part in chunked(keys, settings.id_chunk_size):
            q = build_person_debtors_query(db.dialect, part, db.schema)
            for r in _accepted(intake, "debtors", _read(db, line, "debtors", q, intake)):
                dataset.debtors.append(
                    {
                        "product_line": line,
                        "debtor_id": to_id(r["debtor_id"]),
                        "person_key": _text(r["person_key"]),
                    }
                )
                debtor_ids.add(to_id(r["debtor_id"]))
        own_ids[line] = debtor_ids

    window_from, window_to = history_window(base, date_from)
    for line, db in databases.items():
        ordered = _ordered(own_ids[line])
        for part in chunked(ordered, settings.id_chunk_size):
            q = build_contacts_query(db.dialect, part, window_from, window_to, db.schema)
            for r in _accepted(intake, "contacts", _read(db, line, "contacts", q, intake)):
                dataset.contacts.append(_contact(line, r))
            b = build_breaks_query(db.dialect, settings, part, db.schema)
            for r in _accepted(intake, "breaks", _read(db, line, "breaks", b, intake)):
                dataset.breaks.append(_break(line, r))
    sort_tables(dataset)
    return dataset


def _id_key(value: Any) -> tuple[int, Any]:
    """A sort key for ids that may be numbers, text or missing, without mixed-type comparisons."""
    if value is None:
        return (2, 0)
    return (1, value) if isinstance(value, str) else (0, value)


def sort_tables(dataset: Dataset) -> None:
    """Put every table in a fixed order, so the file does not depend on the database's row order."""
    rank = {line: i for i, line in enumerate(dataset.meta.product_lines)}
    dataset.base.sort(
        key=lambda r: (
            rank[r["product_line"]],
            _id_key(r["agreement_id"]),
            _id_key(r["portfolio_id"]),
            _id_key(r["record_id"]),
        )
    )
    dataset.debtors.sort(key=lambda r: (rank[r["product_line"]], _id_key(r["debtor_id"])))
    dataset.contacts.sort(
        key=lambda r: (
            rank[r["product_line"]],
            _id_key(r["debtor_id"]),
            r["contact_date"] or "",
            _id_key(r["result_id"]),
        )
    )
    dataset.breaks.sort(
        key=lambda r: (
            rank[r["product_line"]],
            _id_key(r["debtor_id"]),
            r["agreement_date"] or "",
            _id_key(r["agreement_id"]),
            _id_key(r["portfolio_id"]),
        )
    )


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
