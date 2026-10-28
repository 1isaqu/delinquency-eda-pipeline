"""Stage 01: schema discovery against the contract."""

from __future__ import annotations

import sqlite3

from delinquency_eda.config import Settings
from delinquency_eda.contract import TABLES, TableSpec
from delinquency_eda.db import SQLiteDatabase
from delinquency_eda.db.base import ColumnInfo
from delinquency_eda.stages.s01_schema import check_database, check_table, configured_labels, render

from clirun import run_cli, sqlite_args
from world import damage


def test_check_table_reports_missing_required_missing_optional_and_extra():
    spec = TableSpec("t", required=("a", "b"), optional=("c",))
    check = check_table(spec, [ColumnInfo("A", "int", False), ColumnInfo("x", "text", True)])
    assert check.missing_required == ["b"]
    assert check.missing_optional == ["c"]
    assert check.extra == ["x"]
    assert check.exists and not check.ok


def test_names_match_case_insensitively():
    spec = TableSpec("t", required=("agreement_id",))
    assert check_table(spec, [ColumnInfo("AGREEMENT_ID", "int", False)]).ok


def test_an_absent_table_is_not_ok_and_lists_no_columns():
    check = check_table(TableSpec("t", required=("a",)), [])
    assert not check.exists and not check.ok
    assert check.missing_required == []


def test_the_test_world_satisfies_the_whole_contract(world):
    with SQLiteDatabase.open(world) as db:
        line = check_database(db, "auto", Settings())
    assert line.ready("slips") and line.ready("base")
    assert line.labels_checked and line.missing_labels == []
    assert {t.spec.name for t in line.tables} == {t.name for t in TABLES}


def test_contact_events_lists_only_agent_id_as_optional(world):
    with SQLiteDatabase.open(world) as db:
        line = check_database(db, "auto", Settings())
    events = next(t for t in line.tables if t.spec.name == "contact_events")
    assert events.spec.optional == ("agent_id",)
    assert events.missing_optional == []


def test_a_missing_base_only_table_leaves_the_slip_stages_ready(world):
    connection = sqlite3.connect(world)
    connection.execute("DROP TABLE debtors")
    connection.commit()
    connection.close()
    with SQLiteDatabase.open(world) as db:
        line = check_database(db, "auto", Settings())
    assert line.ready("slips")
    assert not line.ready("base")


def test_a_configured_status_label_missing_from_the_catalog_is_reported(world):
    settings = Settings(generated_status_labels=("ACTIVE", "NOT_A_STATUS"))
    with SQLiteDatabase.open(world) as db:
        line = check_database(db, "auto", settings)
    assert line.missing_labels == ["NOT_A_STATUS"]
    assert not line.ready("slips")


def test_status_labels_are_not_checked_without_a_catalog(world):
    connection = sqlite3.connect(world)
    connection.execute("DROP TABLE agreement_statuses")
    connection.commit()
    connection.close()
    with SQLiteDatabase.open(world) as db:
        line = check_database(db, "auto", Settings())
    assert not line.labels_checked
    assert "not checked" in render([line]).render()


def test_configured_labels_are_unique_and_ordered():
    settings = Settings(
        generated_status_labels=("A", "B"), broken_status_labels=("B", "C"), manual_break_label="C"
    )
    assert configured_labels(settings) == ["A", "B", "C"]


def test_the_report_lists_types_of_required_columns(world):
    with SQLiteDatabase.open(world) as db:
        text = render([check_database(db, "auto", Settings())]).render()
    assert "Types of the required columns found" in text
    assert "| agreements | installment_no |" in text


def test_cli_exits_zero_when_ready_and_one_when_not(world, tmp_path):
    out_dir = str(tmp_path / "r")
    code, out = run_cli(*sqlite_args(("auto", world)), "--out-dir", out_dir, "schema")
    assert code == 0 and "ready for base = True" in out
    connection = sqlite3.connect(world)
    connection.execute("DROP TABLE contact_results")
    connection.commit()
    connection.close()
    code, out = run_cli(*sqlite_args(("auto", world)), "--out-dir", out_dir, "schema")
    assert code == 1
    assert "contact_results: missing" in out


# ---- declared types ------------------------------------------------------------------------


def test_wrong_type():
    spec = TableSpec("agreements", required=("amount", "due_at", "agent_id"))
    columns = [
        ColumnInfo("amount", "varchar(20)", False),
        ColumnInfo("due_at", "int", False),
        ColumnInfo("agent_id", "varchar(10)", False),
    ]
    check = check_table(spec, columns)
    assert [(m.column, m.found, m.kind) for m in check.wrong_types] == [
        ("amount", "varchar(20)", "number"),
        ("due_at", "int", "date"),
    ]
    assert check.structure_ok and not check.ok


def test_unjudged_types():
    spec = TableSpec("debts", required=("debt_id", "balance"), optional=("due_at",))
    columns = [
        ColumnInfo("debt_id", "int", False),
        ColumnInfo("balance", "decimal(12,2)", True),
        ColumnInfo("due_at", "xml", True),
        ColumnInfo("note", "varbinary(4)", True),
    ]
    assert check_table(spec, columns).ok


def test_world_types_fit(world):
    with SQLiteDatabase.open(world) as db:
        line = check_database(db, "auto", Settings())
    assert [m for t in line.tables for m in t.wrong_types] == []


def test_wrong_type_not_ready(world, tmp_path):
    damage(
        world,
        "CREATE TABLE d2 (debtor_id TEXT, person_key INTEGER);"
        "INSERT INTO d2 SELECT * FROM debtors; DROP TABLE debtors;"
        "ALTER TABLE d2 RENAME TO debtors;",
    )
    with SQLiteDatabase.open(world) as db:
        line = check_database(db, "auto", Settings())
    assert line.ready("slips") and not line.ready("base")  # debtors is a base-only table
    code, out = run_cli(*sqlite_args(("auto", world)), "--out-dir", str(tmp_path / "r"), "schema")
    assert code == 1
    assert "debtors.debtor_id: declared 'text', the contract needs integer" in out
    assert "debtors.person_key: declared 'integer', the contract needs text" in out
    report = (tmp_path / "r" / "schema.md").read_text(encoding="utf-8")
    assert "| WRONG |" in report and "| wrong type |" in report


def test_report_kinds(world):
    with SQLiteDatabase.open(world) as db:
        text = render([check_database(db, "auto", Settings())]).render()
    assert "| agreements | due_at | text | yes | date | ok |" in text
