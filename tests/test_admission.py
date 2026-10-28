"""Admission: the entry contract applied to a whole build.

The first half judges real builds of a small SQLite world that is broken in one chosen way. The
second half judges the second route by changing the dataset in memory, one number at a time.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from delinquency_eda.admission import (
    FAIL,
    PASS,
    SKIPPED,
    Admission,
    admit,
    check_domain,
    check_grain,
    reconcile,
    render,
)
from delinquency_eda.config import Settings
from delinquency_eda.contract import CONTRACT_VERSION, RULE_IDS
from delinquency_eda.dataset import build_dataset
from delinquency_eda.db import SQLiteDatabase
from delinquency_eda.intake import Intake
from delinquency_eda.stages.s01_schema import check_database

from frames import raw
from world import damage, make_world, without_key

START, END, AS_OF = date(2026, 3, 1), date(2026, 3, 31), date(2026, 3, 31)
SETTINGS = Settings(grace_days=3)


def judge(path, settings=SETTINGS) -> Admission:
    with SQLiteDatabase.open(path) as db:
        return admit(
            {"auto": db},
            settings,
            {"auto": check_database(db, "auto", settings)},
            date_from=START,
            date_to=END,
            as_of=AS_OF,
        )


def failed_ids(admission: Admission) -> set[str]:
    return {r.id for r in admission.failed}


def by_id(admission: Admission) -> dict:
    return {r.id: r for r in admission.results}


# ---- a clean build -------------------------------------------------------------------------


def test_clean_world(world):
    admission = judge(world)
    assert admission.passed
    assert [r.id for r in admission.results] == list(RULE_IDS)
    assert {r.status for r in admission.results} == {PASS}
    assert all(r.checked > 0 for r in admission.results), "a rule that checks nothing is idle"
    assert admission.dataset is not None and len(admission.dataset.base) == 5


def test_record(world):
    record = judge(world).dataset.admission
    assert record["passed"] is True
    assert record["contract_version"] == CONTRACT_VERSION
    assert [r["id"] for r in record["rules"]] == list(RULE_IDS)
    assert all(r["status"] == "pass" and r["violations"] == 0 for r in record["rules"])
    assert {q["name"] for q in record["queries"]} >= {"slips", "reconcile_slips", "contacts"}
    assert json.loads(json.dumps(record)) == record, "the record is plain JSON"


def test_ledger(world):
    (line,) = judge(world).ledger
    # nine first-installment rows in March; two outside the generated statuses (106, and the
    # re-write of 109); one excluded agent (105); one older write of 108
    assert (line.window, line.status, line.agents, line.slips) == (9, 7, 6, 5)
    assert line.slips_value == 855.0 and line.unknown_agent == 0


def test_record_is_deterministic(world, tmp_path):
    other = make_world(tmp_path / "again.db")
    assert judge(world).record() == judge(other).record()


def test_two_lines(two_lines):
    admission = admit(
        two_lines,
        SETTINGS,
        {name: check_database(db, name, SETTINGS) for name, db in two_lines.items()},
        date_from=START,
        date_to=END,
        as_of=AS_OF,
    )
    assert admission.passed, [r.detail for r in admission.failed]
    assert [x.product_line for x in admission.ledger] == ["auto", "consumer"]


def test_demo_databases(demo_dir):
    databases = {
        "auto": SQLiteDatabase.open(demo_dir / "auto.db"),
        "consumer": SQLiteDatabase.open(demo_dir / "consumer.db"),
    }
    try:
        admission = admit(
            databases,
            SETTINGS,
            {name: check_database(db, name, SETTINGS) for name, db in databases.items()},
            date_from=date(2025, 11, 1),
            date_to=END,
            as_of=AS_OF,
        )
    finally:
        for db in databases.values():
            db.close()
    assert admission.passed, [(r.id, r.detail) for r in admission.failed]
    assert len(admission.dataset.base) > 1000


# ---- data that breaks one rule -------------------------------------------------------------

DEBTORS = "debtor_id INTEGER, person_key TEXT"
STATUSES = "status_id INTEGER, label TEXT"
AGENTS = "agent_id TEXT, display_name TEXT, login TEXT"
RESULTS = "result_id INTEGER, label TEXT, answered INTEGER, right_party INTEGER"


def update(table: str, assignment: str, where: str) -> str:
    return f"UPDATE {table} SET {assignment} WHERE {where}"


# A column of agreements changes type: rebuild the table with ``due_at`` declared as an integer.
DUE_AS_INTEGER = (
    "CREATE TABLE g (row_id INTEGER, agreement_id INTEGER, portfolio_id INTEGER, "
    "debtor_id INTEGER, installment_no INTEGER, amount REAL, plan_installments INTEGER, "
    "paid_amount REAL, issued_at TEXT, due_at INTEGER, paid_at TEXT, status_id INTEGER, "
    "agent_id TEXT); INSERT INTO g SELECT * FROM agreements; DROP TABLE agreements; "
    "ALTER TABLE g RENAME TO agreements;"
)
SYSTEM_TWIN = without_key(
    "agents", AGENTS, extra="INSERT INTO agents_copy VALUES ('a1', 'System X', 'system_x');"
)
RESULT_TWIN = without_key(
    "contact_results", RESULTS, extra="INSERT INTO contact_results_copy VALUES (3, 'Again', 1, 1);"
)

# (what is wrong, SQL that does it, the rules that must fail). A rule outside the set must not.
ONE_RULE = [
    (
        "a paid amount of zero",
        update("agreements", "paid_amount = 0", "row_id = 1"),
        {"row.payment"},
    ),
    (
        "a payment with no date",
        update("agreements", "paid_at = NULL", "row_id = 1"),
        {"row.payment"},
    ),
    (
        "a date with no payment",
        update("agreements", "paid_amount = NULL", "row_id = 1"),
        {"row.payment"},
    ),
    (
        "an amount that is not a number",
        update("agreements", "amount = 'n/a'", "row_id = 1"),
        {"row.readable"},
    ),
    (
        "an agreement date in another format",
        update("agreements", "issued_at = '03/01/2026'", "row_id = 1"),
        {"row.readable"},
    ),
    (
        "an agreement with no agent",
        update("agreements", "agent_id = NULL", "row_id = 1"),
        {"row.required"},
    ),
    (
        "a contact flag written as text",
        update("contact_results", "answered = 'Y'", "result_id = 3"),
        {"row.readable"},
    ),
    (
        "a contact flag left empty",
        update("contact_results", "right_party = NULL", "result_id = 3"),
        {"row.required"},
    ),
    (
        "a due date in another format",
        update("agreements", "due_at = '05/03/2026'", "row_id = 1"),
        {"row.due_date"},
    ),
    (
        "a first installment with no due date",
        update("agreements", "due_at = NULL", "row_id = 3"),
        {"row.due_date"},
    ),
    (
        "a slip whose debtor has no row",
        "DELETE FROM debtors WHERE debtor_id = 2",
        {"ref.debtor"},
    ),
    (
        "a contact event whose result is not in the catalog",
        "INSERT INTO contact_events VALUES (99, 2, 77, 'a1', '2026-03-06 10:00:00')",
        {"ref.contact_result"},
    ),
    (
        "a duplicate agent row that matches an exclusion pattern",
        SYSTEM_TWIN,
        {"recon.slips", "recon.outcome", "recon.value"},
    ),
    ("a duplicate row in the result catalog", RESULT_TWIN, {"recon.history"}),
    ("a date column declared as an integer", DUE_AS_INTEGER, {"source.types"}),
    (
        "a configured status missing from the catalog",
        "DELETE FROM agreement_statuses WHERE label = 'AUTO_BROKEN'",
        {"source.labels"},
    ),
    ("a table that is not there", "DROP TABLE debtors", {"source.tables"}),
]


@pytest.mark.parametrize(("what", "sql", "rules"), ONE_RULE, ids=[c[0] for c in ONE_RULE])
def test_rule_rejects(world, what, sql, rules):
    damage(world, sql)
    admission = judge(world)
    assert not admission.passed and admission.dataset is None, what
    assert failed_ids(admission) == rules, what


def test_duplicate_debtor(world):
    # A repeated debtor id doubles every slip that joins to it.
    twin = without_key("debtors", DEBTORS, extra="INSERT INTO debtors_copy VALUES (2, 'x');")
    damage(world, twin)
    admission = judge(world)
    assert {"key.grain", "recon.slips", "recon.outcome", "recon.value"} <= failed_ids(admission)
    grain = by_id(admission)["key.grain"]
    assert "slips that repeat an agreement and portfolio: 2" in grain.detail
    assert "debtor rows that repeat a debtor id: 1" in grain.detail


def test_duplicate_status(world):
    extra = (
        "INSERT INTO agreement_statuses_copy "
        "SELECT * FROM agreement_statuses WHERE label = 'ACTIVE';"
    )
    damage(world, without_key("agreement_statuses", STATUSES, extra=extra))
    assert {"key.grain", "recon.slips"} <= failed_ids(judge(world))


def test_failure_detail(world):
    damage(world, "UPDATE agreements SET agent_id = NULL WHERE row_id IN (1, 2)")
    rule = by_id(judge(world))["row.required"]
    assert rule.violations == 2 and rule.checked > 5
    assert rule.detail == ("slips.agent_id: 2 row(s)",)


def test_source_failure(world):
    damage(world, "DROP TABLE debtors")
    admission = judge(world)
    results = by_id(admission)
    assert results["source.tables"].status == FAIL
    assert {r.status for r in admission.results if r.id != "source.tables"} <= {PASS, SKIPPED}
    assert results["row.required"].status == SKIPPED
    assert results["recon.slips"].status == SKIPPED
    assert admission.queries == [], "no query ran against a database that failed the source rules"


def test_bad_rows_skip_recon(world):
    # With rows left out every count would differ; reporting that repeats the problem.
    damage(world, "UPDATE agreements SET agent_id = NULL WHERE row_id = 1")
    results = by_id(judge(world))
    assert results["row.required"].status == FAIL
    assert results["key.grain"].status == PASS and results["base.domain"].status == PASS
    held_back = ("row.due_date", "ref.debtor", "ref.contact_result", "recon.slips", "recon.history")
    for rule_id in held_back:
        assert results[rule_id].status == SKIPPED
        assert "left out at intake" in results[rule_id].detail[0]


def test_all_failures_listed(world):
    damage(
        world,
        "UPDATE agreements SET paid_amount = 0 WHERE row_id = 1",
        "UPDATE contact_results SET answered = 'Y' WHERE result_id = 3",
        "UPDATE agreements SET agent_id = NULL WHERE row_id = 2",
    )
    assert failed_ids(judge(world)) == {"row.payment", "row.readable", "row.required"}


# ---- the pure rules ------------------------------------------------------------------------


def test_grain_per_line():
    from frames import dataset

    ds = dataset(
        [raw(1), raw(1, product_line="consumer")], lines=("auto", "consumer")
    )
    ds.debtors = [
        {"product_line": "auto", "debtor_id": 1, "person_key": "k"},
        {"product_line": "consumer", "debtor_id": 1, "person_key": "k"},
    ]
    assert check_grain(ds).status == PASS


def test_grain_repeats():
    from frames import dataset

    ds = dataset([raw(1), raw(1), raw(2, record_id=1)])
    ds.debtors = [{"product_line": "auto", "debtor_id": 5, "person_key": "k"}] * 3
    result = check_grain(ds)
    assert result.status == FAIL
    assert result.detail == (
        "slips that repeat an agreement and portfolio: 1",
        "slips that repeat a row id: 2",
        "debtor rows that repeat a debtor id: 2",
    )
    assert result.checked == 2 * 3 + 3


@pytest.mark.parametrize(
    ("change", "flaw"),
    [
        ({"bucket": "lost"}, "a bucket the pipeline does not define"),
        ({"status_label": "PENDING_EXCEPTION"}, "a status label outside generated_status_labels"),
        ({"status_label": None}, "a status label outside generated_status_labels"),
        ({"product_line": "boat"}, "a product line that was not requested"),
        ({"due_date": "2026-05-01"}, "a due date outside the requested window"),
        ({"due_date": "2025-12-31"}, "a due date outside the requested window"),
    ],
)
def test_domain_flaws(change, flaw):
    from frames import dataset

    ds = dataset([raw(1)])
    ds.base[0].update(change)
    result = check_domain(ds)
    assert result.status == FAIL and result.violations == 1
    assert result.detail == (f"slips with {flaw}: 1",)


def test_domain_edges():
    from frames import dataset

    ds = dataset([raw(1, status_label="ACTIVE "), raw(2, due_date="2026-01-01")])
    ds.meta.period_from, ds.meta.period_to = "2026-01-01", "2026-03-10"
    assert check_domain(ds).status == PASS


# ---- the second route, one number at a time ------------------------------------------------


@pytest.fixture
def compared(world):
    """The extraction of the test world, ready to be changed before it is compared again."""
    intake = Intake()
    db = SQLiteDatabase.open(world)
    try:
        ds = build_dataset(
            {"auto": db}, SETTINGS, date_from=START, date_to=END, as_of=AS_OF, intake=intake
        )

        def run():
            results, _ = reconcile(
                {"auto": db}, SETTINGS, ds, intake, date_from=START, date_to=END, as_of=AS_OF
            )
            return {r.id: r for r in results}

        yield ds, run
    finally:
        db.close()


def test_recon_agrees(compared):
    _, run = compared
    assert {r.status for r in run().values()} == {PASS}


def test_recon_missing_slip(compared):
    ds, run = compared
    ds.base.pop()
    results = run()
    assert results["recon.slips"].detail == ("auto: the dataset has 4, the second query 5",)
    assert results["recon.value"].status == FAIL


def test_recon_bucket(compared):
    ds, run = compared
    row = next(r for r in ds.base if r["bucket"] == "overdue_unpaid")
    row["bucket"] = "paid_on_time"
    results = run()
    assert failed_ids_of(results) == {"recon.outcome"}
    assert results["recon.outcome"].violations == 2


def test_recon_amount(compared):
    ds, run = compared
    ds.base[0]["first_installment_value"] += 5
    results = run()
    assert failed_ids_of(results) == {"recon.value"}
    assert "first-installment value" in results["recon.value"].detail[0]


def test_recon_agreement_total(compared):
    ds, run = compared
    ds.base[0]["total_value"] += 5
    assert "whole-agreement value" in run()["recon.value"].detail[0]


def test_recon_installments(compared):
    ds, run = compared
    ds.base[0]["total_installments"] += 1
    assert "installments" in run()["recon.value"].detail[0]


def test_recon_cent_tolerance(compared):
    # SQL and Python add money in a different order, so sub-cent noise is not a difference.
    ds, run = compared
    ds.base[0]["first_installment_value"] += 0.004
    assert run()["recon.value"].status == PASS


def test_recon_lost_contact(compared):
    ds, run = compared
    ds.contacts.pop()
    assert failed_ids_of(run()) == {"recon.history"}


def test_recon_lost_break(compared):
    ds, run = compared
    ds.breaks.pop()
    results = run()
    assert failed_ids_of(results) == {"recon.history"}
    assert "earlier breaks" in results["recon.history"].detail[0]


def failed_ids_of(results) -> set[str]:
    return {rule_id for rule_id, r in results.items() if r.status == FAIL}


# ---- the report ----------------------------------------------------------------------------


def test_report_admitted(world, tmp_path):
    text = render(judge(world), dataset_path=tmp_path / "d.json", previous_file_kept=False).render()
    assert "**Result: ADMITTED.**" in text and "All 15 entry rules passed" in text
    assert "## Filter ledger" in text and "## Queries run" in text
    assert "What to fix" not in text


def test_report_rejected(world, tmp_path):
    damage(world, "UPDATE agreements SET paid_amount = 0 WHERE row_id = 1")
    admission = judge(world)
    path = tmp_path / "d.json"
    text = render(admission, dataset_path=path, previous_file_kept=False).render()
    assert "**Result: REJECTED.** 1 of 15 entry rules failed. No dataset was written" in text
    assert "was left as it was" not in text
    assert "### `row.payment`: fail" in text
    assert "To fix it: Return NULL in both columns" in text
    assert "### `recon.slips`: skipped" in text
    kept = render(admission, dataset_path=path, previous_file_kept=True).render()
    assert f"The file at `{path}` was left as it was" in kept


def test_report_has_no_row_values(world, tmp_path):
    damage(world, "UPDATE agreements SET paid_amount = 0 WHERE row_id = 1")
    text = render(judge(world), dataset_path=tmp_path / "d.json", previous_file_kept=False).render()
    for secret in ("pk_alpha", "pk_beta", "agent_01", "Agent 01", "Portfolio A-01", "a1"):
        assert secret not in text
    record = json.dumps(judge(make_world(tmp_path / "ok.db")).record())
    for secret in ("pk_alpha", "agent_01", "Portfolio A-01"):
        assert secret not in record
