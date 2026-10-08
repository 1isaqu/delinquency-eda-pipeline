"""The extraction: SQL construction for both dialects, and the real SQL run on a SQLite world."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import date

import pytest

from delinquency_eda import derive
from delinquency_eda.config import Settings
from delinquency_eda.contract import TABLE_NAMES
from delinquency_eda.dataset import Dataset, build_dataset, fetch_slips
from delinquency_eda.db import SQLiteDatabase, SQLiteDialect, SQLServerDialect
from delinquency_eda.errors import DatasetError, ReadOnlyViolation
from delinquency_eda.features import (
    FOLLOWUP_ATTEMPTS,
    FOLLOWUP_CPC,
    RATIO_NO_LINK,
    RATIO_OK,
    RATIO_ZERO,
    analysis_frame,
    enrich,
)
from delinquency_eda.queries import (
    build_breaks_query,
    build_contacts_query,
    build_person_debtors_query,
    build_slip_query,
)

from world import make_world

START, END, AS_OF = date(2026, 3, 1), date(2026, 3, 31), date(2026, 3, 31)
SETTINGS = Settings(grace_days=3)

# ---- SQL text ------------------------------------------------------------------------------

TABLE_REF = re.compile(
    r"\b(?:FROM|JOIN)\s+(" + "|".join(TABLE_NAMES) + r")\s+AS\s+\w+(\s+WITH\s*\(NOLOCK\))?",
    re.IGNORECASE,
)


def unhinted_tables(sql: str) -> list[str]:
    return [m.group(1) for m in TABLE_REF.finditer(sql) if not m.group(2)]


def all_queries(dialect):
    detail = build_slip_query(dialect, SETTINGS, START, END, detail=True)
    lean = build_slip_query(dialect, SETTINGS, START, END, detail=False)
    debtors = build_person_debtors_query(dialect, ["k1", "k2"])
    contacts = build_contacts_query(dialect, [1, 2], START, END)
    breaks = build_breaks_query(dialect, SETTINGS, [1, 2])
    return [detail, lean, debtors, contacts, breaks]


def test_every_table_reference_carries_nolock_on_sql_server():
    for query in all_queries(SQLServerDialect()):
        assert TABLE_REF.search(query.sql), "the checker found no table reference at all"
        assert unhinted_tables(query.sql) == []


def test_sqlite_sql_has_no_sql_server_hints():
    for query in all_queries(SQLiteDialect()):
        assert "NOLOCK" not in query.sql
        assert "OUTER APPLY" not in query.sql


def test_sql_server_resolves_the_portfolio_with_a_single_top_one_lookup():
    sql = build_slip_query(SQLServerDialect(), SETTINGS, START, END, detail=True).sql
    assert "OUTER APPLY" in sql
    assert "SELECT TOP 1 da.portfolio_label" in sql
    assert "ORDER BY ad.debt_id" in sql
    assert "portfolio_label IS NOT NULL" in sql


def test_lean_query_skips_the_detail_joins():
    sql = build_slip_query(SQLServerDialect(), SETTINGS, START, END, detail=False).sql
    for name in ("debtors", "debt_attributes", "debts", "debt_totals"):
        assert name not in sql
    assert "totals" in sql  # totals are cheap and the card needs them


def test_filters_sit_inside_the_ranked_table_before_the_row_number():
    sql = build_slip_query(SQLServerDialect(), SETTINGS, START, END, detail=False).sql
    ranked = sql.split("selected AS")[0]
    assert "ROW_NUMBER()" in ranked
    for needle in ("r.installment_no = 0", "r.due_at >= ?", "g.label IN", "NOT LIKE ?"):
        assert needle in ranked
    assert "WHERE rn = 1" in sql


def test_parameter_order_and_binding_per_dialect():
    settings = Settings(
        generated_status_labels=("ACTIVE", "PAID"), excluded_agent_patterns=("Sys%",)
    )
    server = build_slip_query(SQLServerDialect(), settings, START, END, detail=False)
    assert server.params == (date(2026, 3, 1), date(2026, 4, 1), "ACTIVE", "PAID", "sys%", "sys%")
    lite = build_slip_query(SQLiteDialect(), settings, START, END, detail=False)
    assert lite.params == ("2026-03-01", "2026-04-01", "ACTIVE", "PAID", "sys%", "sys%")
    assert server.sql.count("?") == len(server.params)


def test_no_excluded_patterns_means_no_agent_clause():
    settings = Settings(excluded_agent_patterns=())
    query = build_slip_query(SQLiteDialect(), settings, START, END, detail=False)
    assert "NOT LIKE" not in query.sql
    assert query.sql.count("?") == len(query.params)


def test_in_lists_have_one_marker_per_value():
    q = build_contacts_query(SQLiteDialect(), [5, 6, 7], START, END)
    assert q.sql.count("?") == len(q.params) == 5
    q = build_breaks_query(SQLiteDialect(), SETTINGS, [5, 6, 7])
    assert q.params == ("BROKEN", "AUTO_BROKEN", 5, 6, 7)
    assert q.sql.count("?") == 5


def test_schema_qualifies_every_table():
    sql = build_slip_query(SQLServerDialect(), SETTINGS, START, END, detail=True, schema="dbo").sql
    assert "dbo.agreements AS r WITH (NOLOCK)" in sql
    assert "dbo.debtors AS dbt WITH (NOLOCK)" in sql


# ---- extraction against SQLite -------------------------------------------------------------


def slips(world, **kwargs):
    with SQLiteDatabase.open(world) as db:
        return {
            r["agreement_id"]: r
            for r in fetch_slips(
                db, "auto", SETTINGS, kwargs.pop("start", START), kwargs.pop("end", END),
                as_of=AS_OF, detail=True,
            )
        }  # fmt: skip


def test_only_in_scope_first_installments_are_extracted(world):
    rows = slips(world)
    # 105 excluded agent, 106 wrong status, 107 outside the window, 104 due after the window.
    assert sorted(rows) == [101, 102, 103, 108, 109]


def test_outcomes_follow_the_reference_date_and_grace_period(world):
    rows = slips(world)
    assert rows[101]["bucket"] == derive.PAID_ON_TIME
    assert rows[102]["bucket"] == derive.OVERDUE_UNPAID
    assert rows[103]["bucket"] == derive.OVERDUE_UNPAID  # paid, but after the grace period
    assert rows[103]["paid_amount"] == 300.0
    assert rows[102]["days_overdue"] == 21
    assert rows[102]["days_to_due"] == 8


def test_a_later_due_date_brings_in_an_unmatured_slip(world):
    rows = slips(world, end=date(2026, 4, 30))
    assert rows[104]["bucket"] == derive.TO_MATURE
    assert rows[104]["days_overdue"] < 0


def test_rewritten_first_installment_keeps_the_latest_row(world):
    row = slips(world)[108]
    assert row["record_id"] == 9
    assert row["first_installment_value"] == 175.0
    assert row["status_label"] == "BROKEN"


def test_filters_run_before_ranking_so_a_valid_earlier_row_survives(world):
    row = slips(world)[109]
    assert row["record_id"] == 10
    assert row["first_installment_value"] == 80.0
    assert row["status_label"] == "ACTIVE"


def test_totals_count_each_installment_once_using_its_latest_row(world):
    row = slips(world)[102]
    assert row["total_installments"] == 3
    assert row["total_value"] == 200.0 + 200.0 + 250.0
    assert slips(world)[101]["total_installments"] == 1


def test_declared_plan_is_kept_apart_from_generated_installments(world):
    assert slips(world)[101]["plan_installments"] == 3
    assert slips(world)[101]["total_installments"] == 1


def test_portfolio_label_is_the_lowest_debt_id_with_a_label(world):
    rows = slips(world)
    assert rows[102]["portfolio_label"] == "Bank A II"  # debt 2, though debt 3 was linked first
    assert rows[103]["portfolio_label"] is None  # its only debt has no label
    assert rows[109]["portfolio_label"] is None  # no linked debt at all
    assert rows[101]["portfolio_label"] == "Bank A I"


def test_original_debt_distinguishes_no_link_from_zero_balance(world):
    rows = slips(world)
    assert rows[102]["original_debt"] == 100.0  # 0.0 + 100.0
    assert rows[109]["original_debt"] is None
    assert rows[101]["original_debt"] == 500.0


def test_statuses_are_resolved_by_label_not_by_id(tmp_path):
    shifted = make_world(tmp_path / "shifted.db", status_base=70)
    assert sorted(slips(shifted)) == [101, 102, 103, 108, 109]


def test_agent_fields_and_person_key_are_carried(world):
    row = slips(world)[101]
    assert (row["agent_name"], row["agent_login"], row["person_key"]) == (
        "Agent 01",
        "agent_01",
        "pk_alpha",
    )


def test_lean_rows_have_no_detail_columns(world):
    with SQLiteDatabase.open(world) as db:
        rows = fetch_slips(db, "auto", SETTINGS, START, END, as_of=AS_OF, detail=False)
    assert rows and all(r["person_key"] is None and r["portfolio_label"] is None for r in rows)
    assert {r["agreement_id"]: r["total_value"] for r in rows}[102] == 650.0


def test_the_database_is_opened_read_only(world):
    with SQLiteDatabase.open(world) as db, pytest.raises(ReadOnlyViolation):
        db.query("DELETE FROM agents")


# ---- the assembled dataset -----------------------------------------------------------------


def test_dataset_links_debtors_contacts_and_breaks_across_product_lines(two_lines):
    ds = build_dataset(two_lines, SETTINGS, date_from=START, date_to=END, as_of=AS_OF)
    assert ds.meta.product_lines == ["auto", "consumer"]
    assert len(ds.base) == 6
    lines = {(r["product_line"], r["agreement_id"]) for r in ds.base}
    assert ("auto", 101) in lines and ("consumer", 101) in lines  # same id, two agreements
    auto_debtors = {r["debtor_id"] for r in ds.debtors if r["product_line"] == "auto"}
    assert auto_debtors == {1, 2, 3, 5}  # debtor 5 is only reachable through the person key
    assert len(ds.contacts) == 6  # the April event is outside the window
    assert {(b["product_line"], b["agreement_id"]) for b in ds.breaks} == {
        ("auto", 90), ("auto", 91), ("auto", 93), ("auto", 108),
    }  # fmt: skip


def test_dataset_round_trips_through_json(two_lines, tmp_path):
    ds = build_dataset(two_lines, SETTINGS, date_from=START, date_to=END, as_of=AS_OF)
    path = ds.save(tmp_path / "out" / "dataset.json")
    again = Dataset.load(path)
    assert again.meta == ds.meta
    assert again.base == ds.base and again.contacts == ds.contacts and again.breaks == ds.breaks


def test_empty_period_gives_an_empty_but_valid_dataset(two_lines):
    ds = build_dataset(
        two_lines, SETTINGS, date_from=date(2020, 1, 1), date_to=date(2020, 1, 31), as_of=AS_OF
    )
    assert ds.base == [] and ds.contacts == []
    assert list(ds.base_frame().columns)[:2] == ["product_line", "record_id"]


def test_reversed_period_is_an_error(two_lines):
    with pytest.raises(DatasetError, match="before its start"):
        build_dataset(two_lines, SETTINGS, date_from=END, date_to=START, as_of=AS_OF)


def test_loading_a_bad_file_is_a_clear_error(tmp_path):
    with pytest.raises(DatasetError, match="cannot read"):
        Dataset.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(DatasetError, match="not valid JSON"):
        Dataset.load(bad)
    bad.write_text('{"meta": {"schema_version": 99}}', encoding="utf-8")
    with pytest.raises(DatasetError, match="missing a section"):
        Dataset.load(bad)
    bad.write_text(
        '{"meta": {"schema_version": 99}, "base": [], "debtors": [], "contacts": [], "breaks": []}',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="schema version"):
        Dataset.load(bad)


def test_ids_chunking_still_finds_everything(two_lines):
    small = Settings(grace_days=3, id_chunk_size=1)
    big = build_dataset(two_lines, SETTINGS, date_from=START, date_to=END, as_of=AS_OF)
    tiny = build_dataset(two_lines, small, date_from=START, date_to=END, as_of=AS_OF)
    key = lambda r: sorted(r.items(), key=str)  # noqa: E731
    assert sorted(map(key, tiny.contacts)) == sorted(map(key, big.contacts))
    assert sorted(map(key, tiny.breaks)) == sorted(map(key, big.breaks))
    assert sorted(map(key, tiny.debtors)) == sorted(map(key, big.debtors))


# ---- enrichment ----------------------------------------------------------------------------


@pytest.fixture
def enriched(two_lines):
    ds = build_dataset(two_lines, SETTINGS, date_from=START, date_to=END, as_of=AS_OF)
    frame = enrich(ds)
    return frame.set_index(["product_line", "agreement_id"])


def test_follow_up_counts_pool_every_debtor_row_of_the_same_person(enriched):
    r = enriched.loc[("auto", 103)]
    # debtor 3 itself has no events; its person's other debtor rows (auto 1, auto 5, consumer 1)
    # do, and the 04-20 event is outside the window.
    assert (r.contacts_total, r.contacts_after, r.answered_after, r.cpc_after) == (3, 3, 2, 1)
    assert r.followup_level == FOLLOWUP_CPC


def test_contact_on_the_agreement_day_is_not_follow_up(enriched):
    r = enriched.loc[("auto", 102)]
    assert (r.contacts_total, r.contacts_after) == (3, 2)
    assert (r.answered_after, r.cpc_after) == (1, 1)


def test_attempts_only_level(enriched):
    r = enriched.loc[("auto", 109)]
    assert (r.contacts_after, r.answered_after, r.cpc_after) == (1, 0, 0)
    assert r.followup_level == FOLLOWUP_ATTEMPTS


def test_prior_breaks_pool_the_person_and_exclude_the_agreement_itself(enriched):
    assert enriched.loc[("auto", 103)].prior_breaks == 2
    assert enriched.loc[("auto", 103)].prior_manual_breaks == 1  # one BROKEN, one AUTO_BROKEN
    assert enriched.loc[("consumer", 101)].prior_breaks == 2  # history is found across lines


def test_the_manual_break_label_is_the_one_the_dataset_was_built_with(two_lines):
    ds = build_dataset(two_lines, SETTINGS, date_from=START, date_to=END, as_of=AS_OF)
    edited = replace(SETTINGS, manual_break_label="PAID")  # a settings file changed after `base`
    frame = analysis_frame(ds, edited).set_index(["product_line", "agreement_id"])
    assert frame.loc[("auto", 103)].prior_manual_breaks == 1


def test_a_break_made_after_the_agreement_is_not_history(enriched):
    assert enriched.loc[("auto", 102)].prior_breaks == 0  # 93 was made on 03-20, 102 on 03-02
    assert enriched.loc[("auto", 108)].prior_breaks == 0  # its own break does not count


def test_ratio_states(enriched):
    assert enriched.loc[("auto", 101)].ratio_state == RATIO_OK
    assert enriched.loc[("auto", 101)].first_installment_ratio == pytest.approx(100.0 / 500.0)
    assert enriched.loc[("auto", 109)].ratio_state == RATIO_NO_LINK
    assert str(enriched.loc[("auto", 109)].first_installment_ratio) == "nan"


def test_zero_balance_is_not_the_same_as_no_link(tmp_path):
    rows = [
        (1, 5, 1, 1, 0, 100.0, 1, None, "2026-03-01", "2026-03-10", None, "ACTIVE", "a1"),
    ]
    path = make_world(tmp_path / "z.db", rows=rows)
    import sqlite3

    con = sqlite3.connect(path)
    con.execute("INSERT INTO agreement_debts VALUES (5, 1, 2)")  # debt 2 has balance 0.0
    con.commit()
    con.close()
    with SQLiteDatabase.open(path) as db:
        ds = build_dataset({"auto": db}, SETTINGS, date_from=START, date_to=END, as_of=AS_OF)
    frame = enrich(ds)
    assert frame.loc[0, "original_debt"] == 0.0
    assert frame.loc[0, "ratio_state"] == RATIO_ZERO


# ---- ordering ------------------------------------------------------------------------------


class ReversedRows:
    """A database wrapper that hands every result set back in reverse order."""

    def __init__(self, inner):
        self._inner = inner
        self.dialect, self.schema = inner.dialect, inner.schema

    def query(self, sql, params=()):
        return list(reversed(self._inner.query(sql, params)))

    def table_columns(self, table):
        return self._inner.table_columns(table)

    def close(self):
        self._inner.close()


def test_the_dataset_does_not_depend_on_the_order_rows_come_back_in(two_lines):
    normal = build_dataset(two_lines, SETTINGS, date_from=START, date_to=END, as_of=AS_OF)
    flipped = build_dataset(
        {line: ReversedRows(db) for line, db in two_lines.items()},
        SETTINGS,
        date_from=START,
        date_to=END,
        as_of=AS_OF,
    )
    for table in ("base", "debtors", "contacts", "breaks"):
        assert getattr(flipped, table) == getattr(normal, table), table
    assert normal.contacts and normal.breaks and normal.debtors
