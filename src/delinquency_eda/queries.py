"""SQL builders for the extraction. Pure functions: text and parameters out, no database in.

Rules every query follows (they are the pipeline's standard behavior, not options):

* read only, and every table reference carries the dialect's non-blocking hint,
* re-written installments are collapsed: the latest row per
  ``(agreement_id, portfolio_id, installment_no)`` wins. Business filters (date window, status,
  excluded agents) sit *inside* the ranked derived table, before ``ROW_NUMBER``. Filtering after
  ranking would let a re-write that falls outside the filter hide the valid row.
* an agreement whose agent has no row in ``agents`` is kept, with a null name and login, so the
  join to ``agents`` never drops a slip,
* the portfolio label is resolved with a single top-one lookup per agreement, so joins never
  multiply rows,
* dates are bound parameters, never ``GETDATE()``: the reference date comes from the caller, so a
  run can be reproduced.

Parameter order in each builder is documented on the returned ``Query``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from delinquency_eda.config import Settings
from delinquency_eda.dates import next_day
from delinquency_eda.db.dialect import Dialect
from delinquency_eda.sqlutil import placeholders

LATEST = (
    "ROW_NUMBER() OVER (\n"
    "            PARTITION BY {a}.agreement_id, {a}.portfolio_id, {a}.installment_no\n"
    "            ORDER BY {a}.issued_at DESC, {a}.row_id DESC) AS rn"
)


@dataclass(frozen=True)
class Query:
    sql: str
    params: tuple


def _excluded_agents(alias: str, patterns: Sequence[str]) -> tuple[str, list[str]]:
    """``NOT LIKE`` clauses on login and display name for every configured pattern."""
    clauses: list[str] = []
    params: list[str] = []
    for pattern in patterns:
        for column in ("login", "display_name"):
            clauses.append(f"AND COALESCE(LOWER(LTRIM(RTRIM({alias}.{column}))), '') NOT LIKE ?")
            params.append(pattern.lower())
    return "\n      ".join(clauses), params


def build_slip_query(
    dialect: Dialect,
    settings: Settings,
    date_from: date,
    date_to: date,
    *,
    detail: bool,
    schema: str | None = None,
) -> Query:
    """First-installment slips whose due date falls in ``[date_from, date_to]`` (inclusive).

    ``detail=False`` returns the columns the funnel needs (amounts, dates, totals). ``detail=True``
    adds status label, person key, original debt balance and portfolio label for the base build.

    Parameters, in order: lower due bound, upper (exclusive) due bound, one per generated status
    label, then two per excluded agent pattern (login, display name).
    """
    t = dialect.table
    agent_sql, agent_params = _excluded_agents("u", settings.excluded_agent_patterns)
    labels = placeholders(len(settings.generated_status_labels))
    agent_block = f"\n      {agent_sql}" if agent_sql else ""

    ctes = f"""WITH ranked AS (
    SELECT r.row_id, r.agreement_id, r.portfolio_id, r.debtor_id, r.status_id, r.amount,
           r.plan_installments, r.issued_at, r.due_at, r.paid_amount, r.paid_at, r.agent_id,
           u.login AS agent_login, u.display_name AS agent_name,
           {LATEST.format(a="r")}
    FROM {t("agreements", "r", schema)}
    LEFT JOIN {t("agents", "u", schema)} ON u.agent_id = r.agent_id
    WHERE r.installment_no = 0
      AND r.due_at >= ? AND r.due_at < ?
      AND r.status_id IN (
          SELECT g.status_id FROM {t("agreement_statuses", "g", schema)}
          WHERE g.label IN ({labels}))
      {agent_block.strip()}
),
selected AS (
    SELECT * FROM ranked WHERE rn = 1
),
latest_installments AS (
    SELECT x.agreement_id, x.portfolio_id, x.amount,
           {LATEST.format(a="x")}
    FROM {t("agreements", "x", schema)}
    WHERE EXISTS (
        SELECT 1 FROM selected AS s
        WHERE s.agreement_id = x.agreement_id AND s.portfolio_id = x.portfolio_id)
),
totals AS (
    SELECT agreement_id, portfolio_id,
           SUM(amount) AS total_value, COUNT(*) AS total_installments
    FROM latest_installments
    WHERE rn = 1
    GROUP BY agreement_id, portfolio_id
)"""

    select = [
        "s.row_id AS record_id",
        "s.agreement_id",
        "s.portfolio_id",
        "s.debtor_id",
        "s.status_id",
        "s.amount AS first_installment_value",
        "s.plan_installments",
        "COALESCE(tt.total_value, s.amount) AS total_value",
        "COALESCE(tt.total_installments, 1) AS total_installments",
        "s.agent_id",
        "s.agent_name",
        "s.agent_login",
        "s.issued_at AS agreement_date",
        "s.due_at AS due_date",
        "s.paid_amount",
        "s.paid_at AS paid_date",
    ]
    joins = [
        "LEFT JOIN totals AS tt ON tt.agreement_id = s.agreement_id "
        "AND tt.portfolio_id = s.portfolio_id"
    ]

    if detail:
        debt_totals = f""",
debt_totals AS (
    SELECT ad.agreement_id, ad.portfolio_id, SUM(d.balance) AS original_debt
    FROM {t("agreement_debts", "ad", schema)}
    INNER JOIN {t("debts", "d", schema)} ON d.debt_id = ad.debt_id
    WHERE EXISTS (
        SELECT 1 FROM selected AS s
        WHERE s.agreement_id = ad.agreement_id AND s.portfolio_id = ad.portfolio_id)
    GROUP BY ad.agreement_id, ad.portfolio_id
)"""
        ctes += debt_totals
        portfolio_join, portfolio_expr = dialect.first_portfolio_label(schema, "s")
        select += [
            "st.label AS status_label",
            "dt.original_debt",
            "dbt.person_key",
            f"{portfolio_expr} AS portfolio_label",
        ]
        joins += [
            f"LEFT JOIN {t('agreement_statuses', 'st', schema)} ON st.status_id = s.status_id",
            f"LEFT JOIN {t('debtors', 'dbt', schema)} ON dbt.debtor_id = s.debtor_id",
            "LEFT JOIN debt_totals AS dt ON dt.agreement_id = s.agreement_id "
            "AND dt.portfolio_id = s.portfolio_id",
        ]
        if portfolio_join:
            joins.append(portfolio_join)

    sql = (
        f"{ctes}\n"
        f"SELECT {', '.join(select)}\n"
        "FROM selected AS s\n" + "\n".join(joins) + "\nORDER BY s.agreement_id, s.portfolio_id"
    )
    params = (
        dialect.bind_date(date_from),
        dialect.bind_date(next_day(date_to)),
        *settings.generated_status_labels,
        *agent_params,
    )
    return Query(sql, params)


def build_person_debtors_query(
    dialect: Dialect, person_keys: Sequence[str], schema: str | None = None
) -> Query:
    """Every debtor row that carries one of ``person_keys``. Parameters: the keys."""
    sql = (
        "SELECT d.debtor_id, d.person_key\n"
        f"FROM {dialect.table('debtors', 'd', schema)}\n"
        f"WHERE d.person_key IN ({placeholders(len(person_keys))})"
    )
    return Query(sql, tuple(person_keys))


def build_contacts_query(
    dialect: Dialect,
    debtor_ids: Sequence,
    date_from: date,
    date_to: date,
    schema: str | None = None,
) -> Query:
    """Outreach attempts for ``debtor_ids`` in ``[date_from, date_to]`` (inclusive).

    Parameters, in order: lower bound, upper (exclusive) bound, then the debtor ids.
    """
    sql = (
        "SELECT e.debtor_id, e.occurred_at AS contact_date, e.result_id,\n"
        "       c.label AS result_label, c.answered, c.right_party\n"
        f"FROM {dialect.table('contact_events', 'e', schema)}\n"
        f"INNER JOIN {dialect.table('contact_results', 'c', schema)} "
        "ON c.result_id = e.result_id\n"
        "WHERE e.occurred_at >= ? AND e.occurred_at < ?\n"
        f"  AND e.debtor_id IN ({placeholders(len(debtor_ids))})"
    )
    params = (dialect.bind_date(date_from), dialect.bind_date(next_day(date_to)), *debtor_ids)
    return Query(sql, params)


def build_breaks_query(
    dialect: Dialect,
    settings: Settings,
    debtor_ids: Sequence,
    schema: str | None = None,
) -> Query:
    """First-installment agreements of ``debtor_ids`` that ended broken (any date).

    Parameters, in order: one per broken status label, then the debtor ids.
    """
    t = dialect.table
    labels = placeholders(len(settings.broken_status_labels))
    sql = f"""WITH ranked AS (
    SELECT r.debtor_id, r.agreement_id, r.portfolio_id, r.status_id, r.issued_at,
           {LATEST.format(a="r")}
    FROM {t("agreements", "r", schema)}
    WHERE r.installment_no = 0
      AND r.status_id IN (
          SELECT g.status_id FROM {t("agreement_statuses", "g", schema)}
          WHERE g.label IN ({labels}))
      AND r.debtor_id IN ({placeholders(len(debtor_ids))})
)
SELECT b.debtor_id, b.agreement_id, b.portfolio_id, b.status_id, st.label AS status_label,
       b.issued_at AS agreement_date
FROM ranked AS b
LEFT JOIN {t("agreement_statuses", "st", schema)} ON st.status_id = b.status_id
WHERE b.rn = 1"""
    return Query(sql, (*settings.broken_status_labels, *debtor_ids))


def build_status_query(dialect: Dialect, schema: str | None = None) -> Query:
    """The status catalog, used by the schema check to confirm configured labels exist."""
    sql = (
        "SELECT s.status_id, s.label\n"
        f"FROM {dialect.table('agreement_statuses', 's', schema)}\n"
        "ORDER BY s.status_id"
    )
    return Query(sql, ())


# --------------------------------------------------------------------------------------------
# Admission: a second, independent route to the numbers the extraction produced
# --------------------------------------------------------------------------------------------
#
# These queries answer the same questions as the extraction by a different route, so that a
# disagreement means something is wrong with the data or with the extraction. They avoid what the
# extraction uses: no ROW_NUMBER (the latest row is the one with no newer row), no LEFT JOIN to a
# reference table (a lookup is an EXISTS, so a duplicate key cannot multiply a row), and the
# outcome rule is written again in SQL. Where the two routes differ, the cause is nearly always a
# duplicate key in a reference table, a status or agent that the join resolves twice, or a date
# the two rules read differently.


def _newer(alias: str, other: str) -> str:
    """``other`` is a later write of the same installment than ``alias``."""
    return (
        f"({other}.issued_at > {alias}.issued_at "
        f"OR ({other}.issued_at = {alias}.issued_at AND {other}.row_id > {alias}.row_id))"
    )


def build_reconcile_slips_query(
    dialect: Dialect,
    settings: Settings,
    date_from: date,
    date_to: date,
    as_of: date,
    schema: str | None = None,
) -> Query:
    """Counts and sums of the slips by the independent route, one row per measurement.

    Columns: ``kind``, ``name``, ``n`` (rows) and ``total`` (sum of the first-installment amount).

    * ``funnel``: rows left after each filter, in the order the extraction applies them
      (``window``, ``status``, ``agents``) and the slips left once re-writes collapse
      (``latest``),
    * ``bucket``: the slips in each outcome bucket, derived in SQL against ``as_of``,
    * ``debtor_orphans`` / ``unknown_agent``: slips whose debtor or agent has no row,
    * ``agreement``: the latest row of every installment of the agreements that have a slip,
      which is what the whole-agreement value sums,
    * ``first_installments`` / ``unreadable_due``: every first-installment row of the table, and
      those whose due date is NULL or does not read as a date. The window filter cannot see such
      a row, so it would drop out of every window without a trace.

    Parameters, in order: lower due bound, upper (exclusive) due bound, one per generated status
    label, two per excluded agent pattern (login, display name), then the reference date twice.
    """
    t = dialect.table
    day = dialect.day
    labels = placeholders(len(settings.generated_status_labels))
    params: list = [
        dialect.bind_date(date_from),
        dialect.bind_date(next_day(date_to)),
        *settings.generated_status_labels,
    ]

    agent_tests: list[str] = []
    agent_params: list[str] = []
    for pattern in settings.excluded_agent_patterns:
        for column in ("login", "display_name"):
            agent_tests.append(f"COALESCE(LOWER(LTRIM(RTRIM(u.{column}))), '') LIKE ?")
            agent_params.append(pattern.lower())
    agent_filter = ""
    if agent_tests:
        agent_filter = (
            "\n    AND NOT EXISTS (\n"
            f"        SELECT 1 FROM {t('agents', 'u', schema)}\n"
            f"        WHERE u.agent_id = w.agent_id AND ({' OR '.join(agent_tests)}))"
        )
    params += agent_params

    on_time = (
        f"l.paid_amount > 0 AND {day('l.paid_at')} <= ? "
        f"AND {day('l.paid_at')} <= {dialect.plus_days(day('l.due_at'), settings.grace_days)}"
    )
    params += [dialect.bind_date(as_of), dialect.bind_date(as_of)]

    sql = f"""WITH windowed AS (
    SELECT w.row_id, w.agreement_id, w.portfolio_id, w.debtor_id, w.status_id, w.agent_id,
           w.amount, w.issued_at, w.due_at, w.paid_amount, w.paid_at
    FROM {t("agreements", "w", schema)}
    WHERE w.installment_no = 0 AND w.due_at >= ? AND w.due_at < ?
),
in_scope AS (
    SELECT * FROM windowed AS w
    WHERE EXISTS (
        SELECT 1 FROM {t("agreement_statuses", "g", schema)}
        WHERE g.status_id = w.status_id AND g.label IN ({labels}))
),
kept AS (
    SELECT * FROM in_scope AS w
    WHERE 1 = 1{agent_filter}
),
latest AS (
    SELECT k.* FROM kept AS k
    WHERE NOT EXISTS (
        SELECT 1 FROM kept AS o
        WHERE o.agreement_id = k.agreement_id AND o.portfolio_id = k.portfolio_id
          AND {_newer("k", "o")})
),
classified AS (
    SELECT l.*,
           CASE WHEN {on_time} THEN 'paid_on_time'
                WHEN {day("l.due_at")} < ? THEN 'overdue_unpaid'
                ELSE 'to_mature' END AS bucket
    FROM latest AS l
)
SELECT 'funnel' AS kind, 'window' AS name, COUNT(*) AS n, SUM(amount) AS total FROM windowed
UNION ALL SELECT 'funnel', 'status', COUNT(*), SUM(amount) FROM in_scope
UNION ALL SELECT 'funnel', 'agents', COUNT(*), SUM(amount) FROM kept
UNION ALL SELECT 'funnel', 'latest', COUNT(*), SUM(amount) FROM latest
UNION ALL SELECT 'bucket', bucket, COUNT(*), SUM(amount) FROM classified GROUP BY bucket
UNION ALL SELECT 'debtor_orphans', 'debtors', COUNT(*), SUM(l.amount)
    FROM latest AS l
    WHERE NOT EXISTS (
        SELECT 1 FROM {t("debtors", "d", schema)} WHERE d.debtor_id = l.debtor_id)
UNION ALL SELECT 'unknown_agent', 'agents', COUNT(*), SUM(l.amount)
    FROM latest AS l
    WHERE NOT EXISTS (
        SELECT 1 FROM {t("agents", "a", schema)} WHERE a.agent_id = l.agent_id)
UNION ALL SELECT 'first_installments', 'agreements', COUNT(*), SUM(z.amount)
    FROM {t("agreements", "z", schema)}
    WHERE z.installment_no = 0
UNION ALL SELECT 'unreadable_due', 'agreements', COUNT(*), SUM(z.amount)
    FROM {t("agreements", "z", schema)}
    WHERE z.installment_no = 0 AND {dialect.unreadable_day("z.due_at")}
UNION ALL SELECT 'agreement', 'installments', COUNT(*), SUM(x.amount)
    FROM {t("agreements", "x", schema)}
    WHERE EXISTS (
        SELECT 1 FROM latest AS l
        WHERE l.agreement_id = x.agreement_id AND l.portfolio_id = x.portfolio_id)
      AND NOT EXISTS (
        SELECT 1 FROM {t("agreements", "y", schema)}
        WHERE y.agreement_id = x.agreement_id AND y.portfolio_id = x.portfolio_id
          AND y.installment_no = x.installment_no AND {_newer("x", "y")})"""
    return Query(sql, tuple(params))


def build_reconcile_contacts_query(
    dialect: Dialect,
    debtor_ids: Sequence,
    date_from: date,
    date_to: date,
    schema: str | None = None,
) -> Query:
    """Outreach events of ``debtor_ids`` in the window, and how many have no result in the catalog.

    Columns: ``name`` (``events`` or ``without_result``) and ``n``. The extraction joins the
    catalog, so an event without a result would vanish; this counts them instead.

    Parameters, in order, twice over: lower bound, upper (exclusive) bound, then the debtor ids.
    """
    t = dialect.table
    ids = placeholders(len(debtor_ids))
    window = "e.occurred_at >= ? AND e.occurred_at < ?"
    sql = (
        "SELECT 'events' AS name, COUNT(*) AS n\n"
        f"FROM {t('contact_events', 'e', schema)}\n"
        f"WHERE {window} AND e.debtor_id IN ({ids})\n"
        "UNION ALL\n"
        "SELECT 'without_result', COUNT(*)\n"
        f"FROM {t('contact_events', 'e', schema)}\n"
        f"WHERE {window} AND e.debtor_id IN ({ids})\n"
        "  AND NOT EXISTS (\n"
        f"      SELECT 1 FROM {t('contact_results', 'c', schema)} WHERE c.result_id = e.result_id)"
    )
    bounds = (dialect.bind_date(date_from), dialect.bind_date(next_day(date_to)))
    return Query(sql, (*bounds, *debtor_ids, *bounds, *debtor_ids))


def build_reconcile_breaks_query(
    dialect: Dialect,
    settings: Settings,
    debtor_ids: Sequence,
    schema: str | None = None,
) -> Query:
    """Earlier broken first-installment agreements of ``debtor_ids``, counted without a join.

    Column: ``n``. Parameters, in order: one per broken status label, then the debtor ids.
    """
    t = dialect.table
    labels = placeholders(len(settings.broken_status_labels))
    sql = f"""WITH broken AS (
    SELECT r.row_id, r.agreement_id, r.portfolio_id, r.issued_at
    FROM {t("agreements", "r", schema)}
    WHERE r.installment_no = 0
      AND EXISTS (
          SELECT 1 FROM {t("agreement_statuses", "g", schema)}
          WHERE g.status_id = r.status_id AND g.label IN ({labels}))
      AND r.debtor_id IN ({placeholders(len(debtor_ids))})
)
SELECT COUNT(*) AS n
FROM broken AS b
WHERE NOT EXISTS (
    SELECT 1 FROM broken AS o
    WHERE o.agreement_id = b.agreement_id AND o.portfolio_id = b.portfolio_id
      AND {_newer("b", "o")})"""
    return Query(sql, (*settings.broken_status_labels, *debtor_ids))
