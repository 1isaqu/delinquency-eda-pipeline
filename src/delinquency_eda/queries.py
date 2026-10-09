"""SQL builders for the extraction. Pure functions: text and parameters out, no database in.

Rules every query follows (they are the pipeline's standard behavior, not options):

* read only, and every table reference carries the dialect's non-blocking hint,
* re-written installments are collapsed: the latest row per
  ``(agreement_id, portfolio_id, installment_no)`` wins. Business filters (date window, status,
  excluded agents) sit *inside* the ranked derived table, before ``ROW_NUMBER``. Filtering after
  ranking would let a re-write that falls outside the filter hide the valid row.
* an agreement whose agent has no row in ``agents`` is kept, with an empty name and login, so the
  join to ``agents`` never drops a slip,
* the portfolio label is resolved with a single top-one lookup per agreement, so joins never
  multiply rows,
* dates are bound parameters, never ``GETDATE()``: the reference date is explicit, so a run can be
  reproduced.

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
    """The status catalogue, used by the schema check to confirm configured labels exist."""
    sql = (
        "SELECT s.status_id, s.label\n"
        f"FROM {dialect.table('agreement_statuses', 's', schema)}\n"
        "ORDER BY s.status_id"
    )
    return Query(sql, ())
