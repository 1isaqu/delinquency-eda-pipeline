# Data contract

The pipeline never touches your real table names. It reads nine tables with fixed names and
columns. You map your own database onto them, most simply with views, and point the adapter at
the schema that holds them. This page is the whole contract.

Names are generic on purpose. `docs/glossary.md` defines the vocabulary.

## Conventions

| Rule | Detail |
|---|---|
| Read only | The pipeline only issues `SELECT` or `WITH ... SELECT`. A guard refuses anything else, but the real protection is a login that only holds read permission. Use one. |
| One database per product line | A *product line* is one database with this schema (for example `auto` and `consumer`). The same numeric id can exist in two product lines, so every key inside the pipeline is prefixed with the product line. |
| Dates | `date` or `datetime`. Only the calendar day is used. SQLite databases may store ISO text (`YYYY-MM-DD` or `YYYY-MM-DD hh:mm:ss`). |
| Money | Any numeric type. One currency per database. |
| Nulls | Allowed where the table below says so. A missing payment is `NULL`, never `0`. |
| Statuses | Resolved by label, never by number. The numbers can differ between databases. |
| Identifiers of people | Never needed. The pipeline links a person across rows through an opaque `person_key` that you supply. Do not put a national id, name, phone number or e-mail in it. |

## Tables

"Required" columns are read by the pipeline. The shared collections contract has a few more
columns per table; those are marked *optional* and are checked by the `schema` stage but not read.

### `agreements`: one row per installment of an agreement

| Column | Type | Null | Meaning |
|---|---|---|---|
| `row_id` | integer | no | Unique per row. Breaks ties when an installment was written twice. |
| `agreement_id` | integer | no | Agreement number. Not unique on its own: pair it with `portfolio_id`. |
| `portfolio_id` | integer | no | Portfolio the agreement belongs to. |
| `debtor_id` | integer | no | The debtor row that signed. Links to `debtors`. |
| `installment_no` | integer | no | `0` is the first installment (the down payment). The pipeline analyses first installments. |
| `amount` | numeric | no | Amount due for this installment. |
| `plan_installments` | integer | yes | Installments the plan declared. |
| `paid_amount` | numeric | yes | Money received. `NULL` when nothing was paid. |
| `issued_at` | date | no | When the agreement was made. |
| `due_at` | date | no | When this installment falls due. |
| `paid_at` | date | yes | When the money arrived. |
| `status_id` | integer | no | Links to `agreement_statuses`. |
| `agent_id` | text or integer | no | Links to `agents`. |

An installment can be written twice by mistake. The latest row wins: for each
`(agreement_id, portfolio_id, installment_no)` the pipeline keeps the row with the greatest
`issued_at`, then the greatest `row_id`. Business filters (due-date window, status, excluded agents)
run *before* that ranking, so a re-write that falls outside a filter cannot hide the valid row.

### `agreement_statuses`: status catalogue

| Column | Type | Meaning |
|---|---|---|
| `status_id` | integer | Key. |
| `label` | text | Matched against the labels in the settings. The `schema` stage checks them exactly, so a label that differs only in case is reported even where your database would match it anyway. |

The default labels are `ACTIVE`, `BROKEN`, `PAID`, `PENDING_EXCEPTION`, `REJECTED`, `AUTO_BROKEN` and
`PAID_MANUAL`. The ids behind them are yours. Settings decide which labels count as *generated*
(the population analyzed) and which as *broken* (history used for repeat-break counts); see
`generated_status_labels`, `broken_status_labels` and `manual_break_label` in the README.

### `agents`: who owns agreements

| Column | Type | Meaning |
|---|---|---|
| `agent_id` | text or integer | Key. |
| `display_name` | text | Label shown in reports. |
| `login` | text | Matched with `display_name` against `excluded_agent_patterns`. |

Use invented or internal ids. Real names are not needed anywhere.

An agreement whose `agent_id` has no row in `agents` is still extracted, with an empty name and
login. The exclusion patterns match on name and login, so they cannot exclude such an agreement;
list every agent, including system accounts, if you rely on them.

### `agreement_debts`: links agreements to debts

| Column | Type | Meaning |
|---|---|---|
| `agreement_id` | integer | Pairs with `portfolio_id`. |
| `portfolio_id` | integer | |
| `debt_id` | integer | Links to `debts`. |

An agreement can cover several debts. Original debt is the sum of their balances, and an agreement
with no link has an *unknown* original debt (not zero).

### `debts`: debt detail

| Column | Type | Null | Meaning |
|---|---|---|---|
| `debt_id` | integer | no | Key. |
| `balance` | numeric | yes | Current balance. A balance of zero is a real value and is kept apart from a missing link. |
| `portfolio_id`, `contract_no`, `due_at`, `days_overdue` | various | yes | Optional, not read. |

### `debt_attributes`: holds the portfolio label

| Column | Type | Null | Meaning |
|---|---|---|---|
| `debt_id` | integer | no | Key. |
| `portfolio_label` | text | yes | Human label such as `Portfolio A-02`. Free text. |

The label hangs off the debt, not the agreement. The pipeline takes one label per agreement (the
lowest `debt_id` that has one) so a join can never multiply rows.

### `debtors`: debtor rows and the opaque person key

| Column | Type | Null | Meaning |
|---|---|---|---|
| `debtor_id` | integer | no | Key, as used in `agreements` and `contact_events`. |
| `person_key` | text | yes | Opaque token: the same person gets the same key on every debtor row, in every product line. |

The key lets the pipeline gather one person's follow-up and earlier breaks across portfolios, and
split train from test by person. Generate it however you like (a salted hash of whatever
identifies a person, a surrogate from a master table). Keep it text; leading zeros matter. Rows
with a `NULL` key are never pooled together.

### `contact_events`: one row per outreach attempt

| Column | Type | Null | Meaning |
|---|---|---|---|
| `event_id` | integer | no | Key. |
| `debtor_id` | integer | no | Debtor the attempt was about. The shared contract calls this `debt_owner_id`; expose it as `debtor_id`. |
| `result_id` | integer | no | Links to `contact_results`. |
| `occurred_at` | datetime | no | When it happened. |
| `agent_id` | text or integer | yes | Optional, not read. |

### `contact_results`: result catalogue

| Column | Type | Meaning |
|---|---|---|
| `result_id` | integer | Key. |
| `label` | text | Shown in reports. |
| `answered` | bit / 0-1 | Someone picked up. |
| `right_party` | bit / 0-1 | The right person was reached. |

A right-party contact (CPC) is `answered = 1 AND right_party = 1`. Requiring both keeps the
funnel monotonic: attempts >= answered >= CPC. A flag that fires without an answer would let
auto-dispatched messages and bounces count as contact.

## Mapping your database

Create views (or a separate schema) with the names above and set `DEA_DB_SCHEMA`. A sketch, with
invented source names:

```sql
CREATE SCHEMA reporting;
GO
CREATE VIEW reporting.agreements AS
SELECT i.id            AS row_id,
       i.contract_no   AS agreement_id,
       i.portfolio_id,
       i.debtor_id,
       i.installment   AS installment_no,
       i.amount,
       i.plan_size     AS plan_installments,
       i.amount_paid   AS paid_amount,
       i.created_on    AS issued_at,
       i.due_on        AS due_at,
       i.paid_on       AS paid_at,
       i.status_code   AS status_id,
       i.owner_user    AS agent_id
FROM   your_schema.installments AS i;
```

Do the same for the other eight tables. Run `delinquency-eda schema` after: it lists every table and
column the pipeline cannot find, the extra columns it ignores, the column types it found, and any
configured status label that is missing from `agreement_statuses`.

## What is and is not exercised against a live database

The extraction SQL is exercised end to end against SQLite (the demo databases and the tests).
For SQL Server the pipeline generates T-SQL (`WITH (NOLOCK)` on every table, `OUTER APPLY
(SELECT TOP 1 ...)` for the portfolio label, bound `date` parameters), and the tests check that SQL
text and the connection-string handling through a fake DB-API connection. It has not been run
against a live SQL Server. Run `schema` first and expect to adjust types in your views.
