# Data contract

The pipeline never touches your real table names. It reads nine tables with fixed names and
columns. You map your own database onto them, most simply with views, and point the adapter at
the schema that holds them. This page is the whole contract, including the rules a build
must satisfy before its data enters (see *The entry contract*).

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

"Required" columns are read by the pipeline. Columns marked *optional* are checked by the `schema`
stage but not read, so a missing one is only reported.

### `agreements`: one row per installment of an agreement

| Column | Type | Null | Meaning |
|---|---|---|---|
| `row_id` | integer | no | Unique per row. Breaks ties when an installment was written twice. |
| `agreement_id` | integer | no | Agreement number. Not unique on its own: pair it with `portfolio_id`. |
| `portfolio_id` | integer | no | Portfolio the agreement belongs to. |
| `debtor_id` | integer | no | The debtor row that signed. Links to `debtors`. |
| `installment_no` | integer | no | `0` is the first installment (the down payment). The pipeline analyzes first installments. |
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

### `agreement_statuses`: status catalog

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

An agreement whose `agent_id` has no row in `agents` is still extracted, with a null name and
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
| `balance` | numeric | yes | Balance as of the agreement date: what the debt carried when the term was agreed. A balance of zero is a real value and is kept apart from a missing link. |
| `portfolio_id`, `contract_no`, `due_at`, `days_overdue` | various | yes | Optional, not read. |

The date matters. Original debt (the sum of the linked balances) and the zero-balance flag are
controls in the payment-term model, which may only use what was known when the term was agreed. A
balance read at extraction time can already reflect payments made after the agreement, so it can
carry the outcome into those controls. The pipeline cannot detect that from the data. If your
database keeps only a current balance, map `balance` to a value that was fixed at the agreement
(a snapshot, or the amount the debt carried when it was agreed). Otherwise treat the original-debt
and zero-balance controls of stage 10 as leakage-suspect when you read its odds ratios.

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
| `debtor_id` | integer | no | Debtor the attempt was about. If your table names it differently (for example `debt_owner_id`), expose it as `debtor_id` in the view. |
| `result_id` | integer | no | Links to `contact_results`. |
| `occurred_at` | datetime | no | When it happened. |
| `agent_id` | text or integer | yes | Optional, not read. |

### `contact_results`: result catalog

| Column | Type | Meaning |
|---|---|---|
| `result_id` | integer | Key. |
| `label` | text | Shown in reports. |
| `answered` | bit / 0-1 | Someone picked up. |
| `right_party` | bit / 0-1 | The right person was reached. |

A right-party contact (CPC) is `answered = 1 AND right_party = 1`. Requiring both keeps the
funnel monotonic: attempts >= answered >= CPC. A flag that fires without an answer would let
auto-dispatched messages and bounces count as contact.

## The entry contract

The tables above say what the database must look like. The entry contract says what a build must
satisfy before its data is used. `delinquency-eda base` judges every build against the rules
below and writes the dataset only if all of them pass. A build that breaks one writes no dataset
and exits with code 1. The `admission.md` report says which rule failed, how many rows broke it,
and where to fix it. No option lets a failing build through. If a rule is wrong for your
database, change the view that maps it onto the contract (or the settings) so that it holds; the
contract is the agreement between the database and the pipeline.

| Rule | Scope | What must hold |
|---|---|---|
| `source.tables` | source | Every table the base build reads exists and has all its required columns. |
| `source.labels` | source | Every status label named in the settings exists in agreement_statuses. |
| `source.types` | source | The declared type of every required column is in the family the contract names. |
| `row.required` | rows | A column the contract marks not null is never null in an extracted row. |
| `row.readable` | rows | Dates read as ISO dates, amounts as numbers, counts as whole numbers, flags as 0 or 1. |
| `row.payment` | rows | paid_amount is NULL or positive, and paid_amount and paid_at are both present or both NULL. |
| `row.due_date` | source | Every first-installment row of agreements has a due date that reads as a date. |
| `key.grain` | dataset | There is one slip per product line, agreement and portfolio; row ids and debtor ids are unique within a product line. |
| `base.domain` | dataset | Every slip has a known bucket, a generated status label, a configured product line and a due date inside the requested window. |
| `ref.debtor` | source | The debtor_id of every extracted slip has a row in debtors. |
| `ref.contact_result` | source | The result_id of every contact event in the window has a row in contact_results. |
| `recon.slips` | source | The number of slips per product line equals the count of a second, independent query. |
| `recon.outcome` | source | The slips in each bucket, per product line, equal the count of the outcome rule re-derived in SQL. |
| `recon.value` | source | The first-installment value and the whole-agreement value per product line equal the sums of the independent queries. |
| `recon.history` | source | The contact events and earlier broken agreements pulled in per product line equal the counts of the independent queries. |

*Source* rules are judged against the live database, *rows* rules on every row as the driver
returned it, and *dataset* rules on the assembled dataset. They run in that order, and a source
that fails its own rules is not queried for rows, because the queries would not mean what they
say. When a row breaks a rule, the rules that would only repeat the problem as a count difference
are reported as skipped.

### Declared types

The `schema` stage and the `source.types` rule compare the declared type of each required column
with the family its kind needs. An engine that declares no type, as SQLite may, is not judged on
type; its values are checked when they are read.

| Kind | Declared type accepted | Columns |
|---|---|---|
| `integer` | any numeric type (`int`, `bigint`, `decimal`, ...) | ids, counts, `installment_no` |
| `number` | any numeric type | amounts and balances |
| `text` | `char`, `varchar`, `nvarchar`, `text` and the like | labels, logins, `person_key` |
| `id` | numeric or text | `agent_id` |
| `date` | `date`, `datetime`, `datetime2`, `smalldatetime`, `datetimeoffset`, `timestamp`, or text that holds ISO dates | `issued_at`, `due_at`, `paid_at`, `occurred_at` |
| `flag` | a numeric type holding 0 or 1; text is refused, because `'N'` reads as true | `answered`, `right_party` |

### The second route

A build is also compared with a second, independent route to the same numbers
(`recon.*`, `ref.*` and `row.due_date`). The second route is a different shape of SQL: the latest
write of an installment is the one with no newer write (an anti-join, not `ROW_NUMBER`), every
lookup is an `EXISTS` (not a join that could multiply rows), and the outcome rule is written
again in SQL against the same `--as-of` date. Per product line it must agree with the dataset on
the number of slips, the number in each bucket, the first-installment and whole-agreement value
(to the cent), the installment count, the contact events and the earlier breaks. A duplicate key
in `agents`, `debtors`, `agreement_statuses` or `contact_results` makes the extraction's joins
return a row twice; the second route does not, so the difference shows. The same query also
counts where the rows of the window went (outside the generated statuses, excluded agents, older
writes), which the report prints as the filter ledger.

### What the report and the dataset record

`admission.md` and the admission record stored in the dataset hold counts and column names, never
a key, an id or a value from a row. The record lists each rule with its result, the filter
ledger, and a fingerprint (SHA-256 of the text, ignoring the length of `IN` lists) of every query
the build ran, so a later build can show whether the extraction changed. The dataset file also
carries a seal: a SHA-256 of everything else in the file and the row count of each table. Every
offline stage checks the seal before it reads a row and refuses a file that was cut, edited or
not written by `base`. The seal catches accidents. Someone who can edit the file can also
recompute the seal, so it is not protection against tampering.

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
against a live SQL Server. The same holds for the second route's T-SQL, which also uses `TRY_CAST`
(SQL Server 2012 or later). Run `schema` first and expect to adjust types in your views.
