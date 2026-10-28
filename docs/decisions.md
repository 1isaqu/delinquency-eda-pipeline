# Key decisions

Each entry says what was decided and why, and some say what it costs.

## 1. Treat the "overdue unpaid" bucket as censored, and cut by maturity

Decision: A slip is only counted as lost once its grace period is over. Rates are computed on
the *mature* cohort; the raw bucket is split into *still open* and *definitive*.

Why: Overdue starts the day after the due date, but paid-on-time runs `grace_days` further, so
the two clocks disagree for the most recent due dates. A bucket defined by "overdue and not paid on
time" holds slips that can still be paid on time, so it inflates only for a period with due dates
in the last `grace_days` days: month to date, plus the previous month during the first `grace_days`
days of a month. Older periods agree exactly. The same pattern applies to any indicator that
compares today with an outcome that has a grace period.

Cost: One more concept to explain, and a smaller denominator.

## 2. Maturity is a function of the due date only

Decision: `is_mature` is `days_overdue > grace_days`. It never looks at the payment.

Why: Selecting "paid, or unpaid and old" picks rows by what happened to them. Any rate on that
set is biased upward for payers.

## 3. Derive outcomes in Python from a reference date the caller sets

Decision: The database returns raw facts. Buckets, days overdue and maturity are computed in
`derive.py` against `--as-of`, never with the database server's clock. `period`, `base` and `card`
take the flag and use today's local date without it.

Why: Reproducibility (same date, same file, byte for byte), replay (the drift stage re-derives
the past), and one unit-tested place for the rule. With the server clock, each re-run uses a
different date, so numbers from two runs cannot be reconciled.

## 4. Business filters run before the "latest row wins" ranking

Decision: In the extraction SQL the due-date window, status set and excluded agents sit inside
the derived table that holds `ROW_NUMBER()`, not outside it.

Why: An installment can be written twice. If the ranking ran first, a later re-write with a
status or date outside the filter would win the rank and hide the valid row, silently dropping an
agreement. A test pins this with a row built to fail if the order is swapped.

## 5. Statuses by label, keys by composite

Decision: Status sets are configured as labels and resolved through the catalog. Agreement
keys are `product_line|agreement_id|portfolio_id`.

Why: Status numbers differ between databases and across catalog reloads; a test shifts every
id to prove nothing depends on them. A row id repeats across product lines (each database has its
own sequence), so on its own it is not a key.

## 6. One portfolio label per agreement, through a top-one lookup

Decision: The label hangs off the debt and an agreement can link several debts, so the SQL takes
one label (lowest debt id that has one) with `OUTER APPLY (SELECT TOP 1 ...)` on SQL Server and a
scalar subquery on SQLite, instead of a join.

Why: A join multiplies agreements by their debts and inflates every count and sum.

## 7. A zero balance is a value; a missing debt is another

Decision: `original_debt = 0` and "no linked debt" are separate states, flagged separately, and
never imputed. The installment-to-debt ratio is `NaN` where undefined.

Why: Testing `if original_debt:` treats zero as falsy and files it with the missing rows under
the wrong label. A zero balance can carry signal of its own (the EDA report shows its share per
product line), so imputing a number would invent one.

## 8. Cleaning only adds

Decision: Rules flag, derive and list. They never delete a row, impute a cell or drop a column,
and the stage reports exactly that: zero rows removed, zero cells imputed, zero columns removed.

Why: Deleting or imputing rows throws away information that a flag keeps, and a flag can be
undone. Everything additive is reversible, and the information-loss summary makes it checkable.
Outliers are kept: winsorizing would erase the large slips, which weigh the most in any value
total.

## 9. Separate concentration from discrimination

Decision: Every factor gets a concentration table (over the whole unpaid bucket) and a
discrimination table (over the mature cohort, with lift and a sample-size flag).

Why: A big group dominates concentration by size alone. Only the rate against a baseline says
whether a factor separates payers from non-payers, and a rate on a handful of slips is noise.

## 10. Follow-up is explanatory, never a feature

Decision: Contact counts between agreement and due date appear in the analysis and are excluded
from the model.

Why: If outreach goes to people who are already slipping, more follow-up with more non-payment
is reverse causality. Using it to predict non-payment confuses cause and effect.

## 11. Group by person, validate by time

Decision: Cross-validation is grouped by `group_key` (the opaque person key); a temporal
hold-out is the second check.

Why: Rows of one person are not independent. A random split leaks people across train and test
and inflates the score. The grouped folds catch leakage between people, and the time hold-out
catches drift.

## 12. Odds ratios as coefficient differences against a stated reference

Decision: Term-band odds ratios are `exp(coef[band] - coef[reference])`, where the reference is
the shortest band that has rows, and the report names it.

Why: With all bands one-hot encoded, a single coefficient is not an odds ratio against anything
in particular. Differences are, and they stay meaningful when the nominal first band is empty.

## 13. Observational, and said so

Decision: Every analysis report and the model report state that results are associations. The
model report names the design that would show cause (a randomized term).

Why: People who ask for a long term may already be the people who cannot pay. The data has no
field for the reason a slip was not paid.

## 14. A thin adapter, read-only twice over

Decision: The pipeline depends on a small `Database` protocol (`query`, `table_columns`,
`close`). A SQLite adapter serves the demo and tests; a SQL Server adapter reads `DEA_DB_*`
environment variables. Every statement passes a guard that refuses anything but a single `SELECT`
or `WITH ... SELECT`; the SQLite connection is also opened read-only.

Why: No schema, host or credential is baked into the code, and the same SQL text runs against
both engines except for three documented dialect differences (table hints, the top-one lookup, date
binding). The guard catches mistakes in the pipeline's own SQL; access control still has to come
from a read-only login.

## 15. No data and no fitted artifacts in the repository

Decision: Tests use inline invented rows or a seeded generator. Datasets, reports and the
cleaned tables are written under gitignored paths. Models are fitted in memory and never saved.

Why: The repository holds code only. A dataset file carries whatever identifiers your database
supplies, so the default location is ignored and the README says to keep it out of version control.

## 16. The debt balance is the balance as of the agreement

Decision: The contract defines `debts.balance` as what the debt carried on the agreement date.
The pipeline cannot check it: a value does not say when it was read.

Why: Original debt and the zero-balance flag are controls in the term model. A balance read after
the agreement can already reflect payments, which would carry the outcome into the controls and
make them look more predictive than they are.

Cost: If you only keep a current balance, you need a snapshot view, or you read the odds ratios
with those two controls marked as leakage-suspect.

## 17. Data enters through a contract, or not at all

Decision: `base` judges every build against the entry contract (`docs/data-contract.md`) and
writes the dataset only if every rule passes. There is no option to let a failing build through,
and `Dataset.save` refuses a dataset that has no passed admission record.

Why: A report that says a column is dirty does not stop the next stage from reading it, and a
number built on it carries no warning. A finding that stops nothing is read after the fact, if at
all. Putting the check at the one door every row passes through makes "this dataset met the
contract" true of every file the other stages can read. An override would turn the contract into
advice.

Cost: A first run against a real database can be rejected, and the repair belongs in the view
that maps the database onto the contract. A rule that is too strict for a database has no
workaround except changing the data, so the rules are limited to what the contract already says
(a column is not null, a flag is 0 or 1, a payment is NULL or positive) and to what would change a
count without any error (a duplicate key, a row the window cannot see).

## 18. Check the extraction against a second route, not against itself

Decision: After the extraction, a second set of queries computes the same counts and sums by a
different route: the latest write of an installment is the one with no newer write (no
`ROW_NUMBER`), every lookup is an `EXISTS` (no join that can multiply rows), and the outcome rule
is written again in SQL. Per product line the two must agree on slips, buckets, first-installment
value, whole-agreement value, installment count, contact events and earlier breaks.

Why: A check on the rows cannot see a row that the query multiplied, dropped or reclassified. A
duplicate debtor id doubles every slip that joins to it, and every later count is still
consistent with the doubled rows. Only another route to the number shows it. The routes share the
database, the window and the reference date, so agreement says the extraction did what the
contract says with the data, not that the data is true.

Cost: One aggregate query per product line and two per chunk of debtor ids, and a second copy of
the outcome rule that has to follow `derive.py`. The tests compare the two on the test world and
on the demo databases, and change the dataset one number at a time to prove each comparison can
fail.

## 19. Seal the file, and keep the verdict inside it

Decision: The dataset file carries the admission record and a seal (a SHA-256 of everything else
in the file, and the row count of each table). Every offline stage verifies the seal before it
reads a row. The format version moved to 2, so a file written before the contract is refused with
the instruction to rebuild it.

Why: The offline stages run where the database is not reachable, so the file is the only evidence
that the data was admitted. Without a seal, a half-copied or hand-edited file produces numbers
that look like any others.

Cost: The seal detects accidents, not tampering: whoever can edit the file can recompute it. Old
files have to be rebuilt with `base`.
