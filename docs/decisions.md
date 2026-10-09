# Key decisions

Each entry says what was decided and why.

## 1. Treat the "overdue unpaid" bucket as censored, and cut by maturity

**Decision.** A slip is only counted as lost once its grace period is over. Rates are computed on
the *mature* cohort; the raw bucket is split into *still open* and *definitive*.

**Why.** Overdue starts the day after the due date, but paid-on-time runs `grace_days` further, so
the two clocks disagree for the most recent due dates. A bucket defined by "overdue and not paid on
time" holds slips that can still be paid on time, and it inflates only for the open period.
Closed periods agree exactly. The same pattern applies to any indicator that compares today with an
outcome that has a grace period.

**Cost.** One more concept to explain, and a smaller denominator.

## 2. Maturity is a function of the due date only

**Decision.** `is_mature` is `days_overdue > grace_days`. It never looks at the payment.

**Why.** Selecting "paid, or unpaid and old" picks rows by what happened to them. Any rate on that
set is biased upward for payers.

## 3. Derive outcomes in Python from a reference date the caller sets

**Decision.** The database returns raw facts. Buckets, days overdue and maturity are computed in
`derive.py` against `--as-of`, never with the database server's clock. `period`, `base` and `card`
take the flag and use today's local date without it.

**Why.** Reproducibility (same date, same file, byte for byte), replay (the drift stage re-derives
the past), and one unit-tested place for the rule. With the server clock, each re-run uses a
different date, so numbers from two runs cannot be reconciled.

## 4. Business filters run before the "latest row wins" ranking

**Decision.** In the extraction SQL the due-date window, status set and excluded agents sit inside
the derived table that holds `ROW_NUMBER()`, not outside it.

**Why.** An installment can be written twice. If the ranking ran first, a later re-write with a
status or date outside the filter would win the rank and hide the valid row, silently dropping an
agreement. A test pins this with a row built to fail if the order is swapped.

## 5. Statuses by label, keys by composite

**Decision.** Status sets are configured as labels and resolved through the catalog. Agreement
keys are `product_line|agreement_id|portfolio_id`.

**Why.** Status numbers differ between databases and across catalog reloads; a test shifts every
id to prove nothing depends on them. A row id repeats across product lines (each database has its
own sequence), so on its own it is not a key.

## 6. One portfolio label per agreement, through a top-one lookup

**Decision.** The label hangs off the debt and an agreement can link several debts, so the SQL takes
one label (lowest debt id that has one) with `OUTER APPLY (SELECT TOP 1 ...)` on SQL Server and a
scalar subquery on SQLite, instead of a join.

**Why.** A join multiplies agreements by their debts and inflates every count and sum.

## 7. A zero balance is a value; a missing debt is another

**Decision.** `original_debt = 0` and "no linked debt" are separate states, flagged separately, and
never imputed. The installment-to-debt ratio is `NaN` where undefined.

**Why.** Testing `if original_debt:` treats zero as falsy and files it with the missing rows under
the wrong label. A zero balance can carry signal of its own (the EDA report shows its share per
product line), so imputing a number would invent one.

## 8. Cleaning only adds

**Decision.** Rules flag, derive and list. They never delete a row, impute a cell or drop a column,
and the stage reports exactly that: zero rows removed, zero cells imputed, zero columns removed.

**Why.** Deleting or imputing rows throws away information that a flag keeps, and a flag can be
undone. Everything additive is reversible, and the information-loss summary makes it checkable.
Outliers are kept: winsorizing would erase the large slips, which weigh the most in any value
total.

## 9. Separate concentration from discrimination

**Decision.** Every factor gets a concentration table (over the whole unpaid bucket) and a
discrimination table (over the mature cohort, with lift and a sample-size flag).

**Why.** A big group dominates concentration by size alone. Only the rate against a baseline says
whether a factor separates payers from non-payers, and a rate on a handful of slips is noise.

## 10. Follow-up is explanatory, never a feature

**Decision.** Contact counts between agreement and due date appear in the analysis and are excluded
from the model.

**Why.** If outreach goes to people who are already slipping, more follow-up with more non-payment
is reverse causality. Using it to predict non-payment confuses cause and effect.

## 11. Group by person, validate by time

**Decision.** Cross-validation is grouped by `group_key` (the opaque person key); a temporal
hold-out is the second check.

**Why.** Rows of one person are not independent. A random split leaks people across train and test
and inflates the score. The grouped folds catch leakage between people, and the time hold-out
catches drift.

## 12. Odds ratios as coefficient differences against a stated reference

**Decision.** Term-band odds ratios are `exp(coef[band] - coef[reference])`, where the reference is
the shortest band that has rows, and the report names it.

**Why.** With all bands one-hot encoded, a single coefficient is not an odds ratio against anything
in particular. Differences are, and they stay meaningful when the nominal first band is empty.

## 13. Observational, and said so

**Decision.** Every analysis report and the model report state that results are associations. The
model report names the design that would show cause (a randomized term).

**Why.** People who ask for a long term may already be the people who cannot pay. The data has no
field for the reason a slip was not paid.

## 14. A thin adapter, read-only twice over

**Decision.** The pipeline depends on a small `Database` protocol (`query`, `table_columns`,
`close`). A SQLite adapter serves the demo and tests; a SQL Server adapter reads `DEA_DB_*`
environment variables. Every statement passes a guard that refuses anything but a single `SELECT`
or `WITH ... SELECT`; the SQLite connection is also opened read-only.

**Why.** No schema, host or credential is baked into the code, and the same SQL text runs against
both engines except for three documented dialect differences (table hints, the top-one lookup, date
binding). The guard catches mistakes in the pipeline's own SQL; access control still has to come
from a read-only login.

## 15. No data and no fitted artifacts in the repository

**Decision.** Tests use inline invented rows or a seeded generator. Datasets, reports and the
cleaned tables are written under gitignored paths. Models are fitted in memory and never saved.

**Why.** The repository holds code only. A dataset file carries whatever identifiers your database
supplies, so the default location is ignored and the README says to keep it out of version control.

## 16. The debt balance is the balance as of the agreement

**Decision.** The contract defines `debts.balance` as what the debt carried on the agreement date.
The pipeline cannot check it: a value does not say when it was read.

**Why.** Original debt and the zero-balance flag are controls in the term model. A balance read after
the agreement can already reflect payments, which would carry the outcome into the controls and
make them look more predictive than they are.

**Cost.** If you only keep a current balance, you need a snapshot view, or you read the odds ratios
with those two controls marked as leakage-suspect.
