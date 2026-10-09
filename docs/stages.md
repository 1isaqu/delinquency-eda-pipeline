# Stage reference

Eleven stages, numbered in the order you normally run them. Four read the database (`period`,
`schema`, `base` and, unless told otherwise, `card`). Everything after `base` is offline and reads
the dataset file `base` wrote, so it is repeatable and safe to run on a laptop with no connection.

| # | Command | Reads | Writes (under `--out-dir`) |
|--:|---|---|---|
| 00 | `period` | database | `period.md` |
| 01 | `schema` | database | `schema.md` |
| 02 | `base` | database | the dataset file, `base.md` |
| 03 | `analysis` | dataset | `analysis.md` |
| 04 | `drift` | dataset | `drift.md` |
| 05 | `censoring` | dataset | `censoring.md` |
| 06 | `eda` | dataset | `eda.md` |
| 07 | `diagnose` | dataset | `diagnose.md` |
| 08 | `clean` | dataset | `clean_full.csv`, `clean_analytic.csv`, `clean_meta.json`, `clean.md` |
| 09 | `card` | database, or dataset with `--from-dataset` | `card.md` |
| 10 | `model` | dataset | `model.md` |

Exit codes: `0` success, `1` a stage ran and reported a failed check (only `schema` does this),
`2` usage or configuration problem. Errors print an `error: ...` line (a usage error prints the
usage text before it), never a traceback.

## The outcome every stage shares

A **first-installment slip** is an `installment_no = 0` row. Its outcome at a reference date
(`--as-of`) is one of three buckets, derived in Python from the raw dates and amounts:

| Bucket | Rule |
|---|---|
| `paid_on_time` | money received on or before `due + grace_days`, and not after the reference date |
| `overdue_unpaid` | due before the reference date and not paid on time |
| `to_mature` | not due yet |

A slip is **mature** once the reference date is more than `grace_days` past its due date. From then
on its outcome cannot change. Maturity depends on the due date alone, never on the outcome:
selecting "paid, or unpaid and old" would pick slips by what happened to them and bias every rate
computed on them.

The reference date is an explicit parameter, never the server clock, so the same command with the
same date builds the same dataset byte for byte.

## 00 `period`: which window reproduces the funnel?

Use this stage when a summary card does not say which window it covers: month to date up to today
or up to yesterday, or the whole previous month? It evaluates the funnel
`generated = paid_on_time + overdue_unpaid + to_mature` over candidate windows and ranks them by
closeness to the counts you already have.

- Slips are fetched once over the widest span; every window is evaluated in Python, so adding a
  candidate is free.
- Default candidates: month to date, today only, month to yesterday, previous month, last 7 days,
  last 30 days. Empty and duplicate spans are dropped (on the first of a month, "month to yesterday"
  would be empty). Supply your own with `--window NAME=FROM:TO`.
- `--reference KEY=COUNT` (repeatable; keys `generated`, `to_mature`, `overdue_unpaid`,
  `paid_on_time`) scores each window by the summed absolute difference over the keys you gave.
  A non-zero best distance is expected when your reference was taken earlier: slips paid since then
  have left the bucket, and `drift` explains which.

## 01 `schema`: does the database look like the contract?

Run it first against a new database. For every contract table it reports whether the table exists,
which required columns are missing, which optional ones are, how many extra columns it carries, and
the types of the required columns it found. It also checks that every status label named in the
settings exists in `agreement_statuses`, because statuses are resolved by label. Two readiness
levels: *period/card* (agreements, statuses, agents) and *base* (everything). Exit code `1` when a
line is not ready for `base`.

## 02 `base`: build the dataset

Extracts four tables into one JSON file:

| Table | Content |
|---|---|
| `base` | first-installment slips due in `[--from, --to]`, each with its derived outcome |
| `debtors` | every debtor row that shares a `person_key` with a slip |
| `contacts` | outreach attempts for those debtors between the earliest agreement and the last due date |
| `breaks` | earlier first-installment agreements of those debtors that ended broken |

Rules the SQL follows (details in `docs/decisions.md`): read only; latest row wins per installment
with filters applied before the ranking; excluded-agent patterns applied inside the SQL; one
portfolio label per agreement through a single top-one lookup; bound dates. Whole-agreement value
and installment count are aggregated per agreement from each installment's latest row; original
debt is the sum of the balances of the linked debts, which the contract defines as of the agreement
date. The file holds whatever keys and agent ids your database supplies: keep it out of version
control (the default `data/` path is gitignored).

## 03 `analysis`: which factors separate payers from non-payers?

Two tables per factor, because they answer different questions:

| Table | Population | Question |
|---|---|---|
| Concentration | every overdue-unpaid slip | where does the unpaid value sit? |
| Discrimination | the mature cohort only | does this group fail more or less than the baseline? |

A factor can top the concentration table just by being big. Only discrimination says whether it
separates payers from non-payers, and it uses mature slips because a slip still inside its grace
window can yet be paid on time: counting it as a failure would measure recency. Each group gets a
lift (its unpaid rate over the baseline) and a sample-size flag (`reliable`, `weak`, `not reliable`
by `min_n_reliable` and `min_n_weak`).

Factors: agent, portfolio label, portfolio group, product line, follow-up intensity, earlier broken
agreements (grouped and yes/no), first installment against original debt, first-installment value,
term (days from agreement to first due date), installment count, and follow-up by repeat debtor.
Bands come from the settings. The report also shows the same bucket under three value measures
(first installment, whole agreement, original debt), because a headline figure sums one of them and
they can differ a lot.

Follow-up is descriptive only. If outreach goes to people who are already slipping, more
follow-up going with more non-payment is reverse causality, not evidence that outreach hurts.

## 04 `drift`: which slips moved?

A card shown on one day need not match a re-run made later. This stage re-derives every outcome as
of an earlier `--reference-date` (a payment dated after it did not exist yet; an agreement made
after it did not exist yet) and compares with now: bucket counts and values on both dates, a
transition matrix, the slips that left the unpaid bucket (movers), and those that entered it. If you
pass `--headline-value`, it reports which measure (first installment, whole agreement, original
debt) your figure is closest to, on which of the two dates. Limit: the dataset holds the latest
status of each agreement, so later breaks and re-writes cannot be rewound.

## 05 `censoring`: how much of the bucket is not lost yet?

"Overdue" starts the day after the due date; "paid on time" runs `grace_days` further. A slip due
yesterday is in the overdue-unpaid bucket today while still able to be paid on time. The stage
splits the bucket into *still open* and *definitive*, reports the inflation of the naive count over
the definitive one, and contrasts the on-time rate over all slips with the rate over mature slips,
whose denominator cannot move. Recent due dates are listed one by one; older ones, all definitive,
are summarized. The distortion lives in the open period, which is the view people check daily.

## 06 `eda`: exploratory report

Structure, column profile (no sample values), the outcome, numeric statistics with skew, data-quality
checks with severities, two deep dives (a zero balance is a value, not a gap; target leakage),
cardinality with rare categories, referential integrity between the four tables, monthly coverage
with a stability verdict, Spearman correlation with non-payment (leaking columns flagged), IQR
outliers, findings, recommendations and limits. Every finding is a sentence built from numbers
computed on the dataset at hand. Each row of the recommendation table appears only when a check on
that dataset fired (for example, the term row needs a long right tail in days to the first due
date). One line of fixed advice sits below the table: keep columns measured after the outcome out of
a model.

## 07 `diagnose`: cleaning diagnostics (report only)

In the order in which each step depends on the one before: structural (whitespace, control
characters, names stuck at one exact length, which suggests truncation at the source), types
(unparseable dates; person keys that look numeric and would lose leading zeros), duplicates (whole
row, row id, agreement, row id across product lines), missing values (with a check on whether the
null itself predicts the outcome), invalid and inconsistent values (zero or negative balances,
values that differ only in case, accents or spacing, agent name and login mapping, accounts that
look like system accounts but passed the exclusion filter), and outliers. It changes and writes
nothing but the report. It does not validate identifier formats.

## 08 `clean`: additive, reversible cleaning

| Rule | What it does |
|---|---|
| R1 `is_system` | flags rows whose agent name or login is in `system_agents` (case, accent and spacing insensitive); never deletes |
| R2 `zero_balance` | flags a zero original debt; the installment ratio stays `NaN`; nothing is imputed |
| R3 leakage | lists columns measured after the outcome (`paid_amount`, `paid_date`, `days_overdue`, `status_id`, `status_label`) and the explanatory-only follow-up columns in `clean_meta.json`; keeps them for audit |
| R4 `portfolio_group` | the label without its trailing sequence token (`portfolio_group_pattern`); the original stays |
| R5 `agreement_key` | `product_line\|agreement_id\|portfolio_id`, asserted unique; a row id alone collides across product lines |
| R6 `group_key` | the person key, or the debtor row when a slip has none, the unit to split on; keyless slips never pool |
| R7 `is_mature` | the outcome is final; models train on mature rows only |

Outliers are kept: a winsorized value would erase the large slips, which weigh the most in any
value total. `clean_full.csv` has every row; `clean_analytic.csv` drops the flagged system rows.
The report ends with an information-loss summary: rows removed, cells imputed, columns
removed (all zero), columns added, rows flagged, and how far the baseline moved.

## 09 `card`: the summary card with a maturity cut

Recomputes the indicator for any number of periods side by side (`--period NAME=FROM:TO`; default:
the open month and the two before). For each period: generated, paid on time, to mature, overdue
unpaid *as the card counts it*, of which still in grace and of which lost; first-installment and
whole-agreement value of each; the inflation of the card count; and the on-time rate as the card
computes it against the rate over mature slips. In a closed period everything has matured and the
two agree. All periods come from one fetch (`--as-of` sets the date) or from the dataset
(`--from-dataset`; the dataset fixes the date).

## 10 `model`: the effect of the term with the mix controlled

The analysis stage reports the term, the days the agent grants between the agreement and the
first due date, as one factor among others. Reading its unpaid rate at face value invites the
objection of confounding: agents, portfolios and ticket sizes differ, and longer terms may sit
with the riskier mix. The question here is narrow: *given the same mix, is a longer term still
associated with more non-payment?*

| Choice | Why |
|---|---|
| Mature, non-system rows only | the label needs a final outcome |
| Features known when the term is agreed | no leakage: payment amount and date, days overdue and status are out; the debt balance counts only if it was read as of the agreement (data contract) |
| No follow-up counts | reverse causality: if outreach goes to people already slipping, follow-up goes with non-payment |
| Cross-validation grouped by person | rows of one person are not independent; a row split leaks people across folds |
| Temporal hold-out | train on early agreements, test on late ones: the second validation |
| Logistic model and gradient boosting | a transparent model for the effect, a flexible one to see if interactions matter |
| Numeric imputation inside the pipeline | fitted per fold, never on the test side |
| A zero balance and a missing debt are separate flags | they mean different things |

Outputs: grouped cross-validated AUC for both models against a term-only baseline (and the share of
the signal above chance the term alone captures); odds ratios of each term band against the
shortest observed band, computed as differences of coefficients so every band is compared with
the same reference; odds ratios for the other controls; permutation importance on the hold-out; and
a counterfactual that rewrites only the term of every agreement and keeps the rest, which is the
term's effect isolated from the mix.

Read it as an association measured under control, not as proof that changing the term changes
behavior: people who ask for a long term may already be the people who cannot pay. A randomized
term is the only design that would settle that. The model measures an effect under control and is
not a scoring tool, so read the odds ratios and the counterfactual rather than the AUC alone. Models
are fitted in memory and never saved.
