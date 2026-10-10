# Delinquency EDA Pipeline

A numbered, reproducible analysis pipeline for overdue-unpaid receivables. It has eleven
command-line stages: period selection, schema discovery, a repeatable dataset build, factor
analysis, drift replay, a censoring check, an EDA report generator, cleaning diagnostics, additive
cleaning, a corrected summary card, and a payment-term model that asks whether the term a collector
grants still matters once the portfolio, agent and ticket mix are held fixed. Everything reads a
documented, generic data contract, so you connect your own database and no data ships with the
code.

## What it does

Most of the code is about counting the overdue-unpaid bucket correctly:

- "Overdue and not paid on time" counts slips that can still be paid on time (right-censoring), so
  the bucket is inflated for the periods that still hold due dates from the last `grace_days` days:
  month to date, plus the previous month during the first `grace_days` days of a month. Stage 05
  splits the bucket into still open and definitive. Every unpaid rate used to compare groups is
  computed on the mature cohort, selected by due date alone and never by outcome.
- Outcomes are derived in Python against the `--as-of` date, never the database server's clock.
  Without the flag, `period`, `base` and `card` use today's local date, so pass it for a run you
  want to repeat: the same command with the same date writes the same dataset, byte for byte.
  Stage 04 can replay the past from the dataset alone.
- Stage 00 finds the window that reproduces a card you were shown. Stage 04 finds which slips moved
  since, and which value measure (first installment, whole agreement, original debt) a headline
  figure sums.
- Columns measured after the outcome are listed and excluded from the model (leakage). Follow-up
  counts are descriptive only: if outreach goes to people who are already slipping, follow-up and
  non-payment are entangled (reverse causality).
- Rows of one person are not independent, so cross-validation is grouped by an opaque person key
  and a temporal hold-out is the second check.
- Cleaning adds flags, derived columns and a composite key. It deletes no rows and imputes no
  cells, and its report shows both counts. A zero balance and a missing debt are different things
  and stay different.
- The reports call their results associations and name the design that would show cause (a
  randomized term).

## Architecture

```
  env vars (DEA_*) / --config JSON / flags
                  |
               cli.py --> Context --> stage.run(ctx, args)
                  |
     +------------+--------------------------+
     | reads the database                    | offline, reads the dataset file
     | 00 period  01 schema                  | 03 analysis  04 drift  05 censoring
     | 02 base    09 card                    | 06 eda  07 diagnose  08 clean  10 model
     +------------+--------------------------+
                  |                                    |
        db/ adapters + queries.py             dataset.json (base | debtors | contacts | breaks)
        SQLite | SQL Server, read only                 |
                  |                         features.py  derive.py  tables.py
               database                             reports/*.md  clean_*.csv
```

`base` is the only stage that turns database rows into the dataset file, and every offline stage
starts from that file. More in [docs/architecture.md](docs/architecture.md).

## Quick start

Needs Python 3.10 or newer. This runs on the built-in demo data (invented, seeded, two small SQLite
databases) and needs no database server.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"

python examples/build_demo_db.py data/demo

export DEA_BACKEND=sqlite
export DEA_SOURCES=auto=data/demo/auto.db,consumer=data/demo/consumer.db
export DEA_CONFIG=examples/settings.example.json

delinquency-eda schema
delinquency-eda period --as-of 2026-03-31
delinquency-eda base --from 2025-11-01 --to 2026-03-31 --as-of 2026-03-31
delinquency-eda analysis
delinquency-eda censoring
delinquency-eda drift --reference-date 2026-03-27
delinquency-eda eda
delinquency-eda diagnose
delinquency-eda clean
delinquency-eda card --from-dataset
delinquency-eda model
```

In PowerShell, activate with `.venv\Scripts\Activate.ps1` and replace the three `export` lines with:

```powershell
$env:DEA_BACKEND = "sqlite"
$env:DEA_SOURCES = "auto=data/demo/auto.db,consumer=data/demo/consumer.db"
$env:DEA_CONFIG = "examples/settings.example.json"
```

Reports are Markdown files under `reports/`; the dataset is `data/dataset.json`; both are
gitignored. `delinquency-eda --help` lists the stages and `delinquency-eda <stage> --help` its
options. Global options (`--dataset`, `--out-dir`, `--config`, `--backend`, `--source`,
`--env-file`) may come before or after the stage name. The `model` stage is the slow one because of
the gradient-boosting model and permutation importance. On the demo data it takes under a minute on
a CI runner and several minutes on a slow or busy machine.

The demo plants situations the stages should find: a term effect on payment, slips still inside
their grace window, a pseudo-agent that passes the default exclusion patterns and never collects,
re-written installments, repeat debtors, and ids that repeat across the two product lines. Every
constant in the generator is an arbitrary hand-picked value, so the effect sizes mean nothing
outside the demo. [docs/data-sources.md](docs/data-sources.md) lists the constants and the three
tiers of data (synthetic demo, public datasets, your own data).

## Connect your own data

1. Map your database onto the nine contract tables below, most simply with views, as described in
   [docs/data-contract.md](docs/data-contract.md) (types, nulls, rules and a mapping sketch). One
   database per *product line*. Names are fixed; yours can be anything behind a view.

   | Table | One row per | Required columns |
   |---|---|---|
   | `agreements` | installment of an agreement | `row_id`, `agreement_id`, `portfolio_id`, `debtor_id`, `installment_no`, `amount`, `plan_installments`, `paid_amount`, `issued_at`, `due_at`, `paid_at`, `status_id`, `agent_id` |
   | `agreement_statuses` | status | `status_id`, `label` |
   | `agents` | user who owns agreements | `agent_id`, `display_name`, `login` |
   | `agreement_debts` | link of an agreement to a debt | `agreement_id`, `portfolio_id`, `debt_id` |
   | `debts` | debt | `debt_id`, `balance` |
   | `debt_attributes` | debt attribute set | `debt_id`, `portfolio_label` |
   | `debtors` | debtor row | `debtor_id`, `person_key` (an opaque token, never a name or a national id) |
   | `contact_events` | outreach attempt | `event_id`, `debtor_id`, `result_id`, `occurred_at` |
   | `contact_results` | contact result | `result_id`, `label`, `answered`, `right_party` |

   `installment_no = 0` is the first installment, the one the pipeline analyzes. A missing payment
   is `NULL`, never `0`. `debts.balance` must be the balance as of the agreement date: a balance
   read later can already reflect payments and would leak the outcome into the model's controls.
2. Install the SQL Server extra and configure the connection with environment variables only
   (copy `.env.example`; never commit the real file):

   ```bash
   python -m pip install -e ".[sqlserver]"
   export DEA_BACKEND=sqlserver
   export DEA_SOURCES=auto=your_database_one,consumer=your_database_two
   export DEA_DB_HOST=db.example.invalid
   export DEA_DB_USER=report_reader
   export DEA_DB_PASSWORD=change-me
   export DEA_DB_SCHEMA=reporting
   ```

   The same in PowerShell:

   ```powershell
   python -m pip install -e ".[sqlserver]"
   $env:DEA_BACKEND = "sqlserver"
   $env:DEA_SOURCES = "auto=your_database_one,consumer=your_database_two"
   $env:DEA_DB_HOST = "db.example.invalid"
   $env:DEA_DB_USER = "report_reader"
   $env:DEA_DB_PASSWORD = "change-me"
   $env:DEA_DB_SCHEMA = "reporting"
   ```

   On any shell you can instead fill in a copy of `.env.example` and pass `--env-file .env`.

   Use a login that only has read permission. The pipeline also refuses any statement that is not a
   single `SELECT` or `WITH ... SELECT`. That guard catches mistakes in the pipeline's own SQL;
   access control still has to come from a read-only login.
   The SQL Server path has not been run against a live server (see Limitations); the extraction
   SQL has been run end to end on SQLite.
3. Run `delinquency-eda schema`. It lists every missing table and column, the types it found, and
   any configured status label absent from your catalog.
4. Copy `examples/settings.example.json`, calibrate it on your own data, and pass it with
   `--config`. Every threshold in it is an invented default (see below).
5. Run `period` to find the window your existing dashboard uses, then `base` and the rest.

## Configuration

Connection settings come from environment variables only (a `.env`-style file can be loaded with
`--env-file`; variables already set win).

| Variable | Default | Meaning |
|---|---|---|
| `DEA_BACKEND` | `sqlserver` | `sqlserver` or `sqlite` |
| `DEA_SOURCES` | none | `line=target,line=target`: a product line and its database name (SQL Server) or file path (SQLite) |
| `DEA_CONFIG` | none | path of the JSON settings file |
| `DEA_DB_DRIVER` | `ODBC Driver 18 for SQL Server` | ODBC driver name |
| `DEA_DB_HOST` | `localhost` | server host |
| `DEA_DB_PORT` | none | server port |
| `DEA_DB_USER` | none | SQL login; empty means integrated authentication |
| `DEA_DB_PASSWORD` | none | SQL password; never printed, scrubbed from error text |
| `DEA_DB_SCHEMA` | none | schema holding the contract views; qualifies table names |
| `DEA_DB_TRUST_SERVER_CERT` | `no` | `yes` only for local test servers |
| `DEA_DB_CHAR_ENCODING` | `utf-8` | how the driver decodes `CHAR` columns (`cp1252` for Latin-1 collations) |
| `DEA_DB_LOGIN_TIMEOUT` | `10` | seconds |
| `DEA_DB_QUERY_TIMEOUT` | `120` | seconds |

Rules and thresholds live in the JSON settings file. Every default below is invented so the demo
and the tests run; calibrate each on your own data. Unknown keys are an error.

| Setting | Default | Meaning |
|---|---|---|
| `grace_days` | `3` | days after the due date within which a payment still counts as on time |
| `generated_status_labels` | `ACTIVE, BROKEN, PAID, AUTO_BROKEN, PAID_MANUAL` | statuses in scope, by label |
| `broken_status_labels` | `BROKEN, AUTO_BROKEN` | statuses that count as a break in a person's history |
| `manual_break_label` | `BROKEN` | the break made by hand, counted separately |
| `excluded_agent_patterns` | `system%, test%` | `LIKE` patterns on agent login and display name, applied inside the SQL |
| `system_agents` | none | agents to flag `is_system` in cleaning (not deleted) |
| `id_chunk_size` | `500` | ids per `IN` list (1 to 1000) |
| `term_band_edges` | `0, 2, 5, 10, 20` | upper edges of the term bands, in days (right-closed) |
| `value_band_edges` | `100, 500, 2000, 10000` | first-installment value bands |
| `ratio_band_edges` | `0.05, 0.2, 0.5` | installment-to-debt ratio bands |
| `installment_band_edges` | `1, 2, 4, 8, 16` | installment-count bands |
| `min_n_reliable` | `30` | group size from which a rate is flagged reliable |
| `min_n_weak` | `8` | group size from which it is flagged weak (below: not reliable) |
| `chronic_break_threshold` | `4` | earlier breaks from which a debtor is called chronic |
| `mnar_gap_pp` | `10.0` | unpaid-rate gap, in points, from which a null is said to carry signal |
| `stability_range_pp` | `10.0` | spread across months, in points, from which pooling months is flagged |
| `rare_category_min_n` | `25` | below this a category counts as rare |
| `outlier_iqr_k` | `3.0` | width of the IQR fence |
| `portfolio_group_pattern` | `[\s-]+\d+$` | trailing sequence token removed to get the portfolio group |
| `suspect_agent_pattern` | `system\|robot\|bot\|batch\|auto\|test\|import` | agents flagged by the diagnostics as possible system accounts |
| `model.folds` | `5` | grouped cross-validation folds |
| `model.random_state` | `42` | seed for boosting and permutation importance (the grouped folds need none) |
| `model.logistic_c` | `1.0` | inverse regularization of the logistic model |
| `model.logistic_max_iter` | `2000` | optimizer iterations |
| `model.boosting_max_iter` | `300` | boosting rounds |
| `model.boosting_learning_rate` | `0.06` | boosting step |
| `model.min_category_count` | `20` | categories rarer than this are pooled before encoding |
| `model.holdout_quantile` | `0.75` | agreement-date quantile that splits train from the time hold-out |
| `model.permutation_repeats` | `10` | shuffles per feature for permutation importance |

## The stages

| # | Command | What it computes, and why |
|--:|---|---|
| 00 | `period` | The funnel (generated, paid on time, overdue unpaid, to mature) over candidate date windows from one fetch, ranked by closeness to counts you already have. Use it when a card does not say which window it covers. |
| 01 | `schema` | Which contract tables and columns exist, their types, and whether configured status labels are in the catalog. Run first against a new database. |
| 02 | `base` | The dataset bundle: first-installment slips with derived outcomes, the debtor rows of the same people, outreach events, and earlier broken agreements. Stages 03 to 08 and 10 read only this file. |
| 03 | `analysis` | Per factor, concentration of the unpaid value and discrimination of the unpaid rate on the mature cohort, with lift and sample-size flags. |
| 04 | `drift` | The same slips re-derived at an earlier date: buckets, transitions, the slips that moved, and which value measure a headline figure sums. |
| 05 | `censoring` | The unpaid bucket split into still open and definitive, the inflation of the naive count, and the on-time rate with and without the maturity cut. |
| 06 | `eda` | A report: profile, outcome, quality checks, zero-balance and leakage deep dives, integrity, time coverage, correlation, outliers, computed findings and recommendations. |
| 07 | `diagnose` | Cleaning diagnostics (structure, types, duplicates, missing values, inconsistencies, outliers). Changes nothing. |
| 08 | `clean` | Additive cleaning: flags, a composite agreement key, a person-level split group, a leakage list, and an information-loss summary. Writes the cleaned tables. |
| 09 | `card` | The summary card with a maturity cut, for any number of periods side by side. |
| 10 | `model` | Odds ratios of the term bands against the shortest observed band with the mix controlled, grouped cross-validated AUC against a term-only baseline, permutation importance on a time hold-out, and a term-only counterfactual. |

The full reference, with what each stage reads and writes, is [docs/stages.md](docs/stages.md).

## Tests

```bash
python -m pytest        # the whole suite
ruff check .            # lint
```

Fixtures are inline invented rows or the seeded demo generator; no fixture file holds a realistic
row. The suite covers the SQL text for both dialects, the real SQL on a small SQLite world where
each extraction rule has one row that proves it, every stage's computation on rows built to trigger
it, the command line, the README itself (its commands parse and its tables match the code), and the
whole pipeline end to end on the demo databases. The model tests are the slowest.

## Project layout

```
src/delinquency_eda/
  cli.py  context.py  config.py     command line, run context, validated settings
  derive.py  features.py  tables.py outcome rule, row-level features, shared group tables
  bands.py  dates.py  report.py     bands, normalization, Markdown builder
  queries.py  dataset.py  contract.py  SQL builders, the dataset bundle, the contract
  sqlutil.py  textutil.py  env.py   read-only guard, text folding, .env reader
  db/                               adapter protocol, SQLite and SQL Server adapters, dialects
  stages/                           s00_period.py ... s10_term_model.py and the registry
examples/
  build_demo_db.py                  seeded generator of the demo databases
  settings.example.json             every setting: invented defaults, the demo's pseudo-agent
docs/
  architecture.md  data-contract.md  data-sources.md  stages.md  decisions.md  glossary.md
tests/                              pytest suite
.github/workflows/ci.yml            install, lint, test, run the quick start
```

## Key design decisions

The reasoning for each is in [docs/decisions.md](docs/decisions.md).

- The unpaid bucket is censored, so rates use the mature cohort, and maturity depends on the due
  date only.
- Outcomes are derived in Python against a date the caller sets.
- The extraction runs business filters before the latest-row-wins ranking, resolves statuses by
  label, keys agreements by a composite, and takes one portfolio label per agreement with a top-one
  lookup.
- A zero balance and a missing debt stay apart, and cleaning only adds.
- Concentration and discrimination are separate tables.
- Follow-up stays out of the model, and the debt balance is read as of the agreement.
- Validation groups by person and holds out by time. Term bands are compared against a stated
  reference band.
- The analysis and model reports call their results associations.

## Limitations

- SQL Server is not exercised against a live server. The SQL Server adapter and dialect are tested
  through a fake DB-API connection and by inspecting the generated T-SQL (hints, top-one lookup,
  bound dates, parameter order). The extraction SQL runs end to end only on SQLite.
- CI runs the suite on Python 3.10, 3.12 and 3.14. `requires-python` is `>=3.10`, so 3.11 and 3.13
  install but are not in the CI matrix. Development and the local test runs used Python 3.14 on
  Windows.
- The default settings (grace period, bands, status labels, exclusion patterns, reliability floors)
  are placeholders so the demo runs. The demo generator is arbitrary too: every constant in it (line
  sizes, ticket scales, zero-balance shares, payment-model coefficients) is a hand-picked value, not
  fitted to or derived from any dataset. It plants effects so each stage has something to find, and
  their sizes are not a finding about any real operation.
- The pipeline cannot tell when a debt balance was read. The contract asks for the balance as of the
  agreement date; with a later balance, the original-debt and zero-balance controls of the model
  can leak the outcome.
- The model measures an association under control. Nothing here shows that changing a term changes
  behavior, and the data has no field for why a slip was not paid.
- The model is not a scoring tool. Its odds ratios and counterfactual answer the term question; the
  AUC only shows how well the features separate payers from non-payers.
- The dataset holds the latest status of each agreement, so later breaks and re-writes cannot be
  rewound by the drift stage.
- Feature construction joins people, contacts and breaks in Python loops. A rough check on one
  Windows machine built the features (`analysis_frame`) for 20,000 invented slips and 80,000
  contacts in about four seconds; the time depends on the machine. The model stage costs far more,
  and nothing is tuned for millions of rows.
- `data/dataset.json` holds the opaque keys and agent identifiers your database supplies. Keep it
  out of version control and treat it like the source data.
- On some Windows hosts without the `wmic` tool, joblib prints a harmless warning about physical
  core detection during the model stage.

This is an independent portfolio project. It is not affiliated with any company and contains no
real data.

## License

MIT. See [LICENSE](LICENSE).
