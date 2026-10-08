# Architecture

## Shape

```
                      env vars (DEA_*), --config JSON, flags
                                     |
                               cli.py  -- builds one Context --> stage.run(ctx, args)
                                     |
   +---------------------------------+-----------------------------------------+
   |  database-reading stages        |             offline stages              |
   |  00 period  01 schema           |  03 analysis  04 drift  05 censoring    |
   |  02 base    09 card             |  06 eda  07 diagnose  08 clean  10 model|
   +----------------+----------------+------------------+----------------------+
                    |                                   |
            db/ adapters + queries.py             dataset.py  (JSON bundle)
          (SQLite | SQL Server dialects)           base | debtors | contacts | breaks
                    |                                   |
              read-only SELECTs                  features.py  derive.py  tables.py
                    |                                   |
                 database                       reports/*.md  clean_*.csv
```

`base` is the bridge: it is the only stage that turns database rows into the dataset file, and
every offline stage starts from that file. That is what makes a run repeatable and lets you do the
analysis on a machine that cannot reach the database.

## Layers

| Layer | Modules | Rule |
|---|---|---|
| Entry | `cli.py`, `__main__.py` | Parses arguments, builds a `Context`, maps `PipelineError` to a one-line message and an exit code. Nothing else reads `sys.argv` or the environment. |
| Context | `context.py` | Settings, sources, paths, output stream, an optional connector for tests. Opens and closes databases. |
| Stages | `stages/s00_period.py` ... `s10_term_model.py` | Each exposes `add_arguments(parser)` and `run(ctx, args)`. Pure computation lives in plain functions next to `run`, so tests call it without a database or a command line. The registry in `stages/__init__.py` is the single list the CLI, the README table and the tests read. |
| Domain | `derive.py`, `features.py`, `tables.py`, `bands.py` | The outcome rule, row-level features, the shared group tables, right-closed bands. No I/O. |
| Data access | `db/`, `queries.py`, `dataset.py`, `contract.py`, `sqlutil.py` | Adapters, SQL text builders, normalization into the dataset, the contract definition, the read-only guard. |
| Support | `config.py`, `dates.py`, `report.py`, `textutil.py`, `env.py`, `errors.py` | Validated settings, date and number normalization, a Markdown builder, small helpers. |

## Data access

### Adapter protocol

`db.base.Database` has three methods: `query(sql, params)`, `table_columns(table)` and `close()`.
`DBAPIDatabase` implements them over any PEP 249 connection with an `INFORMATION_SCHEMA`
catalogue, and checks every statement with `ensure_read_only` first. A subclass only chooses the
dialect and how the connection is opened:

| Adapter | Used for | Connection |
|---|---|---|
| `SQLiteDatabase` | demo databases, tests | a file path, opened `mode=ro` |
| `SQLServerDatabase` | real use | `pyodbc` (optional extra), settings from `DEA_DB_*` variables only; an injected connector makes it testable |

### Dialects

`queries.py` builds portable SQL (CTEs, `ROW_NUMBER`, `IN` lists). Only three things differ, and
`db/dialect.py` isolates them:

| Difference | SQL Server | SQLite |
|---|---|---|
| Table hint | `agreements AS r WITH (NOLOCK)` | `agreements AS r` |
| One label per agreement | `OUTER APPLY (SELECT TOP 1 ... ORDER BY debt_id)` | scalar subquery with `LIMIT 1` |
| Date parameter | a `date` object | ISO text |

### Pure queries

Each builder returns a `Query(sql, params)` and touches no database, which is how the tests can
assert that every SQL Server table reference carries `NOLOCK`, that filters sit inside the ranked
table, and the order of bound parameters.

### Chunking

`IN` lists are sent in chunks (`id_chunk_size`, at most 1000) to stay under driver parameter
limits. A test proves a tiny chunk size returns the same dataset as a large one.

## The dataset bundle

`Dataset` is one JSON file: a `meta` block (period, reference date, grace days, the status labels
and agent patterns used, a schema version) and four tables. `Dataset.load` rejects a file with a
different schema version instead of guessing. Every key is prefixed with the product line.

`features.enrich` joins the tables in memory. It groups debtor rows into people by `person_key`,
gathers a person's follow-up inside the window from agreement to due date, counts earlier broken
agreements made before this one, and derives the installment-to-debt ratio with its state.
`features.analysis_frame` adds the bands, the portfolio group and the maturity flag, using the grace
period the dataset was built with (the one that produced its `bucket` column).

## Reproducibility

- Dates are bound parameters; the reference date is an argument, never the server clock.
- Every table is sorted in Python by stable keys after the build (`dataset.sort_tables`), so the
  file does not depend on the order the database returns rows in, and the JSON is written with a
  fixed layout. The same command on the same data gives the same bytes; tests check both the
  repeat run and a database that returns every result set reversed.
- Randomness (model folds, boosting, permutation importance) is seeded from `model.random_state`.
- Reports are generated, not edited: re-running a stage overwrites its report.

## Testing strategy

| Test group | What it proves |
|---|---|
| Foundations | bands, date and number normalization, settings validation, the read-only guard, the Markdown builder |
| Derive | the grace-period boundary, the maturity rule, replay against an earlier date |
| Extraction | SQL text for both dialects, then the real SQL against a small SQLite world where each rule has one row that proves it |
| Adapters | the SQL Server adapter through a fake DB-API connection; the password never reaches an error message |
| Stages | each stage's computation on rows built to trigger it, then its report and its CLI |
| End to end | the whole pipeline on the seeded demo databases, which plant the situations each stage should find |

All fixtures are inline invented rows or the seeded generator. No fixture file holds a realistic
row.

## Adding a stage

1. Write `stages/sNN_name.py` with a module docstring that starts with `Stage NN` and explains what
   it computes and why, plus `add_arguments` and `run`.
2. Add it to `STAGES` in `stages/__init__.py`. Set `needs_database` to whether it calls
   `ctx.databases()`; a test checks the flag against the source.
3. Put computation in plain functions and test them without a database.
