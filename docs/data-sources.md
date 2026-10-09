# Data sources

The repository holds code only. Data comes in three tiers, and the code and tests that ship here use
only the first.

| Tier | Source | Status |
|---|---|---|
| 0 | Seeded synthetic data from `examples/build_demo_db.py`, and inline invented rows in the tests | used by the demo and the tests |
| 1 | A public dataset fetched at run time | none; the pipeline downloads nothing |
| 2 | Your own database, mapped onto the [data contract](data-contract.md) | used on your side; never committed |

## Tier 0: seeded synthetic data

`python examples/build_demo_db.py data/demo` writes two small SQLite databases, `auto.db` and
`consumer.db`, in the contract schema. The defaults are seed `7` and reference date `2026-03-31`;
the same seed and date give the same files. Every person, agent, portfolio, amount and date comes
from a seeded random generator. The generator reads no file, dataset or network service.

Every constant in the generator is arbitrary. Each was picked by hand so that the situation it plants
is visible in a few thousand rows. None comes from a public source, so none is cited, and none was
fitted to, tuned against or checked against any real dataset or report.

| Group | Constants | What they set |
|---|---|---|
| World | seed `7`; reference date `2026-03-31`; grace period `3` days | the invented "today" and the on-time rule the generator pays against |
| Product lines | `auto`: 900 slips, ticket scale 800.0, zero-balance share 0.12, status ids from 21. `consumer`: 600 slips, ticket scale 450.0, zero-balance share 0.23, status ids from 31 | size and typical amount of each line, and the share of agreements with linked debts whose balance is zero |
| People | a pool of person keys numbering 55% of all slips; 3% of slips get a debtor row with no person key; a person who has one debtor row gets a second with chance 0.15 each time they come up again | repeat debtors and keyless rows |
| Agents | 12 regular agents with a term shift drawn from uniform(-2, 6) days and a personal effect drawn from normal(0, 0.25); three pseudo-agents with weights 0.4, 0.3 and 0.5 against 1.0; the batch loader has a term shift of 8 and never has a slip paid | who grants long terms, and the accounts the diagnostics should find |
| Slips | issued 0 to 150 days before the reference date; term drawn from an exponential with mean max(4 + shift, 1) days and capped at 40; plan sizes drawn from 1, 1, 2, 3, 6, with later installments due every 30 days after the first; agreement total lognormal around the ticket scale with sigma 0.8; one of 8 portfolio labels, uniformly; agreement ids start at 1000 in each line, so they repeat across the two | the mix of terms, tickets and portfolios |
| Payment model | chance of not paying on time = logistic(-2.60 + 0.22 x term days + 0.55 x log(1 + earlier breaks) - 0.20 x log(first installment / ticket scale) + agent effect) | the term effect that stage 10 tries to recover |
| After the due date | payers pay between 2 days before the due date and the end of the grace period, and 90% of them end as PAID, 10% as ACTIVE; 15% of the slips not paid on time, except the batch loader's, are paid 4 to 15 days after the due date; slips not paid on time whose due date is more than 7 days before the reference date, including those paid late, end BROKEN 45%, AUTO_BROKEN 20%, ACTIVE 35%, and the others stay ACTIVE | the buckets and the still-open slips |
| Noise | 3% PENDING_EXCEPTION; 2% of first installments written twice (second row 5.00 higher, issued a day later); 3% of agreements with no linked debt; one linked debt for two agreements in three and two for the rest; balance = total x uniform(1, 4) / debt count; every debt has `due_at` 2025-09-01 and `days_overdue` 90 (optional columns the pipeline does not read); 2% of debts with no portfolio label | the situations the extraction rules exist for |
| Earlier breaks | 0, 1, 2 or 3 earlier broken agreements per person with weights 0.8, 0.12, 0.05 and 0.03; issued 160 to 400 days before the reference date and due 7 days later; amount 120.00; two in three end BROKEN and one in three AUTO_BROKEN; all owned by agent `a01` | the repeat-break history |
| Outreach | Poisson count with mean 1.0 for paid slips and 3.0 for unpaid ones, capped at 8; five result types with weights 0.45, 0.15, 0.20, 0.10, 0.10; each attempt on a day from the agreement date to the due date (or the reference date, if earlier), between 08:00 and 18:59 | the follow-up columns |

The planted effects (a longer term goes with more non-payment, one pseudo-agent never collects, ids
repeat across the two lines) are assumptions of the generator. They give each stage something to
find. They are not findings about any real operation, and their sizes mean nothing outside the demo.

The tests build their own tiny databases and frames from inline rows with invented values. No
fixture file holds data.

The pipeline settings (grace period, band edges, status labels, exclusion patterns, reliability
floors) are placeholders in the same sense. They exist so the demo and the tests run; calibrate them
on your own data.

## Tier 1: public datasets

None. The pipeline does not fetch anything at run time.

## Tier 2: your own data

You connect your own database through views that follow the data contract, using environment
variables for the connection. Nothing from it is stored in the repository. Everything the pipeline
writes lands under `data/` and `reports/`, and `.gitignore` also excludes `.env`, `*.db`, `*.csv`,
`*.parquet`, `*.pkl` and `*.joblib`. The dataset file holds whatever keys and agent ids your database
supplies, so keep it out of version control and treat it like the source data.
