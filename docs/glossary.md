# Glossary

Terms and field names used in the code, reports and settings.

## Concepts

| Term | Meaning |
|---|---|
| Product line | One database with the contract schema (for example `auto`, `consumer`). Ids repeat across lines, so keys carry the line. |
| Agreement | A payment plan a debtor agreed to. One row per installment in `agreements`. |
| Installment | One payment of an agreement. `installment_no = 0` is the first installment (the down payment). |
| First-installment slip (slip) | The payment slip of installment 0. The unit the pipeline analyzes. |
| Generated agreement | An agreement whose status is in `generated_status_labels`: the population in scope. |
| Broken / auto-broken agreement | An agreement that ended broken (by hand or by the system after a grace overflow). Earlier ones count as a person's break history. |
| Exception (pending approval) | A status awaiting approval. Not in the generated set by default, so it is not extracted. |
| Portfolio | A group of debts with one label, for example `Portfolio A-02`. |
| Portfolio group | The label without its trailing sequence token (`Portfolio A`), used to avoid one category per sequence. |
| Agent | The user who owns an agreement. |
| Outreach attempt | One contact event. |
| Answered | Someone picked up. |
| Right-party contact (CPC) | `answered AND right_party`. |
| Follow-up | Outreach attempts after the agreement day and up to the due date. The agreement day itself is the negotiation, not a reminder. |
| Repeat debtor | A person with at least one earlier first-installment agreement that ended broken. |
| Grace period | `grace_days` after the due date within which a payment still counts as on time. |

## Outcome

| Term | Meaning |
|---|---|
| `paid_on_time` | Paid on or before `due + grace_days`, and not after the reference date. |
| `overdue_unpaid` | Due before the reference date and not paid on time. The bucket under study. |
| `to_mature` | Not due yet. |
| Still open | Overdue, unpaid, grace period not over (`days_overdue <= grace_days`). Right-censored. |
| Definitive (lost) | Overdue, unpaid, grace period over. |
| Mature | `days_overdue > grace_days`. The outcome is final. |
| Reference date (`as_of`) | The date outcomes are derived against. Always explicit. |
| Censoring | The outcome is not known yet because the observation window has not closed. |
| Leakage | A feature measured after the outcome, or that encodes it. |

## Dataset columns (base table)

| Column | Meaning |
|---|---|
| `product_line`, `record_id`, `agreement_id`, `portfolio_id`, `debtor_id` | Keys. Never use `record_id` alone. |
| `person_key` | Opaque token for the same person across debtor rows and product lines. |
| `status_id`, `status_label` | Status of the agreement. |
| `first_installment_value` | Amount of the first installment. |
| `total_value` | Sum of all installments of the agreement (each installment's latest row). |
| `total_installments`, `plan_installments` | Installments actually generated, and the plan the agreement declared. |
| `original_debt` | Sum of the balances of the linked debts, each as of the agreement date. `NaN` when nothing is linked. |
| `agent_id`, `agent_name`, `agent_login` | The owner. |
| `portfolio_label` | Label of one linked debt. |
| `agreement_date`, `due_date`, `paid_amount`, `paid_date` | Raw dates and payment. |
| `days_to_due` | The term: days from the agreement to the first due date. |
| `days_overdue` | Days past the due date at the reference date; zero or negative while not due. |
| `bucket` | One of the three outcome buckets. |

## Derived columns

| Column | Meaning |
|---|---|
| `contacts_total` | Outreach attempts in the window, including the agreement day. |
| `contacts_after`, `answered_after`, `cpc_after` | Attempts after the agreement day: all, answered, right-party. |
| `followup_level` | Strongest follow-up: none, right party reached, answered but not the right party, attempts only. |
| `known_person` | The slip's person key was found in the debtors table. |
| `prior_breaks`, `prior_manual_breaks` | Earlier broken agreements of the person (all, and manual breaks only), excluding the agreement itself and anything made after it. |
| `ratio_state`, `first_installment_ratio` | Why the installment-to-debt ratio is or is not defined, and its value. |
| `term_band`, `value_band`, `installment_band`, `ratio_band` | Ordered bands from the settings. |
| `portfolio_group` | See above. |
| `is_mature`, `is_system`, `zero_balance` | Flags. |
| `agreement_key`, `group_key` | Composite agreement key and the unit to split on. |

## Metrics

| Metric | Formula |
|---|---|
| Unpaid rate (mature) | mature slips in `overdue_unpaid` / mature slips |
| Lift | a group's unpaid rate / the baseline unpaid rate |
| Inflation | overdue-unpaid slips as counted / definitive overdue-unpaid slips |
| On-time rate, as the card computes it | `paid_on_time` / all generated slips |
| On-time rate over mature slips | mature `paid_on_time` / mature slips |
| Signal share of the term | (AUC of the term-only model - 0.5) / (AUC of the best model - 0.5) |
| Odds ratio of a term band | `exp(coef[band] - coef[reference band])` |
