"""Stage 08 - cleaning: additive, reversible, and honest about what it changed.

The diagnostics usually show that the data is structurally clean and semantically tricky: columns
are valid but cannot be used as they stand. So the rules add information and remove none.

    R1  ``is_system``      flag rows owned by listed system agents (flagged, never deleted)
    R2  ``zero_balance``   flag a zero original debt; the instalment ratio stays NaN, no imputation
    R3  leakage columns    listed in the meta file, kept for audit, excluded from models
    R4  ``portfolio_group``  portfolio label without its sequence token (original kept)
    R5  ``agreement_key``  ``product_line|agreement_id|portfolio_id``, asserted unique
    R6  ``group_key``      person key (or the debtor row when there is none), the unit to split on
    R7  ``is_mature``      outcome is final; models train on mature rows only

Outliers are kept (rule: none): long-tailed money is real. Two tables are written, ``full`` (every
row) and ``analytic`` (system rows removed, the default view), plus a meta file that models read.
The report ends with an information-loss summary: rows removed, cells imputed, columns removed,
columns added, and how much the baseline moved.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import pandas as pd

from delinquency_eda import features as F
from delinquency_eda.config import Settings
from delinquency_eda.context import Context
from delinquency_eda.dataset import BASE_COLUMNS, Dataset
from delinquency_eda.errors import DatasetError
from delinquency_eda.report import Doc, fmt_int, fmt_money, fmt_rate
from delinquency_eda.tables import unpaid_rate
from delinquency_eda.textutil import normalise_text

KEY_COLUMNS = ["product_line", "agreement_id", "portfolio_id"]
ADDED_COLUMNS = ("agreement_key", "is_system", "zero_balance", "group_key")


def agreement_key(df: pd.DataFrame) -> pd.Series:
    """The composite agreement key. A row id alone collides across product lines."""
    return (
        df["product_line"].astype(str)
        + "|"
        + df["agreement_id"].astype(str)
        + "|"
        + df["portfolio_id"].astype(str)
    )


def split_group(df: pd.DataFrame) -> pd.Series:
    """The person key, or the debtor row when a slip has none, so keyless slips never pool."""
    fallback = df["product_line"].astype(str) + "|debtor|" + df["debtor_id"].astype(str)
    return df["person_key"].astype(object).where(df["person_key"].notna(), fallback)


def system_flag(df: pd.DataFrame, system_agents: tuple[str, ...]) -> pd.Series:
    """True where the agent name or login is one of ``system_agents`` (form-insensitive)."""
    listed = {normalise_text(a) for a in system_agents}
    if not listed:
        return pd.Series(False, index=df.index)

    def hit(value: object) -> bool:
        return isinstance(value, str) and normalise_text(value) in listed

    return df["agent_name"].map(hit) | df["agent_login"].map(hit)


def clean_frame(dataset: Dataset, settings: Settings) -> pd.DataFrame:
    """The full cleaned table: the analysis frame plus the additive columns. Raises on a bad key."""
    df = F.analysis_frame(dataset, settings)
    df["agreement_key"] = agreement_key(df)
    if not df["agreement_key"].is_unique:
        dupes = int(df["agreement_key"].duplicated().sum())
        raise DatasetError(f"grain violated: {dupes} agreement key(s) appear more than once")
    df["is_system"] = system_flag(df, settings.system_agents)
    df["zero_balance"] = df["original_debt"].notna() & (df["original_debt"] == 0)
    df["group_key"] = split_group(df)
    return df


def analytic_view(df: pd.DataFrame) -> pd.DataFrame:
    """The default analysis view: every row except those flagged as system-owned."""
    return df[~df["is_system"]].copy()


@dataclass
class CleaningSummary:
    rows_before: int
    rows_after_full: int
    columns_before: int
    columns_after: int
    added: list[str]
    system_rows: int
    system_value: float
    zero_balance_rows: int
    ratio_undefined: int
    full_rate: float
    analytic_rate: float
    nulls: dict[str, int]
    key_unique: bool
    groups: int


def summarize(dataset: Dataset, full: pd.DataFrame) -> CleaningSummary:
    analytic = analytic_view(full)
    system = full[full["is_system"]]
    added = [c for c in full.columns if c not in BASE_COLUMNS]
    return CleaningSummary(
        rows_before=len(dataset.base),
        rows_after_full=len(full),
        columns_before=len(BASE_COLUMNS),
        columns_after=full.shape[1],
        added=added,
        system_rows=len(system),
        system_value=float(system["first_installment_value"].sum()),
        zero_balance_rows=int(full["zero_balance"].sum()),
        ratio_undefined=int(full["first_installment_ratio"].isna().sum()),
        full_rate=unpaid_rate(full[full["is_mature"]]),
        analytic_rate=unpaid_rate(analytic[analytic["is_mature"]]),
        nulls={c: int(n) for c, n in full.isna().sum().items() if n},
        key_unique=bool(full["agreement_key"].is_unique),
        groups=int(full["group_key"].nunique()),
    )


def meta_for(dataset: Dataset, settings: Settings) -> dict[str, object]:
    return {
        "target": "bucket",
        "positive_class": "overdue_unpaid",
        "leakage_columns": list(F.LEAKAGE_COLUMNS),
        "explanatory_only_columns": list(F.EXPLANATORY_ONLY_COLUMNS),
        "key": KEY_COLUMNS,
        "group_for_split": "group_key",
        "maturity_flag": "is_mature",
        "system_agents": list(settings.system_agents),
        "period": [dataset.meta.period_from, dataset.meta.period_to],
        "as_of": dataset.meta.as_of,
        "grace_days": dataset.meta.grace_days,
    }


def render(summary: CleaningSummary) -> Doc:
    s = summary
    delta = 100 * (s.analytic_rate - s.full_rate)
    doc = Doc().h1("Cleaning summary")
    doc.p("Rules R1-R7 are additive. See the module documentation for what each one does.")
    doc.h2("Information loss")
    doc.table(
        [
            ["rows removed", 0],
            ["cells imputed", 0],
            ["columns removed", 0],
            ["shape before", f"{s.rows_before} rows x {s.columns_before} columns"],
            ["shape after (full)", f"{s.rows_after_full} rows x {s.columns_after} columns"],
            ["columns added", ", ".join(f"`{c}`" for c in s.added)],
            ["rows flagged `is_system`", f"{fmt_int(s.system_rows)} ({fmt_money(s.system_value)})"],
            ["rows flagged `zero_balance`", fmt_int(s.zero_balance_rows)],
            ["instalment ratio undefined (NaN, not imputed)", fmt_int(s.ratio_undefined)],
            ["leakage columns kept, marked in the meta file", len(F.LEAKAGE_COLUMNS)],
        ],
        ["item", "value"],
    )
    doc.h2("Distribution shift from the analytic view")
    doc.table(
        [
            ["full table", fmt_rate(s.full_rate)],
            ["analytic view (system rows removed)", fmt_rate(s.analytic_rate)],
            ["difference", "-" if math.isnan(delta) else f"{delta:+.1f} points"],
        ],
        ["unpaid rate on mature rows", "value"],
    )
    doc.h2("Validation")
    doc.bullets(
        [
            f"Agreement key unique: {'yes' if s.key_unique else 'NO'}.",
            f"Distinct split groups: {fmt_int(s.groups)} for {fmt_int(s.rows_after_full)} rows.",
            "Columns that still hold nulls (expected, not imputed): "
            + (", ".join(f"`{c}` ({n})" for c, n in s.nulls.items()) or "none")
            + ".",
        ]
    )
    return doc


def add_arguments(parser: argparse.ArgumentParser) -> None:
    del parser  # no options of its own; reads the dataset


def run(ctx: Context, args: argparse.Namespace) -> int:
    del args
    dataset = ctx.load_dataset()
    full = clean_frame(dataset, ctx.settings)
    analytic = analytic_view(full)
    out = ctx.out_dir
    out.mkdir(parents=True, exist_ok=True)
    full.to_csv(out / "clean_full.csv", index=False, encoding="utf-8", lineterminator="\n")
    analytic.to_csv(out / "clean_analytic.csv", index=False, encoding="utf-8", lineterminator="\n")
    (out / "clean_meta.json").write_text(
        json.dumps(meta_for(dataset, ctx.settings), indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    summary = summarize(dataset, full)
    ctx.say(
        f"rows: full={len(full)} analytic={len(analytic)}; flagged system={summary.system_rows}, "
        f"zero balance={summary.zero_balance_rows}"
    )
    ctx.say(f"tables written: {out / 'clean_full.csv'}, {out / 'clean_analytic.csv'}")
    ctx.write_report("clean.md", render(summary))
    return 0
