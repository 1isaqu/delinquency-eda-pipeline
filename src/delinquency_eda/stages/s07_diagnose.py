"""Stage 07 - cleaning diagnostics. Quantifies problems, changes nothing.

The order is the one in which each step depends on the one before: structural, types,
duplicates, missing values, invalid or inconsistent values, outliers. Nothing is written except
the report, so this stage is safe to run on any dataset and to re-run after the data changes.

What it deliberately does not do:

* it does not validate identifier formats. A person key is an opaque token supplied by the user;
  the only identifier check is that it stays text, because a key that looks numeric loses leading
  zeros the moment a spreadsheet or a dtype guess touches it,
* it does not call a rare value an error. A tail of small categories or a long right tail on money
  is a property of the data, and the report says so instead of proposing a repair.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from delinquency_eda import features as F
from delinquency_eda.config import Settings
from delinquency_eda.context import Context
from delinquency_eda.dataset import BASE_COLUMNS
from delinquency_eda.derive import OVERDUE_UNPAID
from delinquency_eda.report import Doc, fmt_int, fmt_money, fmt_rate
from delinquency_eda.tables import unpaid_rate
from delinquency_eda.textutil import normalize_text

TEXT_COLUMNS = (
    "agent_id",
    "agent_name",
    "agent_login",
    "portfolio_label",
    "person_key",
    "status_label",
)
DATE_COLUMNS = ("agreement_date", "due_date", "paid_date")
NAME_COLUMNS = ("agent_name", "agent_login", "portfolio_label")
LONG_TAIL_COLUMNS = ("first_installment_value", "total_value", "original_debt", "days_to_due")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


@dataclass
class Diagnosis:
    rows: int
    columns: int
    whitespace: dict[str, int]
    control_chars: dict[str, int]
    truncation: pd.DataFrame
    date_failures: dict[str, int]
    numeric_like_keys: int
    leading_zero_keys: int
    duplicates: dict[str, int]
    missing: pd.DataFrame
    zero_balance: int
    negative_balance: int
    variants: dict[str, list[list[str]]]
    agent_conflicts: tuple[int, int]
    suspects: pd.DataFrame
    outliers: pd.DataFrame


def _text_values(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].dropna().astype(str)


def whitespace_counts(df: pd.DataFrame) -> dict[str, int]:
    out = {}
    for c in (c for c in TEXT_COLUMNS if c in df.columns):
        s = _text_values(df, c)
        n = int((s != s.str.strip()).sum())
        if n:
            out[c] = n
    return out


def control_counts(df: pd.DataFrame) -> dict[str, int]:
    out = {}
    for c in (c for c in TEXT_COLUMNS if c in df.columns):
        n = int(_text_values(df, c).map(lambda v: bool(_CONTROL.search(v))).sum())
        if n:
            out[c] = n
    return out


def truncation_suspects(df: pd.DataFrame) -> pd.DataFrame:
    """Names stuck at one exact length. Several *different* values sharing the maximum length
    means the source column probably cuts them off; a single repeated value does not."""
    rows = []
    for c in (c for c in NAME_COLUMNS if c in df.columns):
        s = _text_values(df, c)
        if s.empty:
            continue
        length = s.str.len()
        top = int(length.max())
        at_top = s[length == top]
        rows.append(
            {
                "column": c,
                "max length": top,
                "rows at max": len(at_top),
                "distinct at max": at_top.nunique(),
                "suspect": at_top.nunique() >= 2,
            }
        )
    return pd.DataFrame(
        rows, columns=["column", "max length", "rows at max", "distinct at max", "suspect"]
    )


def date_failures(df: pd.DataFrame) -> dict[str, int]:
    """Values present but not readable as ISO dates (after accounting for real nulls)."""
    out = {}
    for c in (c for c in DATE_COLUMNS if c in df.columns):
        coerced = pd.to_datetime(df[c], errors="coerce", format="%Y-%m-%d")
        out[c] = int(coerced.isna().sum() - df[c].isna().sum())
    return out


def key_shape(df: pd.DataFrame) -> tuple[int, int]:
    """Person keys that look like plain numbers, and how many of those start with a zero."""
    keys = _text_values(df, "person_key")
    numeric = keys[keys.str.fullmatch(r"\d+")]
    return len(numeric), int(numeric.str.startswith("0").sum())


def duplicate_counts(df: pd.DataFrame) -> dict[str, int]:
    return {
        "whole row": int(df.duplicated().sum()),
        "(product_line, record_id)": int(df.duplicated(["product_line", "record_id"]).sum()),
        "(product_line, agreement_id, portfolio_id)": int(
            df.duplicated(["product_line", "agreement_id", "portfolio_id"]).sum()
        ),
        "record_id alone (collides across product lines)": int(df["record_id"].duplicated().sum()),
    }


def missing_table(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Nulls per column and whether the null itself predicts the outcome (MNAR check)."""
    mature = df[df["is_mature"]]
    rows = []
    for c in BASE_COLUMNS:
        n = int(df[c].isna().sum())
        if not n:
            continue
        missing_flag = mature[c].isna()
        present, absent = unpaid_rate(mature[~missing_flag]), unpaid_rate(mature[missing_flag])
        gap = 100 * (absent - present) if not (np.isnan(present) or np.isnan(absent)) else np.nan
        verdict = "n/a (one group has no mature slips)"
        if not np.isnan(gap):
            verdict = (
                "the null carries signal (not missing at random)"
                if abs(gap) > settings.mnar_gap_pp
                else "no apparent signal"
            )
        rows.append(
            {
                "column": c,
                "nulls": n,
                "% null": 100 * n / len(df),
                "unpaid rate if present": present,
                "unpaid rate if null": absent,
                "gap (pp)": gap,
                "reading": verdict,
            }
        )
    return pd.DataFrame(rows)


def form_variants(df: pd.DataFrame) -> dict[str, list[list[str]]]:
    """Groups of values that differ only in case, accents or spacing."""
    out = {}
    for c in (c for c in NAME_COLUMNS if c in df.columns):
        groups: dict[str, list[str]] = {}
        for value in _text_values(df, c).unique():
            groups.setdefault(normalize_text(value), []).append(value)
        out[c] = [sorted(g) for g in groups.values() if len(g) > 1]
    return out


def agent_mapping_conflicts(df: pd.DataFrame) -> tuple[int, int]:
    """Agents with more than one login, and logins with more than one agent name."""
    pairs = df[["agent_name", "agent_login"]].dropna()
    return (
        int((pairs.groupby("agent_name")["agent_login"].nunique() > 1).sum()),
        int((pairs.groupby("agent_login")["agent_name"].nunique() > 1).sum()),
    )


def suspect_agents(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """Agents whose name or login looks like a system account but passed the extraction filter."""
    rx = re.compile(pattern, re.IGNORECASE)
    hit = df["agent_name"].fillna("").map(lambda v: bool(rx.search(v))) | df["agent_login"].fillna(
        ""
    ).map(lambda v: bool(rx.search(v)))
    sub = df[hit]
    if sub.empty:
        return pd.DataFrame(columns=["agent_name", "agent_login", "slips", "unpaid", "value"])
    out = sub.groupby(["agent_name", "agent_login"], dropna=False).agg(
        slips=("record_id", "size"),
        unpaid=("bucket", lambda s: int((s == OVERDUE_UNPAID).sum())),
        value=("first_installment_value", "sum"),
    )
    return out.reset_index()


def outlier_table(df: pd.DataFrame, k: float) -> pd.DataFrame:
    rows = []
    for c in (c for c in LONG_TAIL_COLUMNS if c in df.columns):
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        fence = q3 + k * (q3 - q1)
        beyond = s[s > fence]
        rows.append(
            {
                "column": c,
                "above fence": len(beyond),
                "% of column": 100 * len(beyond) / len(s),
                "fence": fence,
                "maximum": s.max(),
            }
        )
    return pd.DataFrame(rows, columns=["column", "above fence", "% of column", "fence", "maximum"])


def diagnose(df: pd.DataFrame, settings: Settings) -> Diagnosis:
    """Every diagnostic over the base table. ``df`` comes from ``features.analysis_frame``."""
    numeric_keys, leading_zero = key_shape(df)
    balance = df["original_debt"]
    return Diagnosis(
        rows=len(df),
        columns=len(BASE_COLUMNS),
        whitespace=whitespace_counts(df),
        control_chars=control_counts(df),
        truncation=truncation_suspects(df),
        date_failures=date_failures(df),
        numeric_like_keys=numeric_keys,
        leading_zero_keys=leading_zero,
        duplicates=duplicate_counts(df),
        missing=missing_table(df, settings),
        zero_balance=int((balance == 0).sum()),
        negative_balance=int((balance < 0).sum()),
        variants=form_variants(df),
        agent_conflicts=agent_mapping_conflicts(df),
        suspects=suspect_agents(df, settings.suspect_agent_pattern),
        outliers=outlier_table(df, settings.outlier_iqr_k),
    )


def render(d: Diagnosis, settings: Settings) -> Doc:
    doc = Doc().h1("Cleaning diagnostics")
    doc.p(f"{fmt_int(d.rows)} rows, {d.columns} columns. Diagnosis only: nothing was changed.")
    doc.h2("1. Structural")
    doc.bullets(
        [
            f"Leading or trailing whitespace: {d.whitespace or 'none'}.",
            f"Control characters: {d.control_chars or 'none'}.",
        ]
    )
    doc.table(d.truncation, float_format="{:,.0f}")
    doc.p(
        "A name column whose longest values are many *different* strings of one exact length is "
        "probably cut off by the source column width. That is a property of the source: it only "
        "affects labels, so report it rather than repair it."
    )
    doc.h2("2. Types")
    doc.table(
        [[c, n] for c, n in d.date_failures.items()], ["date column", "values that fail to parse"]
    )
    doc.p(
        f"Person keys that look like plain numbers: {d.numeric_like_keys}, of which "
        f"{d.leading_zero_keys} start with a zero. Keep person keys as text; a numeric dtype "
        "would silently drop those zeros and merge different people."
    )
    doc.h2("3. Duplicates")
    doc.table([[k, fmt_int(v)] for k, v in d.duplicates.items()], ["key", "extra rows"])
    doc.p(
        "A repeated person across slips is legitimate (one person, several agreements) and is "
        "handled in the split, not by deduplicating."
    )
    doc.h2("4. Missing values")
    _missing(doc, d.missing)
    doc.h2("5. Invalid and inconsistent values")
    doc.bullets(
        [
            f"`original_debt` equal to zero: {d.zero_balance} (a real zero balance, not missing).",
            f"`original_debt` negative: {d.negative_balance}.",
            f"Agents with more than one login: {d.agent_conflicts[0]}; logins with more than one "
            f"agent name: {d.agent_conflicts[1]}.",
        ]
    )
    for column, groups in d.variants.items():
        doc.p(
            f"`{column}`: {len(groups)} group(s) of values that differ only in case, accents "
            "or spacing."
        )
        if groups:
            doc.bullets(str(g) for g in groups[:5])
    _suspects(doc, d.suspects, settings)
    doc.h2(f"6. Outliers (IQR fence, k = {settings.outlier_iqr_k:g})")
    doc.table(d.outliers, float_format="{:,.2f}")
    doc.p(
        "A long right tail makes an IQR fence flag many legitimate values. Keep them; if a "
        "linear model needs them, use log1p, not capping."
    )
    doc.rule()
    doc.p("Diagnosis finished. Nothing was changed or written besides this report.")
    return doc


def _missing(doc: Doc, table: pd.DataFrame) -> None:
    if table.empty:
        doc.p("No column has nulls.")
        return
    doc.table(
        [
            [
                r["column"],
                fmt_int(r["nulls"]),
                f"{r['% null']:.1f}%",
                fmt_rate(r["unpaid rate if present"]),
                fmt_rate(r["unpaid rate if null"]),
                "-" if np.isnan(r["gap (pp)"]) else f"{r['gap (pp)']:+.1f}",
                r["reading"],
            ]
            for r in table.to_dict("records")
        ],
        ["column", "nulls", "% null", "unpaid if present", "unpaid if null", "gap (pp)", "reading"],
    )


def _suspects(doc: Doc, table: pd.DataFrame, settings: Settings) -> None:
    doc.p(
        f"Agents matching the suspect-account pattern `{settings.suspect_agent_pattern}` that "
        "passed the extraction filter:"
    )
    if table.empty:
        doc.p("none")
        return
    doc.table(
        [
            [r.agent_name, r.agent_login, fmt_int(r.slips), fmt_int(r.unpaid), fmt_money(r.value)]
            for r in table.itertuples(index=False)
        ],
        ["agent", "login", "slips", "unpaid", "first-installment value"],
    )
    doc.p(
        "If these are system accounts, list them in `system_agents`: the cleaning stage flags "
        "them instead of deleting them."
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    del parser  # no options of its own; reads the dataset


def run(ctx: Context, args: argparse.Namespace) -> int:
    del args
    dataset = ctx.load_dataset()
    df = F.analysis_frame(dataset, ctx.settings)
    result = diagnose(df, ctx.settings)
    ctx.say(
        f"rows={result.rows} duplicates={sum(result.duplicates.values())} "
        f"zero_balance={result.zero_balance} suspect_agents={len(result.suspects)}"
    )
    ctx.write_report("diagnose.md", render(result, ctx.settings))
    return 0
