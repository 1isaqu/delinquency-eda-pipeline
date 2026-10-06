"""Stage 06 - EDA report generator.

Reads the dataset bundle and writes one Markdown report that covers, in order: structure, column
profile, the target, numeric statistics, data-quality checks (with two deep dives: zero balances
and target leakage), cardinality, referential integrity, time coverage, correlation with the
outcome, outliers, findings and recommendations.

Nothing in the narrative is typed in advance. Every finding is a sentence built from numbers
computed on the dataset at hand, and a recommendation only appears when its trigger fired. The
report never prints example values from a column, so it carries no row-level content.

Rates use the mature cohort (outcome final); see stage 05 for why.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from delinquency_eda import features as F
from delinquency_eda.config import Settings
from delinquency_eda.context import Context
from delinquency_eda.dataset import Dataset
from delinquency_eda.derive import OVERDUE_UNPAID
from delinquency_eda.report import Doc, fmt_int, fmt_rate
from delinquency_eda.tables import unpaid_rate

NUMERIC = (
    "first_installment_value",
    "total_value",
    "original_debt",
    "total_installments",
    "plan_installments",
    "days_to_due",
    "days_overdue",
    "paid_amount",
)
CATEGORICAL = (
    "product_line",
    "agent_id",
    "portfolio_label",
    "portfolio_id",
    "status_label",
    "bucket",
)
MONEY = ("first_installment_value", "total_value", "original_debt", "paid_amount")
HIGH, MEDIUM, INFO = "high", "medium", "info"
PERCENTILES = [0.01, 0.25, 0.5, 0.75, 0.95, 0.99]


@dataclass(frozen=True)
class Check:
    name: str
    rows: int
    share: float
    severity: str
    note: str


# --------------------------------------------------------------------------------------------
# Computation
# --------------------------------------------------------------------------------------------


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column type, nulls and distinct count. No example values, on purpose."""
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "nulls": [int(df[c].isna().sum()) for c in df.columns],
            "% null": [100 * df[c].isna().mean() if len(df) else 0.0 for c in df.columns],
            "distinct": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )


def quality_checks(df: pd.DataFrame) -> list[Check]:
    """Rule-based checks over the base table, each with a severity and a one-line meaning."""
    debt, first, total = df["original_debt"], df["first_installment_value"], df["total_value"]
    same_line = df.groupby("record_id")["product_line"].transform("nunique") > 1
    specs: list[tuple[str, pd.Series, str, str]] = [
        ("original_debt missing", debt.isna(), MEDIUM, "no linked debt was found"),
        (
            "original_debt equals zero",
            debt.notna() & (debt == 0),
            HIGH,
            "debt linked but balance zero: the instalment ratio divides by zero, it is not missing",
        ),
        ("original_debt negative", debt < 0, HIGH, "negative balance"),
        (
            "portfolio_label missing",
            df["portfolio_label"].isna(),
            MEDIUM,
            "the label lookup found nothing",
        ),
        (
            "person_key missing",
            df["person_key"].isna(),
            MEDIUM,
            "history and follow-up cannot be linked for these slips",
        ),
        ("first_installment_value <= 0", first <= 0, HIGH, "slip with nothing to collect"),
        (
            "first installment above original debt",
            (debt > 0) & (first > debt),
            MEDIUM,
            "interest and fees, or a stale balance",
        ),
        (
            "total_value below first_installment_value",
            total < first,
            HIGH,
            "installments sum to less than the first one: aggregation inconsistency",
        ),
        ("days_to_due negative", df["days_to_due"] < 0, HIGH, "due date before the agreement"),
        (
            "money received on an unpaid slip",
            (df["bucket"] == OVERDUE_UNPAID) & (df["paid_amount"] > 0),
            INFO,
            "paid, but after the grace period: lost by timing, not by value",
        ),
        (
            "plan_installments differs from generated installments",
            df["plan_installments"].notna() & (df["plan_installments"] != df["total_installments"]),
            MEDIUM,
            "declared plan differs from the installments actually generated",
        ),
        (
            "duplicate (product_line, record_id)",
            df.duplicated(["product_line", "record_id"]),
            HIGH,
            "grain violated",
        ),
        (
            "duplicate (product_line, agreement_id, portfolio_id)",
            df.duplicated(["product_line", "agreement_id", "portfolio_id"]),
            HIGH,
            "more than one first installment for the same agreement",
        ),
        (
            "record_id shared by two product lines",
            same_line,
            INFO,
            "expected: row ids are sequences per database, never a key on their own",
        ),
    ]
    n = len(df)
    return [
        Check(name, int(m.sum()), float(m.sum()) / n if n else 0.0, sev, note)
        for name, m, sev, note in specs
    ]


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in (c for c in CATEGORICAL if c in df.columns):
        counts = df[column].value_counts(dropna=True)
        rows.append(
            {
                "column": column,
                "distinct": int(df[column].nunique(dropna=True)),
                "mode": str(counts.index[0]) if len(counts) else "",
                "% in mode": 100 * counts.iloc[0] / len(df) if len(counts) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def rare_categories(series: pd.Series, minimum: int) -> tuple[int, int, float]:
    """``(categories, categories below minimum, share of rows they cover)``."""
    counts = series.value_counts(dropna=True)
    small = counts[counts < minimum]
    return len(counts), len(small), float(small.sum()) / len(series) if len(series) else 0.0


def orphans(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> tuple[int, int]:
    """Rows of ``left`` whose key tuple does / does not occur in ``right``."""
    if left.empty:
        return 0, 0
    known = set(right[keys].itertuples(index=False, name=None))
    hits = [t in known for t in left[keys].itertuples(index=False, name=None)]
    return int(sum(hits)), int(len(hits) - sum(hits))


def nullness_accuracy(mature: pd.DataFrame) -> float:
    """How well "no payment amount" alone classifies the outcome: the leakage in one number."""
    if mature.empty:
        return float("nan")
    agree = float((mature["paid_amount"].isna() == (mature["bucket"] == OVERDUE_UNPAID)).mean())
    return max(agree, 1 - agree)


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    month = pd.to_datetime(df["due_date"]).dt.to_period("M").astype(str)
    frame = df.assign(month=month, unpaid_mature=df["is_mature"] & (df["bucket"] == OVERDUE_UNPAID))
    out = frame.groupby("month").agg(
        slips=("record_id", "size"),
        mature=("is_mature", "sum"),
        unpaid=("unpaid_mature", "sum"),
        value=("first_installment_value", "sum"),
    )
    out["rate"] = out["unpaid"] / out["mature"].where(out["mature"] > 0)
    return out.reset_index()


def stability_range(months: pd.DataFrame, minimum: int) -> float | None:
    """Spread, in points, of the unpaid rate across months that have enough mature slips."""
    usable = months[months["mature"] >= minimum]["rate"].dropna()
    return None if len(usable) < 2 else float(100 * (usable.max() - usable.min()))


def correlations(mature: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of each numeric column with non-payment, flagged when it leaks."""
    cols = [c for c in NUMERIC if c in mature.columns]
    target = (mature["bucket"] == OVERDUE_UNPAID).astype(int)
    rows = []
    for c in cols:
        value = mature[c].corr(target, method="spearman") if mature[c].nunique() > 1 else np.nan
        rows.append({"column": c, "spearman": value, "leaks": c in F.LEAKAGE_COLUMNS})
    out = pd.DataFrame(rows)
    return out.reindex(out["spearman"].abs().sort_values(ascending=False, na_position="last").index)


def collinear_pairs(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    cols = [c for c in NUMERIC if c in df.columns and c not in F.LEAKAGE_COLUMNS]
    corr = df[cols].corr(method="spearman").abs()
    pairs = [
        (a, b, corr.loc[a, b])
        for i, a in enumerate(cols)
        for b in cols[i + 1 :]
        if corr.loc[a, b] > threshold
    ]
    return pd.DataFrame(pairs, columns=["column a", "column b", "|spearman|"]).sort_values(
        "|spearman|", ascending=False
    )


def outliers(df: pd.DataFrame, k: float) -> pd.DataFrame:
    rows = []
    for c in (c for c in NUMERIC if c in df.columns):
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        hi = q3 + k * (q3 - q1)
        beyond = s[s > hi]
        rows.append(
            {
                "column": c,
                "above fence": len(beyond),
                "% of column": 100 * len(beyond) / len(s),
                "upper fence": hi,
                "maximum": s.max(),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def _basics(doc: Doc, dataset: Dataset, file_size: int | None) -> None:
    meta = dataset.meta
    size = "-" if file_size is None else f"{file_size:,} bytes ({file_size / 1024**2:.1f} MB)"
    doc.h1("Exploratory data analysis")
    doc.h2("1. Basic information")
    doc.table(
        [
            ["file size", size],
            ["format", "JSON, one object with a meta block and four tables"],
            ["due-date window", f"{meta.period_from} to {meta.period_to}"],
            ["reference date", meta.as_of],
            ["grace days", meta.grace_days],
            ["product lines", ", ".join(meta.product_lines)],
        ],
        ["property", "value"],
    )


def _structure(doc: Doc, dataset: Dataset, df: pd.DataFrame) -> None:
    doc.h2("2. Structure")
    frames = {
        "base": (dataset.base_frame(), "one row per first-installment slip; carries the outcome"),
        "debtors": (dataset.debtors_frame(), "debtor rows that share a person key with a slip"),
        "contacts": (dataset.contacts_frame(), "one row per outreach attempt in the window"),
        "breaks": (dataset.breaks_frame(), "earlier first-installment agreements that broke"),
    }
    doc.table(
        [[name, len(f), f.shape[1], role] for name, (f, role) in frames.items()],
        ["table", "rows", "columns", "role"],
    )
    doc.p(f"The base table is profiled below ({len(df)} rows).")


def _target(doc: Doc, df: pd.DataFrame) -> None:
    doc.h2("4. Outcome")
    counts = df["bucket"].value_counts()
    doc.table(
        [
            [
                b,
                fmt_int(n),
                fmt_rate(n / len(df)),
                f"{df.loc[df['bucket'] == b, 'first_installment_value'].sum():,.2f}",
            ]
            for b, n in counts.items()
        ],
        ["bucket", "slips", "share", "first-installment value"],
    )
    mature = df[df["is_mature"]]
    rate = unpaid_rate(mature)
    doc.p(
        f"Mature cohort: {fmt_int(len(mature))} slips, unpaid rate {fmt_rate(rate)}. "
        + (
            "Predicting the majority class everywhere would be right "
            f"{fmt_rate(max(rate, 1 - rate))}"
            " of the time, so raw accuracy is not a useful metric here."
            if len(mature)
            else "There are no mature slips yet."
        )
    )


def _numeric(doc: Doc, df: pd.DataFrame) -> None:
    doc.h2("5. Numeric columns")
    cols = [c for c in NUMERIC if c in df.columns]
    desc = df[cols].describe(percentiles=PERCENTILES).T
    desc.insert(0, "column", desc.index)
    desc["count"] = desc["count"].astype(int)
    doc.table(desc.reset_index(drop=True))
    skew = df[cols].skew().dropna().sort_values(ascending=False)
    doc.p("Most skewed columns, with median against the 99th percentile:")
    lines = []
    for c in skew.index[:3]:
        q50, q99 = df[c].quantile([0.5, 0.99])
        lines.append(f"`{c}`: skew {skew[c]:.1f}, median {q50:,.2f}, p99 {q99:,.2f}")
    doc.bullets(lines or ["not enough data to measure skew"])


def _quality(doc: Doc, df: pd.DataFrame, checks: list[Check], settings: Settings) -> None:
    doc.h2("6. Data quality")
    doc.table(
        [[c.name, fmt_int(c.rows), fmt_rate(c.share), c.severity, c.note] for c in checks],
        ["check", "rows", "% of base", "severity", "meaning"],
    )
    _zero_balance(doc, df, settings)
    _leakage(doc, df)


def _zero_balance(doc: Doc, df: pd.DataFrame, settings: Settings) -> None:
    doc.h3("6.1 A zero balance is a value, not a gap")
    zero = df["original_debt"] == 0
    by_line = df.assign(zero=zero).groupby("product_line")["zero"].agg(["size", "mean"])
    doc.table(
        [[line, fmt_int(r["size"]), fmt_rate(r["mean"])] for line, r in by_line.iterrows()],
        ["product line", "slips", "share with zero balance"],
    )
    mature = df[df["is_mature"]]
    with_balance = unpaid_rate(mature[mature["original_debt"] > 0])
    with_zero = unpaid_rate(mature[mature["original_debt"] == 0])
    if np.isnan(with_balance) or np.isnan(with_zero):
        doc.p("One of the two groups has no mature slips, so the rates cannot be compared.")
        return
    gap = 100 * (with_zero - with_balance)
    verdict = (
        "The gap is large enough to carry signal: keep the zero as its own category."
        if abs(gap) > settings.mnar_gap_pp
        else "The gap is small, but keep the zero flagged anyway: imputing it invents a number."
    )
    doc.p(
        f"Unpaid rate (mature): {fmt_rate(with_balance)} with a positive balance, "
        f"{fmt_rate(with_zero)} with a zero balance ({gap:+.1f} points). {verdict} Code that tests "
        "`if original_debt:` treats zero as falsy and files these rows with the missing ones."
    )


def _leakage(doc: Doc, df: pd.DataFrame) -> None:
    doc.h3("6.2 Target leakage")
    mature = df[df["is_mature"]]
    table = pd.crosstab(mature["paid_amount"].isna(), mature["bucket"])
    table.insert(0, "paid_amount missing", table.index.astype(str))
    doc.table(table.reset_index(drop=True))
    accuracy = nullness_accuracy(mature)
    doc.p(
        f"`paid_amount` and `paid_date` exist only because money arrived. On mature slips, "
        f"whether `paid_amount` is missing classifies the outcome correctly {fmt_rate(accuracy)} "
        f"of the time on its own. Columns measured after the outcome "
        f"({', '.join(f'`{c}`' for c in F.LEAKAGE_COLUMNS)}) must stay out of any predictive model."
    )


def _cardinality(doc: Doc, df: pd.DataFrame, settings: Settings) -> None:
    doc.h2("7. Categorical columns")
    doc.table(categorical_summary(df), float_format="{:.1f}")
    lines = []
    for column in ("agent_id", "portfolio_label"):
        total, small, share = rare_categories(df[column], settings.rare_category_min_n)
        lines.append(
            f"`{column}`: {total} categories, {small} with fewer than "
            f"{settings.rare_category_min_n} slips, covering {fmt_rate(share)} of rows"
        )
    doc.bullets(lines)


def _integrity(doc: Doc, dataset: Dataset, df: pd.DataFrame) -> None:
    doc.h2("8. Referential integrity")
    debtors = dataset.debtors_frame()
    keyed = df[df["person_key"].notna()]
    rows = [
        ("base.person_key -> debtors", *orphans(keyed, debtors, ["person_key"])),
        ("base debtor -> debtors", *orphans(df, debtors, ["product_line", "debtor_id"])),
        (
            "contacts debtor -> debtors",
            *orphans(dataset.contacts_frame(), debtors, ["product_line", "debtor_id"]),
        ),
        (
            "breaks debtor -> debtors",
            *orphans(dataset.breaks_frame(), debtors, ["product_line", "debtor_id"]),
        ),
    ]
    doc.table(
        [
            [name, fmt_int(ok), fmt_int(bad), fmt_rate(bad / (ok + bad) if ok + bad else 0.0)]
            for name, ok, bad in rows
        ],
        ["relation", "matched", "orphans", "% orphans"],
    )
    per_person = keyed.groupby("person_key").size()
    doc.p(
        f"{fmt_int(df['person_key'].nunique())} people for {fmt_int(len(df))} slips; "
        f"{fmt_int((per_person > 1).sum())} people appear more than once (at most "
        f"{int(per_person.max()) if len(per_person) else 0} slips for one person). Rows of one "
        "person are not independent: a random train/test split leaks people across the "
        "partitions. Split by person, or by time."
    )


def _time(doc: Doc, df: pd.DataFrame, settings: Settings) -> None:
    doc.h2("9. Time coverage")
    months = monthly(df)
    doc.table(
        [
            [
                r.month,
                fmt_int(r.slips),
                fmt_int(r.mature),
                fmt_int(r.unpaid),
                fmt_rate(r.rate),
                f"{r.value:,.2f}",
            ]
            for r in months.itertuples(index=False)
        ],
        [
            "due month",
            "slips",
            "mature",
            "unpaid (mature)",
            "unpaid rate",
            "first-installment value",
        ],
    )
    spread = stability_range(months, settings.min_n_reliable)
    if spread is None:
        doc.p("Fewer than two months have enough mature slips to judge stability.")
    elif spread > settings.stability_range_pp:
        doc.p(
            f"The unpaid rate moves {spread:.1f} points across months: pooling the months "
            "hides a change of regime. Validate any model out of time."
        )
    else:
        doc.p(f"The unpaid rate moves {spread:.1f} points across months: stable enough to pool.")


def _correlation(doc: Doc, df: pd.DataFrame) -> None:
    doc.h2("10. Correlation with non-payment")
    mature = df[df["is_mature"]]
    table = correlations(mature)
    doc.table(
        [
            [
                r.column,
                "-" if pd.isna(r.spearman) else f"{r.spearman:+.3f}",
                "yes" if r.leaks else "no",
            ]
            for r in table.itertuples(index=False)
        ],
        ["column", "Spearman vs non-payment (mature)", "leaks the outcome"],
    )
    doc.p("Spearman is monotonic and robust to the long tails these columns have.")
    pairs = collinear_pairs(df)
    doc.p("Collinear predictors (|Spearman| above 0.5, leakage columns excluded):")
    if len(pairs):
        doc.table(pairs, float_format="{:.3f}")
    else:
        doc.p("none")


def _outliers(doc: Doc, df: pd.DataFrame, settings: Settings) -> None:
    doc.h2(f"11. Outliers (IQR fence, k = {settings.outlier_iqr_k:g})")
    doc.table(outliers(df, settings.outlier_iqr_k), float_format="{:,.2f}")
    top = df.nlargest(5, "first_installment_value")[
        [
            "product_line",
            "portfolio_group",
            "first_installment_value",
            "total_value",
            "original_debt",
            "bucket",
        ]
    ]
    doc.p("The five largest first installments (no identifiers shown):")
    doc.table(top)
    doc.p(
        "Large values on long-tailed money columns are usually real. Check them against the "
        "debt before treating them as errors; winsorise only for a model that is scale sensitive."
    )


def _findings(df: pd.DataFrame, checks: list[Check], settings: Settings) -> list[str]:
    by_name = {c.name: c for c in checks}
    mature = df[df["is_mature"]]
    out = []
    grain = by_name["duplicate (product_line, record_id)"].rows
    agreement = by_name["duplicate (product_line, agreement_id, portfolio_id)"].rows
    out.append(f"Grain: {grain} duplicate row ids and {agreement} duplicate agreements.")
    zero = by_name["original_debt equals zero"]
    out.append(f"{fmt_rate(zero.share)} of slips have a zero original debt ({zero.rows} rows).")
    out.append(
        "The payment amount's presence classifies the mature outcome "
        f"{fmt_rate(nullness_accuracy(mature))} of the time: it leaks the target."
    )
    skewed = [c for c in MONEY if c in df.columns and df[c].skew() > 2]
    if skewed:
        out.append(
            "Money columns with a long right tail: " + ", ".join(f"`{c}`" for c in skewed) + "."
        )
    keyed = df[df["person_key"].notna()]
    repeated = int((keyed.groupby("person_key").size() > 1).sum())
    out.append(f"{repeated} people appear on more than one slip.")
    corr = correlations(mature)
    usable = corr[~corr["leaks"] & corr["spearman"].notna()]
    if len(usable):
        top = usable.iloc[0]
        out.append(
            f"Strongest non-leaking monotonic association: `{top['column']}` "
            f"({top['spearman']:+.3f})."
        )
    _, small, share = rare_categories(df["portfolio_label"], settings.rare_category_min_n)
    out.append(
        f"{small} portfolio labels are below the minimum sample size ({fmt_rate(share)} of rows)."
    )
    return out


def _recommendations(doc: Doc, df: pd.DataFrame, checks: list[Check], settings: Settings) -> None:
    by_name = {c.name: c for c in checks}
    rules = [
        (
            by_name["original_debt equals zero"].rows > 0,
            "zero original debt",
            "flag it and keep it as its own category; do not impute or drop",
        ),
        (
            True,
            "payment amount and date, days overdue, status",
            "exclude from any predictive model (they are measured after the outcome)",
        ),
        (
            by_name["portfolio_label missing"].rows > 0,
            "missing portfolio label",
            "use an explicit '(no portfolio)' category",
        ),
        (
            any(df[c].skew() > 2 for c in MONEY if c in df.columns),
            "long-tailed money columns",
            "log1p before a linear model; trees do not need it",
        ),
        (
            rare_categories(df["agent_id"], settings.rare_category_min_n)[1] > 0,
            "rare agents and portfolios",
            "group the tail (portfolio group, not 'other'); show n and a sample-size flag",
        ),
        (
            True,
            "term (days to first due date)",
            "bin it; the signal is in the bands, not the raw number",
        ),
        (True, "split", "group by person or split by time, never a random row split"),
        (
            df["product_line"].nunique() > 1,
            "product lines",
            "model them separately or interact with them: their populations differ",
        ),
    ]
    fired = [(what, action) for ok, what, action in rules if ok]
    doc.table(fired, ["data", "action"])


def render(
    dataset: Dataset, df: pd.DataFrame, settings: Settings, file_size: int | None = None
) -> Doc:
    """The full report. ``df`` comes from ``features.analysis_frame``."""
    checks = quality_checks(df)
    doc = Doc()
    _basics(doc, dataset, file_size)
    _structure(doc, dataset, df)
    doc.h2("3. Base table profile")
    doc.table(profile(dataset.base_frame()), float_format="{:.1f}")
    _target(doc, df)
    _numeric(doc, df)
    _quality(doc, df, checks, settings)
    _cardinality(doc, df, settings)
    _integrity(doc, dataset, df)
    _time(doc, df, settings)
    _correlation(doc, df)
    _outliers(doc, df, settings)
    doc.h2("12. Findings")
    doc.numbered(_findings(df, checks, settings))
    doc.h2("13. Recommendations")
    _recommendations(doc, df, checks, settings)
    doc.p(
        "Next analyses worth running: a logistic regression or boosted model of non-payment on "
        "the term band, product line and portfolio group (stage 10) to separate the term from "
        "the mix; a time-to-payment curve after the due date to see where the grace period "
        "should be cut; and a month-by-month stability check before pooling months."
    )
    doc.h2("14. Limits of this data")
    doc.bullets(
        [
            "There is no field for the reason a slip was not paid: every cause is a correlation.",
            "Contacts cover only the window from the earliest agreement to the last due date.",
            f"The dataset is a snapshot taken as of {dataset.meta.as_of}; buckets change on a "
            "later date.",
        ]
    )
    return doc


def add_arguments(parser: argparse.ArgumentParser) -> None:
    del parser  # no options of its own; reads the dataset


def run(ctx: Context, args: argparse.Namespace) -> int:
    del args
    dataset = ctx.load_dataset()
    df = F.analysis_frame(dataset, ctx.settings)
    size = ctx.dataset_path.stat().st_size if ctx.dataset_path.exists() else None
    doc = render(dataset, df, ctx.settings, size)
    checks = quality_checks(df)
    raised = [c for c in checks if c.severity == HIGH and c.rows]
    ctx.say(f"slips={len(df)} checks={len(checks)} high-severity findings={len(raised)}")
    ctx.write_report("eda.md", doc)
    return 0
