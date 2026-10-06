"""Stage 03 - factor analysis over the overdue-unpaid bucket.

For each candidate factor (agent, portfolio, product line, follow-up, repeat breaks, instalment
ratio, value, term, instalment count) two tables are produced:

* **concentration** over every overdue-unpaid slip: who holds the unpaid value?
* **discrimination** over the *mature* cohort only: does the group fail more or less often than
  the baseline, and is the group big enough to say so?

Concentration says where the money is; only discrimination says whether a factor separates payers
from non-payers. A factor can top the concentration table simply because it is big. The mature
cohort is used for rates because a slip still inside its grace window is censored: it can still be
paid on time, so counting it as a failure measures recency, not behaviour.

The follow-up factor is descriptive only. Outreach is aimed at people who are already slipping, so
follow-up and non-payment are entangled by reverse causality, and a higher unpaid rate among
followed-up slips does not mean outreach hurts.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from delinquency_eda import features as F
from delinquency_eda.config import Settings
from delinquency_eda.context import Context
from delinquency_eda.dataset import Dataset
from delinquency_eda.derive import OVERDUE_UNPAID, PAID_ON_TIME, TO_MATURE
from delinquency_eda.report import Doc, fmt_int, fmt_money, fmt_pct, fmt_rate
from delinquency_eda.tables import concentration, discrimination, unpaid_rate

LabelFn = Callable[[pd.DataFrame, Settings], pd.Series]
OrderFn = Callable[[Settings], list[str]]


@dataclass(frozen=True)
class Factor:
    key: str
    title: str
    labels: LabelFn
    order: OrderFn | None = None


def _text(column: str, missing: str = F.UNKNOWN) -> LabelFn:
    return lambda df, s: df[column].astype(object).where(df[column].notna(), missing)


def _band(column: str, bands: str) -> tuple[LabelFn, OrderFn]:
    return (
        lambda df, s: df[column].astype(object),
        lambda s: getattr(s, bands)().labels,
    )


def _repeat_labels(df: pd.DataFrame, s: Settings) -> pd.Series:
    return pd.Series(
        [F.repeat_level(int(n), s.chronic_break_threshold) for n in df["prior_breaks"]],
        index=df.index,
        dtype=object,
    )


def _repeat_order(s: Settings) -> list[str]:
    t = s.chronic_break_threshold
    return [F.REPEAT_NONE, *[F.repeat_level(n, t) for n in range(1, t + 1)]]


def _repeat_flag(df: pd.DataFrame, s: Settings) -> pd.Series:
    return pd.Series(np.where(df["prior_breaks"] > 0, "repeat", "first time"), index=df.index)


def _cross(df: pd.DataFrame, s: Settings) -> pd.Series:
    follow = np.where(df["contacts_after"] == 0, "no follow-up", "with follow-up")
    repeat = np.where(df["prior_breaks"] > 0, "repeat", "first time")
    return pd.Series([f"{a} / {b}" for a, b in zip(follow, repeat, strict=True)], index=df.index)


_CROSS_ORDER = [
    f"{a} / {b}" for a in ("no follow-up", "with follow-up") for b in ("first time", "repeat")
]
_TERM_LABELS, _TERM_ORDER = _band("term_band", "term_bands")
_VALUE_LABELS, _VALUE_ORDER = _band("value_band", "value_bands")
_INST_LABELS, _INST_ORDER = _band("installment_band", "installment_bands")

FACTORS: dict[str, Factor] = {
    f.key: f
    for f in (
        Factor("agent", "Agent", _text("agent_id")),
        Factor("portfolio", "Portfolio label", _text("portfolio_label", F.NO_PORTFOLIO)),
        Factor("portfolio_group", "Portfolio group", _text("portfolio_group")),
        Factor("product_line", "Product line", _text("product_line")),
        Factor(
            "followup",
            "Follow-up intensity",
            _text("followup_level"),
            lambda s: list(F.FOLLOWUP_LEVELS),
        ),
        Factor("repeat", "Earlier broken agreements", _repeat_labels, _repeat_order),
        Factor(
            "repeat_flag",
            "Repeat debtor, yes or no",
            _repeat_flag,
            lambda s: ["first time", "repeat"],
        ),
        Factor(
            "ratio",
            "First installment as a share of the original debt",
            lambda df, s: F.ratio_group(df),
            lambda s: [*s.ratio_bands().labels, *F.RATIO_STATE_LABELS.values()],
        ),
        Factor("value", "First-installment value", _VALUE_LABELS, _VALUE_ORDER),
        Factor("term", "Days from agreement to first due date", _TERM_LABELS, _TERM_ORDER),
        Factor("installments", "Installments in the agreement", _INST_LABELS, _INST_ORDER),
        Factor("followup_x_repeat", "Follow-up by repeat debtor", _cross, lambda s: _CROSS_ORDER),
    )
}

SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Agent", ("agent",)),
    ("Portfolio and product line", ("portfolio", "portfolio_group", "product_line")),
    ("Follow-up between agreement and due date", ("followup",)),
    ("Repeat breaks", ("repeat", "repeat_flag")),
    ("First installment against the original debt", ("ratio",)),
    ("Value, term and installment count", ("value", "term", "installments")),
    ("Follow-up by repeat debtor", ("followup_x_repeat",)),
)


@dataclass
class FactorTables:
    factor: Factor
    concentration: pd.DataFrame
    discrimination: pd.DataFrame


@dataclass
class AnalysisResult:
    baseline: float
    mature_slips: int
    unpaid_slips: int
    definitive_unpaid: int
    tables: dict[str, FactorTables]


def analyse(df: pd.DataFrame, settings: Settings) -> AnalysisResult:
    """Both tables for every factor. ``df`` comes from ``features.analysis_frame``."""
    unpaid = df[df["bucket"] == OVERDUE_UNPAID]
    mature = df[df["is_mature"]]
    baseline = unpaid_rate(mature)
    tables: dict[str, FactorTables] = {}
    for key, factor in FACTORS.items():
        labels = factor.labels(df, settings)
        order = factor.order(settings) if factor.order else None
        tables[key] = FactorTables(
            factor,
            concentration(unpaid, labels.loc[unpaid.index], order=order),
            discrimination(mature, labels.loc[mature.index], baseline, settings, order=order),
        )
    definitive = int((mature["bucket"] == OVERDUE_UNPAID).sum())
    return AnalysisResult(baseline, len(mature), len(unpaid), definitive, tables)


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def _concentration_rows(table: pd.DataFrame) -> list[list[str]]:
    rows = [
        [
            r.group,
            fmt_int(r.slips),
            fmt_rate(r.share_slips),
            fmt_money(r.value),
            fmt_rate(r.share_value),
            fmt_money(r.avg_ticket),
        ]
        for r in table.itertuples(index=False)
    ]
    slips, value = int(table["slips"].sum()), float(table["value"].sum())
    rows.append(
        [
            "**total**",
            fmt_int(slips),
            "100.0%" if slips else "-",
            fmt_money(value),
            "100.0%" if slips else "-",
            fmt_money(value / slips if slips else None),
        ]
    )
    return rows


def _discrimination_rows(table: pd.DataFrame) -> list[list[str]]:
    return [
        [
            r.group,
            fmt_int(r.mature),
            fmt_int(r.unpaid),
            fmt_rate(r.rate),
            "-" if pd.isna(r.lift) else f"{r.lift:.2f}x",
            r.reliability,
        ]
        for r in table.itertuples(index=False)
    ]


def _factor_block(
    doc: Doc, tables: FactorTables, result: AnalysisResult, *, heading: bool = True
) -> None:
    title = tables.factor.title
    if heading:
        doc.h3(title)
    doc.p(f"Concentration of the overdue-unpaid bucket (factor: {title}).")
    doc.table(
        _concentration_rows(tables.concentration),
        ["group", "slips", "% of slips", "first-installment value", "% of value", "avg ticket"],
    )
    doc.p(
        f"Unpaid rate on the mature cohort (factor: {title}; n = {fmt_int(result.mature_slips)}, "
        f"baseline {fmt_rate(result.baseline)})."
    )
    doc.table(
        _discrimination_rows(tables.discrimination),
        ["group", "mature", "unpaid", "rate", "lift", "sample size"],
    )


def _scope_section(doc: Doc, df: pd.DataFrame, dataset: Dataset, result: AnalysisResult) -> None:
    meta = dataset.meta
    doc.h2("Scope")
    doc.p(
        f"First-installment slips due {meta.period_from} to {meta.period_to}, outcomes as of "
        f"{meta.as_of}, grace period {meta.grace_days} day(s). Product lines: "
        f"{', '.join(meta.product_lines)}."
    )
    rows = []
    for bucket in (PAID_ON_TIME, OVERDUE_UNPAID, TO_MATURE):
        part = df[df["bucket"] == bucket]
        rows.append(
            [
                bucket,
                fmt_int(len(part)),
                fmt_money(part["first_installment_value"].sum()),
                fmt_money(part["total_value"].sum()),
            ]
        )
    rows.append(
        [
            "**all**",
            fmt_int(len(df)),
            fmt_money(df["first_installment_value"].sum()),
            fmt_money(df["total_value"].sum()),
        ]
    )
    doc.table(rows, ["bucket", "slips", "first-installment value", "whole-agreement value"])
    _measures_section(doc, df)
    doc.p(
        f"Mature cohort: {fmt_int(result.mature_slips)} slips, of which "
        f"{fmt_int(result.definitive_unpaid)} unpaid (baseline {fmt_rate(result.baseline)}). "
        f"{fmt_int(result.unpaid_slips - result.definitive_unpaid)} of the "
        f"{fmt_int(result.unpaid_slips)} overdue-unpaid slips are still inside the grace window "
        "and can still be paid on time; see the censoring stage."
    )


def _measures_section(doc: Doc, df: pd.DataFrame) -> None:
    unpaid = df[df["bucket"] == OVERDUE_UNPAID]
    people = unpaid["person_key"].dropna().nunique()
    doc.p(
        "The same overdue-unpaid bucket under three value measures. A headline figure sums one "
        "of them, and they differ by a large factor, so say which one a number uses."
    )
    doc.table(
        [
            [
                "first installment (cash not received)",
                fmt_money(unpaid["first_installment_value"].sum()),
            ],
            ["whole agreement (all installments)", fmt_money(unpaid["total_value"].sum())],
            ["original debt behind the agreements", fmt_money(unpaid["original_debt"].sum())],
        ],
        ["measure", "sum over overdue-unpaid slips"],
    )
    doc.p(
        f"Distinct people: {fmt_int(people)} over {fmt_int(len(unpaid))} slips "
        "(slips without a person key are not counted as people)."
    )


def _side_notes(doc: Doc, df: pd.DataFrame) -> None:
    unpaid = df[df["bucket"] == OVERDUE_UNPAID]
    n = len(unpaid)
    value = unpaid["first_installment_value"].sum()
    quiet = unpaid[unpaid["contacts_after"] == 0]
    repeat = unpaid[unpaid["prior_breaks"] > 0]
    doc.h2("Side notes")
    doc.bullets(
        [
            f"Overdue-unpaid slips with no follow-up after the agreement day: {len(quiet)} of {n} "
            f"({fmt_pct(len(quiet), n)}), value share "
            f"{fmt_pct(quiet['first_installment_value'].sum(), value)}.",
            "With no contact event at all in the window: "
            f"{int((unpaid['contacts_total'] == 0).sum())}.",
            f"Person key not found in the debtors table (history cannot be checked): "
            f"{int((~unpaid['known_person']).sum())}.",
            f"Repeat debtors in the bucket: {len(repeat)} of {n} ({fmt_pct(len(repeat), n)}).",
            f"Of those, with a manual break among the earlier ones: "
            f"{int((unpaid['prior_manual_breaks'] > 0).sum())}.",
        ]
    )


def render(df: pd.DataFrame, dataset: Dataset, result: AnalysisResult) -> Doc:
    doc = Doc().h1("Factor analysis: overdue-unpaid slips")
    doc.quote(
        "Observational. A factor that separates payers from non-payers is an association; "
        "nothing here says what would happen if the factor were changed."
    )
    _scope_section(doc, df, dataset, result)
    doc.h2("Reading the tables")
    doc.bullets(
        [
            "Concentration: every overdue-unpaid slip, so the totals match the headline bucket.",
            "Discrimination: mature slips only. Lift is the group's unpaid rate over the baseline.",
            "Sample size: reliable, weak or not reliable by the configured minimums; a rate on a "
            "handful of slips is noise, so do not rank on it.",
        ]
    )
    for number, (title, keys) in enumerate(SECTIONS, start=1):
        doc.h2(f"{number}. {title}")
        for key in keys:
            _factor_block(doc, result.tables[key], result, heading=len(keys) > 1)
    _side_notes(doc, df)
    return doc


def add_arguments(parser: argparse.ArgumentParser) -> None:
    del parser  # no options of its own; reads the dataset


def run(ctx: Context, args: argparse.Namespace) -> int:
    del args
    dataset = ctx.load_dataset()
    df = F.analysis_frame(dataset, ctx.settings)
    result = analyse(df, ctx.settings)
    ctx.say(
        f"slips={len(df)} overdue_unpaid={result.unpaid_slips} mature={result.mature_slips} "
        f"baseline={fmt_rate(result.baseline)}"
    )
    ctx.write_report("analysis.md", render(df, dataset, result))
    return 0
