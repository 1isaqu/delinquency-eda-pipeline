"""Stage 10 - payment-term model: how much of the term effect survives controlling for the mix?

The analysis stage usually finds one factor that dominates: the number of days the agent grants
between the agreement and the first due date. The obvious objection is confounding. Agents,
portfolios, product lines and ticket sizes differ, and the longer terms may simply sit with the
riskier mix. This stage asks the narrow question: *given the same mix, is a longer term still
associated with more non-payment?*

Constraints the earlier stages impose, all respected here:

* **labels need maturity**: only mature slips (final outcome) are in the cohort,
* **no leakage**: only information known when the term is agreed is a feature; payment amount,
  payment date, days overdue and status are out (see ``features.LEAKAGE_COLUMNS``),
* **no reverse causality**: follow-up counts are out too (``EXPLANATORY_ONLY_COLUMNS``),
* **people are not independent**: cross-validation is grouped by person, never by row,
* **time**: a temporal hold-out (train on early agreements, test on late ones) is the second check,
* **system rows** are out; a zero balance stays a category, never an imputed number.

Outputs: grouped cross-validated AUC for a logistic model and a boosted-tree model, against a
term-only baseline; odds ratios of the term bands against the shortest observed band (differences of
coefficients, so every band is compared with the same reference); permutation importance on the
hold-out; and a counterfactual that rewrites only the term and keeps everything else.

Read the result as an association measured under control, not as proof that changing the term
changes behaviour: people who ask for a long term may already be the people who cannot pay. Only a
randomised term would settle that. A modest AUC is expected and fine; the model is here to
measure an effect, not to predict who pays.
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from delinquency_eda import features as F
from delinquency_eda.config import Settings
from delinquency_eda.context import Context
from delinquency_eda.derive import OVERDUE_UNPAID
from delinquency_eda.errors import ModelError
from delinquency_eda.report import Doc, fmt_int, fmt_rate
from delinquency_eda.stages.s08_clean import analytic_view, clean_frame

if TYPE_CHECKING:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

NUMERIC = (
    "log_first_installment",
    "log_total_value",
    "log_original_debt",
    "total_installments",
    "due_weekday",
    "prior_breaks",
)
BINARY = ("zero_balance", "no_debt_link", "repeat_debtor")
TERM = ("term_band",)
CATEGORICAL = ("product_line", "portfolio_group", "agent_id")
FEATURES = (*NUMERIC, *BINARY, *TERM, *CATEGORICAL)
CONTROL_LABELS = {
    "log_first_installment": "log first installment (per +1 sd)",
    "log_total_value": "log whole-agreement value (per +1 sd)",
    "log_original_debt": "log original debt (per +1 sd)",
    "total_installments": "installment count (per +1 sd)",
    "due_weekday": "weekday of the due date (per +1 sd)",
    "prior_breaks": "earlier broken agreements (per +1 sd)",
    "zero_balance": "zero original debt (yes vs no)",
    "no_debt_link": "no linked debt (yes vs no)",
    "repeat_debtor": "repeat debtor (yes vs no)",
}


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------


def model_cohort(full: pd.DataFrame) -> pd.DataFrame:
    """Rows the model may use: not system-owned, and mature."""
    view = analytic_view(full)
    return view[view["is_mature"]].reset_index(drop=True)


def _text(series: pd.Series) -> pd.Series:
    return series.astype(object).where(series.notna(), F.UNKNOWN).astype(str)


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """The feature table. Every column is knowable when the term is agreed."""
    debt = df["original_debt"]
    return pd.DataFrame(
        {
            "log_first_installment": np.log1p(df["first_installment_value"].clip(lower=0)),
            "log_total_value": np.log1p(df["total_value"].clip(lower=0)),
            "log_original_debt": np.log1p(debt.clip(lower=0)),
            "total_installments": df["total_installments"],
            "due_weekday": pd.to_datetime(df["due_date"]).dt.dayofweek,
            "prior_breaks": df["prior_breaks"],
            "zero_balance": df["zero_balance"].astype(int),
            "no_debt_link": debt.isna().astype(int),
            "repeat_debtor": (df["prior_breaks"] > 0).astype(int),
            "term_band": _text(df["term_band"]),
            "product_line": _text(df["product_line"]),
            "portfolio_group": _text(df["portfolio_group"]),
            "agent_id": _text(df["agent_id"]),
        },
        index=df.index,
    )[list(FEATURES)]


def target(df: pd.DataFrame) -> np.ndarray:
    return (df["bucket"] == OVERDUE_UNPAID).astype(int).to_numpy()


def check_cohort(df: pd.DataFrame, y: np.ndarray, folds: int) -> None:
    """Refuse to fit when the data cannot support the validation scheme."""
    if len(np.unique(y)) < 2:
        raise ModelError("the mature cohort has a single outcome class; there is nothing to model")
    groups = df["group_key"].nunique()
    if groups < folds:
        raise ModelError(
            f"only {groups} distinct people in the cohort, fewer than the {folds} folds; "
            "widen the period or lower model.folds"
        )


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------


def _preprocessor(settings: Settings, term_labels: list[str]) -> ColumnTransformer:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    terms = OneHotEncoder(
        categories=[[*term_labels, F.UNKNOWN]], handle_unknown="ignore", sparse_output=False
    )
    # Dense output: the boosted-tree estimator does not accept a sparse matrix.
    rest = OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=settings.model.min_category_count,
        sparse_output=False,
    )
    return ColumnTransformer(
        [
            ("num", numeric, list(NUMERIC)),
            ("bin", "passthrough", list(BINARY)),
            ("term", terms, list(TERM)),
            ("cat", rest, list(CATEGORICAL)),
        ]
    )


def make_logistic(settings: Settings, term_labels: list[str]) -> Pipeline:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    m = settings.model
    return Pipeline(
        [
            ("pre", _preprocessor(settings, term_labels)),
            ("m", LogisticRegression(C=m.logistic_c, max_iter=m.logistic_max_iter)),
        ]
    )


def make_boosting(settings: Settings, term_labels: list[str]) -> Pipeline:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline

    m = settings.model
    return Pipeline(
        [
            ("pre", _preprocessor(settings, term_labels)),
            (
                "m",
                HistGradientBoostingClassifier(
                    random_state=m.random_state,
                    max_iter=m.boosting_max_iter,
                    learning_rate=m.boosting_learning_rate,
                ),
            ),
        ]
    )


def make_term_only(settings: Settings, term_labels: list[str]) -> Pipeline:
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(
        categories=[[*term_labels, F.UNKNOWN]], handle_unknown="ignore", sparse_output=False
    )
    return Pipeline(
        [
            ("pre", ColumnTransformer([("term", encoder, list(TERM))])),
            ("m", LogisticRegression(max_iter=settings.model.logistic_max_iter)),
        ]
    )


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC, or ``nan`` when the fold has a single class (the metric is undefined there)."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def grouped_auc(
    make: Callable[[], Pipeline], X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, folds: int
) -> list[float]:
    """Cross-validated AUC with whole people kept on one side of every split."""
    from sklearn.model_selection import GroupKFold

    scores = []
    for train, test in GroupKFold(n_splits=folds).split(X, y, groups):
        if len(np.unique(y[train])) < 2:
            scores.append(float("nan"))
            continue
        pipe = make()
        pipe.fit(X.iloc[train], y[train])
        scores.append(safe_auc(y[test], pipe.predict_proba(X.iloc[test])[:, 1]))
    return scores


def summarise(scores: list[float]) -> tuple[float, float, int]:
    """``(mean, standard deviation, folds that had a defined AUC)``."""
    valid = [s for s in scores if not np.isnan(s)]
    if not valid:
        return float("nan"), float("nan"), 0
    return float(np.mean(valid)), float(np.std(valid)), len(valid)


# --------------------------------------------------------------------------------------------
# Effects
# --------------------------------------------------------------------------------------------


def _feature_name(prefix: str, column: str, category: str) -> str:
    return f"{prefix}__{column}_{category}"


def reference_band(X: pd.DataFrame, labels: list[str]) -> str | None:
    """The shortest term band that has any rows: the baseline every other band is compared with."""
    present = set(X["term_band"])
    return next((label for label in labels if label in present), None)


def term_odds(
    pipe: Pipeline, X: pd.DataFrame, labels: list[str]
) -> tuple[str | None, pd.DataFrame]:
    """Odds ratios of each term band against the reference band, mix held fixed."""
    reference = reference_band(X, labels)
    names = list(pipe.named_steps["pre"].get_feature_names_out())
    coef = dict(zip(names, pipe.named_steps["m"].coef_[0], strict=True))
    counts = X["term_band"].value_counts()
    rows = []
    if reference is not None:
        base = coef[_feature_name("term", "term_band", reference)]
        for label in labels:
            if label == reference:
                continue
            key = _feature_name("term", "term_band", label)
            if key in coef and counts.get(label, 0) > 0:
                rows.append(
                    {
                        "band": label,
                        "rows": int(counts[label]),
                        "odds_ratio": float(np.exp(coef[key] - base)),
                    }
                )
    return reference, pd.DataFrame(rows, columns=["band", "rows", "odds_ratio"])


def control_odds(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Odds ratios of the numeric and yes/no controls, and of product line against the first."""
    names = list(pipe.named_steps["pre"].get_feature_names_out())
    coef = dict(zip(names, pipe.named_steps["m"].coef_[0], strict=True))
    rows = []
    for column in (*NUMERIC, *BINARY):
        prefix = "num" if column in NUMERIC else "bin"
        key = f"{prefix}__{column}"
        if key in coef:
            rows.append({"control": CONTROL_LABELS[column], "odds_ratio": float(np.exp(coef[key]))})
    lines = sorted(X["product_line"].unique())
    if len(lines) > 1:
        base = coef.get(_feature_name("cat", "product_line", lines[0]))
        for line in lines[1:]:
            key = _feature_name("cat", "product_line", line)
            if base is not None and key in coef:
                rows.append(
                    {
                        "control": f"product line {line} (vs {lines[0]})",
                        "odds_ratio": float(np.exp(coef[key] - base)),
                    }
                )
    return pd.DataFrame(rows, columns=["control", "odds_ratio"])


@dataclass
class Holdout:
    cut: date
    train_rows: int
    test_rows: int
    auc: float
    importance: pd.DataFrame
    pipeline: Pipeline


def temporal_split(df: pd.DataFrame, quantile: float) -> tuple[np.ndarray, np.ndarray, date]:
    """Boolean train / test masks split by agreement date at ``quantile``."""
    dates = pd.to_datetime(df["agreement_date"])
    cut = dates.quantile(quantile)
    return (dates <= cut).to_numpy(), (dates > cut).to_numpy(), cut.date()


def run_holdout(
    X: pd.DataFrame, y: np.ndarray, df: pd.DataFrame, settings: Settings, labels: list[str]
) -> tuple[Holdout | None, str]:
    """Fit the boosted model on early agreements, score late ones, and measure importance."""
    from sklearn.inspection import permutation_importance

    train, test, cut = temporal_split(df, settings.model.holdout_quantile)
    if train.sum() == 0 or test.sum() == 0:
        return None, "the time split left one side empty"
    if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
        return None, "one side of the time split has a single outcome class"
    pipe = make_boosting(settings, labels)
    pipe.fit(X[train], y[train])
    auc = safe_auc(y[test], pipe.predict_proba(X[test])[:, 1])
    imp = permutation_importance(
        pipe,
        X[test],
        y[test],
        n_repeats=settings.model.permutation_repeats,
        random_state=settings.model.random_state,
        scoring="roc_auc",
    )
    table = pd.DataFrame(
        {"feature": list(X.columns), "auc_drop": imp.importances_mean, "sd": imp.importances_std}
    ).sort_values("auc_drop", ascending=False)
    return Holdout(cut, int(train.sum()), int(test.sum()), auc, table, pipe), ""


def counterfactual(
    pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, labels: list[str]
) -> pd.DataFrame:
    """Predicted unpaid probability if only the term band were rewritten, everything else fixed."""
    observed = pd.Series(y, index=X.index).groupby(X["term_band"]).agg(["mean", "size"])
    rows = []
    for label in labels:
        shifted = X.assign(term_band=label)
        rows.append(
            {
                "band": label,
                "predicted": float(pipe.predict_proba(shifted)[:, 1].mean()),
                "observed": float(observed["mean"].get(label, np.nan)),
                "rows": int(observed["size"].get(label, 0)),
            }
        )
    out = pd.DataFrame(rows)
    first = out.loc[out["rows"] > 0, "predicted"]
    out["delta_pp"] = 100 * (out["predicted"] - (first.iloc[0] if len(first) else np.nan))
    return out


# --------------------------------------------------------------------------------------------
# Orchestration and report
# --------------------------------------------------------------------------------------------


@dataclass
class ModelResult:
    rows: int
    people: int
    unpaid_rate: float
    cv: dict[str, list[float]]
    term_only: list[float]
    reference: str | None
    term_odds: pd.DataFrame
    control_odds: pd.DataFrame
    holdout: Holdout | None
    holdout_note: str
    counterfactual: pd.DataFrame | None
    convergence_warnings: int


def fit_all(full: pd.DataFrame, settings: Settings) -> ModelResult:
    """Everything the report needs, from the cleaned full table."""
    from sklearn.exceptions import ConvergenceWarning

    cohort = model_cohort(full)
    y = target(cohort)
    check_cohort(cohort, y, settings.model.folds)
    X = design_matrix(cohort)
    groups = cohort["group_key"].to_numpy()
    labels = settings.term_bands().labels
    folds = settings.model.folds
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        cv = {
            "Logistic (L2)": grouped_auc(
                lambda: make_logistic(settings, labels), X, y, groups, folds
            ),
            "Gradient boosting": grouped_auc(
                lambda: make_boosting(settings, labels), X, y, groups, folds
            ),
        }
        term_only = grouped_auc(lambda: make_term_only(settings, labels), X, y, groups, folds)
        logistic = make_logistic(settings, labels).fit(X, y)
        reference, odds = term_odds(logistic, X, labels)
        holdout, note = run_holdout(X, y, cohort, settings, labels)
    convergence = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))
    for w in caught:
        if not issubclass(w.category, ConvergenceWarning):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return ModelResult(
        rows=len(cohort),
        people=int(cohort["group_key"].nunique()),
        unpaid_rate=float(y.mean()),
        cv=cv,
        term_only=term_only,
        reference=reference,
        term_odds=odds,
        control_odds=control_odds(logistic, X),
        holdout=holdout,
        holdout_note=note,
        counterfactual=counterfactual(holdout.pipeline, X, y, labels) if holdout else None,
        convergence_warnings=convergence,
    )


def signal_share(term_auc: float, best_auc: float) -> float:
    """Share of the signal above chance (AUC 0.5) that the term alone captures."""
    if np.isnan(term_auc) or np.isnan(best_auc) or best_auc <= 0.5:
        return float("nan")
    return (term_auc - 0.5) / (best_auc - 0.5)


def _auc_cell(scores: list[float]) -> list[str]:
    mean, sd, valid = summarise(scores)
    folds = ", ".join("n/a" if np.isnan(s) else f"{s:.3f}" for s in scores)
    return ["n/a" if valid == 0 else f"**{mean:.3f}** +/- {sd:.3f}", folds]


def render(result: ModelResult, settings: Settings) -> Doc:
    r = result
    doc = Doc().h1("Payment-term model: the term, with the mix controlled")
    doc.p(
        f"Cohort: {fmt_int(r.rows)} mature slips from {fmt_int(r.people)} people, unpaid rate "
        f"{fmt_rate(r.unpaid_rate)}. Features: {', '.join(f'`{f}`' for f in FEATURES)}."
    )
    doc.p(
        "Excluded as leakage: "
        + ", ".join(f"`{c}`" for c in F.LEAKAGE_COLUMNS)
        + ". Excluded as reverse-causal: "
        + ", ".join(f"`{c}`" for c in F.EXPLANATORY_ONLY_COLUMNS)
        + "."
    )
    _predictive(doc, r, settings)
    _odds(doc, r)
    _holdout(doc, r)
    doc.h2("How to read this")
    doc.bullets(
        [
            "Odds ratios above 1 mean more non-payment than the reference band, with the other "
            "features held fixed. They are associations: people who ask for a long term may "
            "already be the people who cannot pay.",
            "A modest AUC is normal for collections data. The model measures an effect; it is "
            "not a scoring tool.",
            "A randomised term (draw the term offered among eligible agreements, compare payment "
            "at the end of the grace period) is the only design that would show cause.",
        ]
    )
    if r.convergence_warnings:
        doc.p(
            f"Note: the optimiser raised {r.convergence_warnings} convergence warning(s); "
            "raise `model.logistic_max_iter`."
        )
    return doc


def _predictive(doc: Doc, r: ModelResult, settings: Settings) -> None:
    doc.h2(f"1. Predictive power (grouped by person, {settings.model.folds} folds)")
    rows = [[name, *_auc_cell(scores)] for name, scores in r.cv.items()]
    rows.append(["Term band only (baseline)", *_auc_cell(r.term_only)])
    doc.table(rows, ["model", "AUC (mean +/- sd)", "AUC per fold"])
    best = max((summarise(s)[0] for s in r.cv.values()), default=float("nan"))
    term = summarise(r.term_only)[0]
    share = signal_share(term, best)
    if np.isnan(share):
        doc.p("The term-only share of the signal is not defined (no model beat chance).")
    elif share >= 1:
        doc.p(
            f"The term alone reaches AUC {term:.3f}; the best full model {best:.3f}. The extra "
            "features add nothing out of sample: the term carries all the usable signal."
        )
    else:
        doc.p(
            f"The term alone reaches AUC {term:.3f}; the best full model {best:.3f}. The term "
            f"captures {100 * share:.0f}% of the signal above chance."
        )


def _odds(doc: Doc, r: ModelResult) -> None:
    doc.h2("2. Does the term survive the control for mix?")
    if r.reference is None:
        doc.p("No term band has rows, so no odds ratio is defined.")
        return
    doc.p(
        f"Logistic model with **{r.reference}** as the reference band (the shortest band with "
        "rows). Ridge-shrunk coefficients, so ratios are conservative."
    )
    doc.table(
        [[b.band, fmt_int(b.rows), f"{b.odds_ratio:.2f}x"] for b in r.term_odds.itertuples()],
        ["term band (days)", "rows", "odds ratio vs reference"],
    )
    doc.p("Other controls:")
    doc.table(
        [[c.control, f"{c.odds_ratio:.2f}x"] for c in r.control_odds.itertuples()],
        ["control", "odds ratio"],
    )


def _holdout(doc: Doc, r: ModelResult) -> None:
    doc.h2("3. Out-of-time check and permutation importance")
    if r.holdout is None:
        doc.p(f"Not available: {r.holdout_note}.")
        return
    h = r.holdout
    cv_boost = summarise(r.cv["Gradient boosting"])[0]
    gap = abs(h.auc - cv_boost)
    verdict = (
        "within 0.05 of each other: the relation looks stable over time"
        if gap < 0.05
        else "more than 0.05 apart: the relation moves over time"
    )
    doc.p(
        f"Train on agreements up to {h.cut} ({fmt_int(h.train_rows)} rows), test on later ones "
        f"({fmt_int(h.test_rows)} rows). Out-of-time AUC {h.auc:.3f}, against {cv_boost:.3f} in "
        f"grouped cross-validation: {verdict}."
    )
    doc.table(
        [
            [f"`{x.feature}`", f"{x.auc_drop:+.4f}", f"{x.sd:.4f}"]
            for x in h.importance.itertuples()
        ],
        ["feature", "AUC drop when shuffled", "sd"],
    )
    if r.counterfactual is not None:
        _counterfactual(doc, r.counterfactual)


def _counterfactual(doc: Doc, table: pd.DataFrame) -> None:
    doc.h2("4. Predicted unpaid probability by term, mix held constant")
    doc.p(
        "Every agreement is rewritten to each term band in turn, keeping portfolio, agent, "
        "value and debtor fixed. A band with no observed rows is an extrapolation."
    )
    doc.table(
        [
            [
                c.band,
                fmt_int(c.rows),
                fmt_rate(c.predicted),
                "-" if np.isnan(c.delta_pp) else f"{c.delta_pp:+.1f} pp",
                fmt_rate(c.observed),
            ]
            for c in table.itertuples()
        ],
        [
            "term band (days)",
            "observed rows",
            "predicted unpaid",
            "change vs reference",
            "raw rate",
        ],
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    del parser  # model settings live in the settings file under "model"


def run(ctx: Context, args: argparse.Namespace) -> int:
    del args
    dataset = ctx.load_dataset()
    full = clean_frame(dataset, ctx.settings)
    result = fit_all(full, ctx.settings)
    for name, scores in result.cv.items():
        mean, sd, valid = summarise(scores)
        ctx.say(f"{name}: AUC {mean:.3f} +/- {sd:.3f} over {valid} fold(s)")
    ctx.say(f"term only: AUC {summarise(result.term_only)[0]:.3f}")
    ctx.write_report("model.md", render(result, ctx.settings))
    return 0
