"""Stage 10: the payment-term model, on a seeded synthetic cohort with a planted term effect."""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from delinquency_eda import features as F
from delinquency_eda.config import ModelSettings, Settings
from delinquency_eda.errors import ModelError
from delinquency_eda.stages import s10_term_model as M
from delinquency_eda.stages.s08_clean import clean_frame

from clirun import run_cli
from frames import dataset, raw

SETTINGS = Settings(
    system_agents=("sys",),
    model=ModelSettings(folds=4, permutation_repeats=3, boosting_max_iter=60),
)
LABELS = SETTINGS.term_bands().labels
TERMS = [0, 1, 2, 4, 7, 9, 12, 18, 25]


def synthetic_rows(n: int = 360, seed: int = 3, effect: float = 0.7) -> list[dict]:
    """``n`` mature slips. The chance of not paying rises with the term by ``effect``."""
    rng = random.Random(seed)
    rows = []
    for i in range(1, n + 1):
        term = rng.choice(TERMS)
        agreement = date(2026, 1, 1) + timedelta(days=rng.randint(0, 60))
        due = agreement + timedelta(days=term)
        unpaid = rng.random() < 0.15 + effect * min(term, 25) / 25
        row = raw(
            i,
            person_key=f"pk_{i % 150}",
            agent_id=f"a{i % 5}",
            agent_name=f"Agent {i % 5}",
            agent_login=f"agent_{i % 5}",
            portfolio_label=("Portfolio A-01", "Portfolio B-01", "Portfolio C-01")[i % 3],
            agreement_date=agreement.isoformat(),
            due_date=due.isoformat(),
            first_installment_value=50.0 + (i % 17) * 20,
            total_value=100.0 + (i % 17) * 40,
            original_debt=None if i % 23 == 0 else 400.0 + (i % 11) * 90,
        )
        if not unpaid:
            row |= {"paid_amount": row["first_installment_value"], "paid_date": due.isoformat()}
        rows.append(row)
    return rows


@pytest.fixture(scope="module")
def full():
    return clean_frame(dataset(synthetic_rows()), SETTINGS)


@pytest.fixture(scope="module")
def result(full):
    return M.fit_all(full, SETTINGS)


# ---- cohort and features -------------------------------------------------------------------


def test_cohort():
    rows = synthetic_rows(60)
    rows.append(raw(900, agent_login="sys", agent_name="System", due_date="2026-02-01"))
    rows.append(raw(901, due_date="2026-03-30"))  # still inside its grace window
    cohort = M.model_cohort(clean_frame(dataset(rows), SETTINGS))
    assert len(cohort) == 60
    assert cohort["is_mature"].all() and not cohort["is_system"].any()


def test_design_matrix(full):
    X = M.design_matrix(M.model_cohort(full))
    assert list(X.columns) == list(M.FEATURES)
    assert set(X.columns).isdisjoint(F.LEAKAGE_COLUMNS)
    assert set(X.columns).isdisjoint(F.EXPLANATORY_ONLY_COLUMNS)


def test_missing_debt_flag():
    rows = [raw(1, original_debt=None), raw(2, original_debt=0.0), raw(3, original_debt=250.0)]
    X = M.design_matrix(M.model_cohort(clean_frame(dataset(rows), SETTINGS)))
    assert X["no_debt_link"].tolist() == [1, 0, 0]
    assert X["zero_balance"].tolist() == [0, 1, 0]
    assert X["log_original_debt"].iloc[1] == 0.0  # log1p(0)
    assert math.isnan(X["log_original_debt"].iloc[0])  # imputed inside the pipeline, per fold


def test_negative_balance():
    X = M.design_matrix(
        M.model_cohort(clean_frame(dataset([raw(1, original_debt=-50.0)]), SETTINGS))
    )
    assert X["log_original_debt"].iloc[0] == 0.0


def test_check_cohort():
    one_class = clean_frame(dataset([raw(i) for i in range(1, 11)]), SETTINGS)
    cohort = M.model_cohort(one_class)
    with pytest.raises(ModelError, match="single outcome class"):
        M.check_cohort(cohort, M.target(cohort), 4)
    few_people = clean_frame(dataset([*synthetic_rows(40)[:12], raw(500)]), SETTINGS)
    cohort = M.model_cohort(few_people).assign(group_key="same_person")
    with pytest.raises(ModelError, match="distinct people"):
        M.check_cohort(cohort, np.array([0, 1] * (len(cohort) // 2) + [0]), 4)


# ---- validation scheme ---------------------------------------------------------------------


class Recorder:
    """A stand-in estimator that records which rows it was trained on."""

    seen: list[tuple[set, set]] = []

    def fit(self, X, y):
        self.train = set(X.index)
        return self

    def predict_proba(self, X):
        Recorder.seen.append((self.train, set(X.index)))
        return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])


def test_grouped_folds():
    people = np.array([f"p{i // 3}" for i in range(60)])  # 20 people, 3 rows each
    X = pd.DataFrame({"x": range(60)})
    y = np.array([i % 2 for i in range(60)])
    Recorder.seen = []
    scores = M.grouped_auc(Recorder, X, y, people, 5)
    assert len(scores) == 5 and len(Recorder.seen) == 5
    for train, test in Recorder.seen:
        assert not ({people[i] for i in train} & {people[i] for i in test})
        assert train.isdisjoint(test)
    assert all(s == 0.5 for s in scores)  # a constant score is no better than chance


def test_single_class_fold():
    people = np.array([f"p{i // 2}" for i in range(20)])
    X = pd.DataFrame({"x": range(20)})
    y = np.array([1, 1] + [0] * 18)  # the only positives belong to one person
    Recorder.seen = []
    scores = M.grouped_auc(Recorder, X, y, people, 2)
    # the fold that holds that person out trains on one class; the other tests on one class
    assert len(scores) == 2 and all(math.isnan(s) for s in scores)
    assert len(Recorder.seen) == 1  # only the fold with two training classes was fitted


def test_auc_summary():
    assert math.isnan(M.safe_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))
    assert M.summarize([float("nan"), float("nan")])[2] == 0
    assert M.summarize([0.6, float("nan"), 0.8]) == (pytest.approx(0.7), pytest.approx(0.1), 2)


def test_signal_share():
    assert M.signal_share(0.6, 0.7) == pytest.approx(0.5)
    assert M.signal_share(0.72, 0.7) > 1
    for term, best in ((float("nan"), 0.7), (0.6, float("nan")), (0.6, 0.5), (0.6, 0.4)):
        assert math.isnan(M.signal_share(term, best))


# ---- the effect ----------------------------------------------------------------------------


def test_planted_effect(result):
    logistic, _, valid = M.summarize(result.cv["Logistic (L2)"])
    assert valid == SETTINGS.model.folds
    assert logistic > 0.65
    assert M.summarize(result.term_only)[0] > 0.65


def test_odds_ratio_reference(result):
    assert result.reference == LABELS[0]
    odds = result.term_odds.set_index("band")["odds_ratio"]
    assert LABELS[0] not in odds.index
    assert odds[LABELS[-1]] > 2.0  # the longest term is far likelier to go unpaid
    assert odds[LABELS[-1]] > odds[LABELS[1]]
    assert (odds > 0).all()


def test_empty_reference_band():
    rows = [r for r in synthetic_rows(360) if r["due_date"] != r["agreement_date"]]
    cohort = M.model_cohort(clean_frame(dataset(rows), SETTINGS))
    X = M.design_matrix(cohort)
    pipe = M.make_logistic(SETTINGS, LABELS).fit(X, M.target(cohort))
    reference, odds = M.term_odds(pipe, X, LABELS)
    assert reference == LABELS[1]
    assert LABELS[0] not in odds["band"].tolist()


def test_line_control():
    rows = synthetic_rows(360)
    for i, r in enumerate(rows):
        r["product_line"] = "consumer" if i % 2 else "auto"
    ds = dataset(rows, lines=("auto", "consumer"))
    full = clean_frame(ds, SETTINGS)
    cohort = M.model_cohort(full)
    X = M.design_matrix(cohort)
    pipe = M.make_logistic(SETTINGS, LABELS).fit(X, M.target(cohort))
    controls = M.control_odds(pipe, X)["control"].tolist()
    assert "product line consumer (vs auto)" in controls
    assert "repeat debtor (yes vs no)" in controls


def test_counterfactual(result):
    table = result.counterfactual.set_index("band")
    assert table["rows"].sum() == result.rows
    assert table.loc[LABELS[-1], "predicted"] > table.loc[LABELS[0], "predicted"]
    assert table.loc[LABELS[0], "delta_pp"] == pytest.approx(0.0)


def test_permutation_importance(result):
    importance = result.holdout.importance
    assert importance.iloc[0]["feature"] == "term_band"
    assert set(importance["feature"]) == set(M.FEATURES)
    assert result.holdout.train_rows + result.holdout.test_rows == result.rows


def test_fixed_seed(full, result):
    again = M.fit_all(full, SETTINGS)
    assert again.cv == result.cv
    assert again.term_only == result.term_only


def test_skipped_holdout():
    rows = synthetic_rows(120)
    for r in rows:
        r["agreement_date"] = "2026-01-05"
        r["due_date"] = "2026-01-10"
    outcome = M.fit_all(clean_frame(dataset(rows), SETTINGS), SETTINGS)
    assert outcome.holdout is None and outcome.counterfactual is None
    assert "time split" in outcome.holdout_note
    assert "Not available" in M.render(outcome, SETTINGS).render()


def test_no_effect_cohort():
    outcome = M.fit_all(
        clean_frame(dataset(synthetic_rows(360, seed=11, effect=0.0)), SETTINGS), SETTINGS
    )
    assert M.summarize(outcome.cv["Logistic (L2)"])[0] < 0.62


# ---- the report ----------------------------------------------------------------------------


def test_report_wording(result):
    text = M.render(result, SETTINGS).render()
    for column in F.LEAKAGE_COLUMNS:
        assert f"`{column}`" in text
    assert "associations" in text
    assert "randomized" in text
    assert "grouped by person" in text
    assert f"**{result.reference}** as the reference band" in text


def test_report_sections(result):
    text = M.render(result, SETTINGS).render()
    for heading in ("## 1.", "## 2.", "## 3.", "## 4."):
        assert heading in text


def test_cli_runs_from_a_dataset_file(tmp_path):
    path = dataset(synthetic_rows(240)).save(tmp_path / "ds.json")
    config = tmp_path / "s.json"
    config.write_text(
        '{"model": {"folds": 3, "permutation_repeats": 2, "boosting_max_iter": 40}}',
        encoding="utf-8",
    )
    code, out = run_cli(
        "model", "--dataset", str(path), "--out-dir", str(tmp_path / "r"), "--config", str(config)
    )
    assert code == 0
    assert "Logistic (L2): AUC" in out and "term only: AUC" in out
    assert (tmp_path / "r" / "model.md").exists()


def test_cli_unmodelable_cohort(tmp_path):
    path = dataset([raw(i) for i in range(1, 9)]).save(tmp_path / "ds.json")
    code, _ = run_cli("model", "--dataset", str(path), "--out-dir", str(tmp_path / "r"))
    assert code == 2
