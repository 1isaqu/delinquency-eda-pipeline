"""The whole pipeline, stage by stage, on the seeded demo databases.

The generator plants the situations each stage is built to find. These tests check that the stages
find them, which is also what the README quick start promises.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from delinquency_eda.config import Settings
from delinquency_eda.dataset import Dataset
from delinquency_eda.stages import s03_analysis

from clirun import run_cli, sqlite_args

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
BASE = ["base", "--from", "2025-11-01", "--to", "2026-03-31", "--as-of", "2026-03-31"]


@pytest.fixture(scope="module")
def workspace(demo_dir, tmp_path_factory):
    """Arguments shared by every stage, and a built dataset."""
    root = tmp_path_factory.mktemp("run")
    settings = json.loads((EXAMPLES / "settings.example.json").read_text(encoding="utf-8"))
    settings["model"] |= {"permutation_repeats": 2, "boosting_max_iter": 40}  # keep the suite quick
    (root / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    common = [
        *sqlite_args(("auto", demo_dir / "auto.db"), ("consumer", demo_dir / "consumer.db")),
        "--dataset", str(root / "dataset.json"),
        "--out-dir", str(root / "reports"),
        "--config", str(root / "settings.json"),
    ]  # fmt: skip
    code, _ = run_cli(*common, *BASE)
    assert code == 0
    return {
        "common": common,
        "dataset": root / "dataset.json",
        "reports": root / "reports",
        "settings": root / "settings.json",
    }


def stage(workspace, *args):
    return run_cli(*workspace["common"], *args)


def report(workspace, name):
    return (workspace["reports"] / name).read_text(encoding="utf-8")


def test_the_demo_contract_is_satisfied(workspace):
    code, out = stage(workspace, "schema")
    assert code == 0
    assert "auto: ready for period/card = True, ready for base = True" in out
    assert "consumer: ready" in out


def test_the_base_build_spans_both_product_lines_and_resolves_every_rule(workspace):
    dataset = Dataset.load(workspace["dataset"])
    frame = dataset.base_frame()
    assert {"auto", "consumer"} == set(frame["product_line"])
    assert len(frame) > 1000
    assert set(frame["bucket"]) == {"paid_on_time", "overdue_unpaid", "to_mature"}
    # the generator re-writes some installments and the pipeline keeps one row per agreement
    assert not frame.duplicated(["product_line", "agreement_id", "portfolio_id"]).any()
    # the excluded system agent never reaches the base; the pseudo-agent that passes the filter does
    assert "system_batch" not in set(frame["agent_login"])
    assert "batch_loader" in set(frame["agent_login"])
    assert dataset.debtors and dataset.contacts and dataset.breaks


def test_period_picks_the_window_the_reference_counts_came_from(workspace):
    ds = Dataset.load(workspace["dataset"])
    frame = ds.base_frame()
    march = frame[(frame["due_date"] >= "2026-03-01") & (frame["due_date"] <= "2026-03-31")]
    code, out = stage(
        workspace, "period", "--as-of", "2026-03-31",
        "--reference", f"generated={len(march)}",
        "--reference", f"overdue_unpaid={int((march['bucket'] == 'overdue_unpaid').sum())}",
    )  # fmt: skip
    assert code == 0
    assert "closest window: month to date (distance 0)" in out


def test_censoring_finds_slips_that_are_not_lost_yet(workspace):
    code, out = stage(workspace, "censoring")
    assert code == 0
    still_open = int(out.split("still_open=")[1].split()[0])
    assert still_open > 0
    assert "definitive (grace period over)" in report(workspace, "censoring.md")


def test_the_card_inflates_only_the_open_month(workspace):
    code, out = stage(workspace, "card", "--as-of", "2026-03-31")
    assert code == 0
    lines = {line.split("  ")[0].strip(): line for line in out.splitlines() if "generated=" in line}
    mtd = lines["month to date"]
    as_counted = int(mtd.split("as_counted=")[1].split()[0])
    lost = int(mtd.split("lost=")[1].split()[0])
    assert as_counted > lost  # the open month counts slips that are not lost yet
    closed = lines["previous month"]
    assert closed.split("as_counted=")[1].split()[0] == closed.split("lost=")[1].split()[0]


def test_drift_finds_slips_that_changed_bucket_since_a_few_days_ago(workspace):
    code, out = stage(workspace, "drift", "--reference-date", "2026-03-27")
    assert code == 0 and "entered:" in out
    assert "Transitions" in report(workspace, "drift.md")


def test_analysis_puts_the_pseudo_agent_at_the_top_of_the_unpaid_rate(workspace):
    ds = Dataset.load(workspace["dataset"])
    from delinquency_eda import features

    frame = features.analysis_frame(ds, Settings())
    result = s03_analysis.analyze(frame, Settings())
    agents = result.tables["agent"].discrimination
    top = agents.iloc[0]
    assert top["group"] == "ldr" and top["rate"] == 1.0
    terms = result.tables["term"].discrimination.set_index("group")["rate"]
    assert terms.iloc[-1] > terms.iloc[0]  # long terms fail more often than short ones
    code, _ = stage(workspace, "analysis")
    assert code == 0


def test_eda_and_diagnostics_report_what_the_generator_planted(workspace):
    for name in ("eda", "diagnose"):
        code, _ = stage(workspace, name)
        assert code == 0
    assert "6.2 Target leakage" in report(workspace, "eda.md")
    diagnose = report(workspace, "diagnose.md")
    assert "Batch Loader 01" in diagnose  # the pseudo-agent that passed the exclusion filter
    assert "Nothing was changed" in diagnose


def test_cleaning_flags_the_pseudo_agent_and_removes_nothing(workspace):
    code, out = stage(workspace, "clean")
    assert code == 0
    full = pd.read_csv(workspace["reports"] / "clean_full.csv", dtype={"person_key": str})
    analytic = pd.read_csv(workspace["reports"] / "clean_analytic.csv", dtype={"person_key": str})
    flagged = full[full["is_system"]]
    assert set(flagged["agent_login"]) == {"batch_loader"}
    assert len(analytic) == len(full) - len(flagged)
    assert full["agreement_key"].is_unique
    meta = json.loads((workspace["reports"] / "clean_meta.json").read_text(encoding="utf-8"))
    assert meta["system_agents"] == ["batch_loader"]
    assert "| rows removed | 0 |" in report(workspace, "clean.md")
    assert "analytic=" in out


def test_the_term_model_recovers_the_planted_effect(workspace):
    code, out = stage(workspace, "model")
    assert code == 0 and "Logistic (L2): AUC" in out
    text = report(workspace, "model.md")
    assert "## 4. Predicted unpaid probability" in text
    auc = float(out.split("Logistic (L2): AUC ")[1].split()[0])
    assert auc > 0.6
    section = text.split("## 2.")[1].split("Other controls")[0]
    ratios = [float(x) for x in re.findall(r"\| ([\d.]+)x \|", section)]
    assert len(ratios) >= 3
    assert ratios[-1] > 2.0  # the longest term is far likelier to go unpaid than the shortest


def test_the_whole_pipeline_is_reproducible(workspace, tmp_path):
    again = [*workspace["common"]]
    again[again.index("--dataset") + 1] = str(tmp_path / "again.json")
    code, _ = run_cli(*again, *BASE)
    assert code == 0
    assert workspace["dataset"].read_bytes() == (tmp_path / "again.json").read_bytes()
