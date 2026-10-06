"""Stage 03: factor analysis on a small invented dataset."""

from __future__ import annotations

import pytest

from delinquency_eda import features as F
from delinquency_eda.config import Settings
from delinquency_eda.stages.s03_analysis import FACTORS, SECTIONS, analyse, render

from clirun import run_cli
from frames import dataset, paid_on_time, raw

SETTINGS = Settings(min_n_reliable=2, min_n_weak=1)
OTHER_AGENT = {"agent_id": "a2", "agent_name": "Agent 02", "agent_login": "agent_02"}


def _rows():
    return [
        paid_on_time(1),  # mature, paid
        raw(2, **OTHER_AGENT),  # mature, unpaid
        raw(3, due_date="2026-03-12", **OTHER_AGENT),  # mature, unpaid
        raw(4, due_date="2026-03-29"),  # unpaid but still inside the grace window
        raw(5, due_date="2026-04-05"),  # not due yet
    ]


@pytest.fixture
def frame():
    return F.analysis_frame(dataset(_rows()), SETTINGS)


def test_baseline_uses_only_the_mature_cohort(frame):
    result = analyse(frame, SETTINGS)
    # mature rows: 1 (paid), 2 and 3 (unpaid). Row 4 is censored, row 5 is not due.
    assert result.mature_slips == 3
    assert result.definitive_unpaid == 2
    assert result.baseline == pytest.approx(2 / 3)
    # the bucket as usually reported also holds the censored row 4
    assert result.unpaid_slips == 3


def test_concentration_covers_the_whole_bucket_discrimination_only_the_mature(frame):
    tables = analyse(frame, SETTINGS).tables["agent"]
    assert tables.concentration["slips"].sum() == 3
    assert tables.discrimination["mature"].sum() == 3
    by_agent = tables.discrimination.set_index("group")
    assert by_agent.loc["a2", "rate"] == 1.0
    assert by_agent.loc["a1", "rate"] == 0.0  # row 1 paid; rows 4 and 5 are not mature


def test_every_factor_is_computed_and_every_factor_is_in_a_section(frame):
    result = analyse(frame, SETTINGS)
    assert set(result.tables) == set(FACTORS)
    assert {k for _, keys in SECTIONS for k in keys} == set(FACTORS)


def test_the_report_states_what_each_table_is_and_that_it_is_observational(frame):
    ds = dataset(_rows())
    text = render(frame, ds, analyse(frame, SETTINGS)).render()
    assert "Observational" in text
    assert "still inside the grace window" in text
    assert "whole-agreement value" in text
    agent_section = text.split("## 1. Agent")[1].split("## 2.")[0]
    assert "### " not in agent_section  # one factor in the section: no repeated sub-heading


def test_a_zero_balance_gets_its_own_group_instead_of_the_missing_ones():
    rows = [raw(1, original_debt=0.0), raw(2, original_debt=None), raw(3, original_debt=1000.0)]
    frame = F.analysis_frame(dataset(rows), SETTINGS)
    groups = analyse(frame, SETTINGS).tables["ratio"].concentration["group"].tolist()
    assert "(zero balance - ratio undefined)" in groups
    assert "(no debt link - ratio unknown)" in groups


def test_cli_writes_the_report(tmp_path):
    path = dataset(_rows()).save(tmp_path / "ds.json")
    code, out = run_cli("analysis", "--dataset", str(path), "--out-dir", str(tmp_path / "r"))
    assert code == 0
    report = (tmp_path / "r" / "analysis.md").read_text(encoding="utf-8")
    assert report.startswith("# Factor analysis")
    assert "baseline=" in out


def test_cli_refuses_an_empty_dataset(tmp_path):
    path = dataset([]).save(tmp_path / "ds.json")
    code, _ = run_cli("analysis", "--dataset", str(path), "--out-dir", str(tmp_path / "r"))
    assert code == 2
