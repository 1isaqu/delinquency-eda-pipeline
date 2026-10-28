"""Stage 08: additive cleaning rules and the information-loss summary."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from delinquency_eda import features as F
from delinquency_eda.config import Settings
from delinquency_eda.dataset import BASE_COLUMNS
from delinquency_eda.errors import DatasetError
from delinquency_eda.stages.s08_clean import (
    agreement_key,
    analytic_view,
    clean_frame,
    meta_for,
    render,
    split_group,
    summarize,
    system_flag,
)

from clirun import run_cli
from frames import dataset, paid_on_time, raw

SETTINGS = Settings(system_agents=("batch_loader", "Café Robot"))


def _rows():
    return [
        paid_on_time(1),
        raw(2, original_debt=0.0),  # a real zero balance
        raw(3, original_debt=None),  # no debt link
        raw(4, agent_login="Batch_Loader", agent_name="Loader 01"),  # listed system agent
        raw(5, agent_name="  cafe robot ", agent_login="svc_5"),  # same name, other case/accents
        raw(6, person_key=None, debtor_id=60),  # no person key
        raw(7, person_key=None, debtor_id=70),  # another one: must not pool with row 6
    ]


@pytest.fixture
def full():
    return clean_frame(dataset(_rows()), SETTINGS)


def test_additive_cleaning(full):
    assert len(full) == 7
    assert set(BASE_COLUMNS) <= set(full.columns)
    for added in ("agreement_key", "is_system", "zero_balance", "group_key", "is_mature"):
        assert added in full.columns


def test_zero_balance_flag(full):
    by_id = full.set_index("agreement_id")
    assert bool(by_id.loc[2, "zero_balance"]) and not bool(by_id.loc[3, "zero_balance"])
    assert by_id.loc[2, "original_debt"] == 0.0
    assert pd.isna(by_id.loc[3, "original_debt"])
    assert pd.isna(by_id.loc[2, "first_installment_ratio"])  # undefined, not 0 and not imputed
    assert pd.isna(by_id.loc[3, "first_installment_ratio"])


def test_system_agent_match(full):
    flagged = full.set_index("agreement_id")["is_system"]
    assert flagged[[4, 5]].all()
    assert not flagged[[1, 2, 3, 6, 7]].any()


def test_no_listed_agents():
    frame = clean_frame(dataset(_rows()), Settings())
    assert not frame["is_system"].any()


def test_analytic_view(full):
    view = analytic_view(full)
    assert len(view) == 5
    assert not view["is_system"].any()
    assert len(full) == 7  # the full table is untouched


def test_agreement_key():
    ds = dataset([raw(1), raw(1, product_line="consumer")], lines=("auto", "consumer"))
    frame = clean_frame(ds, Settings())
    assert frame["agreement_key"].tolist() == ["auto|1|1", "consumer|1|1"]
    assert agreement_key(frame).is_unique


def test_repeated_key():
    with pytest.raises(DatasetError, match="grain violated"):
        clean_frame(dataset([raw(1), raw(1, record_id=99)]), Settings())


def test_keyless_slips(full):
    groups = full.set_index("agreement_id")["group_key"]
    assert groups[6] != groups[7]
    assert groups[6] == "auto|debtor|60"
    assert groups[1] == "pk_1"
    assert split_group(full).equals(full["group_key"])


def test_system_flag_no_match():
    frame = dataset([raw(1)]).base_frame()
    assert not system_flag(frame, ("someone_else",)).any()


def test_summary(full):
    ds = dataset(_rows())
    s = summarize(ds, full)
    assert (s.rows_before, s.rows_after_full) == (7, 7)
    assert s.system_rows == 2 and s.zero_balance_rows == 1
    assert s.columns_after > s.columns_before
    assert "agreement_key" in s.added
    assert s.key_unique
    text = render(s).render()
    assert "| rows removed | 0 |" in text
    assert "| cells imputed | 0 |" in text
    assert "| columns removed | 0 |" in text


def test_meta_file():
    ds = dataset([raw(1)])
    meta = meta_for(ds, SETTINGS)
    assert meta["leakage_columns"] == list(F.LEAKAGE_COLUMNS)
    assert meta["explanatory_only_columns"] == list(F.EXPLANATORY_ONLY_COLUMNS)
    assert meta["group_for_split"] == "group_key"
    assert meta["grace_days"] == 3


def test_cli_outputs(tmp_path):
    rows = [raw(1, person_key="007"), raw(2, person_key="0042", agent_login="batch_loader")]
    path = dataset(rows).save(tmp_path / "ds.json")
    config = tmp_path / "settings.json"
    config.write_text(json.dumps({"system_agents": ["batch_loader"]}), encoding="utf-8")
    out_dir = tmp_path / "r"
    code, out = run_cli(
        "clean", "--dataset", str(path), "--out-dir", str(out_dir), "--config", str(config)
    )
    assert code == 0 and "analytic=1" in out
    full = pd.read_csv(out_dir / "clean_full.csv", dtype=str)
    assert full["person_key"].tolist() == ["007", "0042"]  # leading zeros survive the file
    assert len(pd.read_csv(out_dir / "clean_analytic.csv")) == 1
    meta = json.loads((out_dir / "clean_meta.json").read_text(encoding="utf-8"))
    assert meta["system_agents"] == ["batch_loader"]
    assert (out_dir / "clean.md").exists()
