"""Stage 07: cleaning diagnostics. Every check is tested on rows built to trigger it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from delinquency_eda import features as F
from delinquency_eda.config import Settings
from delinquency_eda.stages import s07_diagnose as D
from delinquency_eda.textutil import normalize_text

from clirun import run_cli
from frames import dataset, paid_on_time, raw

SETTINGS = Settings()


def frame(rows, **kwargs):
    return F.analysis_frame(dataset(rows, **kwargs), SETTINGS)


def test_text_normalization_folds_case_accents_and_spacing():
    assert normalize_text("  Pörtfolio   A-01\t") == "portfolio a-01"
    assert normalize_text("Façade") == "facade"
    assert normalize_text("Straße") == "strasse"


def test_whitespace_and_control_characters_are_counted_per_column():
    df = frame([raw(1, agent_name=" Agent 01 "), raw(2, agent_name="Agent\x0702"), raw(3)])
    assert D.whitespace_counts(df) == {"agent_name": 1}
    assert D.control_counts(df) == {"agent_name": 1}


def test_truncation_needs_several_different_values_at_the_maximum_length():
    cut = [raw(1, portfolio_label="Portfolio AAAAAAA"), raw(2, portfolio_label="Portfolio BBBBBBB")]
    repeated = [
        raw(3, portfolio_label="Portfolio CCCCCCC"),
        raw(4, portfolio_label="Portfolio CCCCCCC"),
    ]
    suspicious = D.truncation_suspects(frame(cut)).set_index("column")
    assert bool(suspicious.loc["portfolio_label", "suspect"])
    benign = D.truncation_suspects(frame(repeated)).set_index("column")
    assert not bool(benign.loc["portfolio_label", "suspect"])


def test_dates_that_cannot_be_parsed_are_counted_beyond_real_nulls():
    ds = dataset([raw(1), raw(2), paid_on_time(3)])
    ds.base[0]["paid_date"] = "31/03/2026"
    df = F.analysis_frame(ds, SETTINGS)
    failures = D.date_failures(df)
    assert failures["paid_date"] == 1
    assert failures["due_date"] == 0


def test_numeric_looking_keys_and_their_leading_zeros_are_counted():
    df = frame([raw(1, person_key="007"), raw(2, person_key="12"), raw(3, person_key="pk_3")])
    assert D.key_shape(df) == (2, 1)


def test_duplicate_counts_by_key():
    df = frame(
        [raw(1), raw(1), raw(2, product_line="consumer", record_id=1)], lines=("auto", "consumer")
    )
    counts = D.duplicate_counts(df)
    assert counts["whole row"] == 1
    assert counts["(product_line, record_id)"] == 1
    assert counts["(product_line, agreement_id, portfolio_id)"] == 1
    assert counts["record_id alone (collides across product lines)"] == 2


def test_a_null_that_predicts_the_outcome_is_flagged_as_not_missing_at_random():
    rows = [raw(i, original_debt=None) for i in range(1, 5)]  # all unpaid (mature)
    rows += [paid_on_time(i, original_debt=300.0) for i in range(5, 9)]
    table = D.missing_table(frame(rows), SETTINGS).set_index("column")
    debt = table.loc["original_debt"]
    assert debt["unpaid rate if null"] == 1.0 and debt["unpaid rate if present"] == 0.0
    assert "not missing at random" in debt["reading"]


def test_a_null_with_no_signal_is_reported_as_such():
    rows = [raw(1, portfolio_label=None), paid_on_time(2, portfolio_label=None)]
    rows += [raw(3), paid_on_time(4)]
    table = D.missing_table(frame(rows), SETTINGS).set_index("column")
    assert table.loc["portfolio_label", "reading"] == "no apparent signal"


def test_missing_table_only_lists_base_columns_with_nulls():
    table = D.missing_table(frame([raw(1), paid_on_time(2)]), SETTINGS)
    assert "ratio_band" not in table["column"].tolist()  # derived columns are not diagnosed


def test_values_that_differ_only_in_form_are_grouped():
    rows = [
        raw(1, portfolio_label="Portfolio A-01"),
        raw(2, portfolio_label="portfolio a-01 "),
        raw(3, portfolio_label="Pörtfolio A-01"),
        raw(4, portfolio_label="Portfolio B-01"),
    ]
    variants = D.form_variants(frame(rows))
    assert len(variants["portfolio_label"]) == 1
    assert len(variants["portfolio_label"][0]) == 3


def test_agent_name_and_login_must_map_one_to_one():
    rows = [raw(1), raw(2, agent_login="agent_99"), raw(3, agent_name="Someone Else")]
    assert D.agent_mapping_conflicts(frame(rows)) == (1, 1)


def test_suspect_agents_are_found_by_pattern_on_name_or_login():
    rows = [
        raw(1, agent_login="batch_loader", agent_name="Loader 01"),
        raw(2, agent_login="svc_2", agent_name="Import Service"),
        raw(3),
    ]
    suspects = D.suspect_agents(frame(rows), SETTINGS.suspect_agent_pattern)
    assert sorted(suspects["agent_login"]) == ["batch_loader", "svc_2"]
    assert D.suspect_agents(frame([raw(3)]), SETTINGS.suspect_agent_pattern).empty


def test_outliers_are_counted_above_the_iqr_fence():
    rows = [raw(i, first_installment_value=float(100 + i)) for i in range(1, 11)]
    rows.append(raw(11, first_installment_value=100000.0))
    table = D.outlier_table(frame(rows), 3.0).set_index("column")
    assert table.loc["first_installment_value", "above fence"] == 1


def test_the_full_diagnosis_and_report_change_nothing():
    rows = [raw(1, original_debt=0.0), raw(2, original_debt=-5.0), raw(3), paid_on_time(4)]
    df = frame(rows)
    before = df.copy()
    diagnosis = D.diagnose(df, SETTINGS)
    text = D.render(diagnosis, SETTINGS).render()
    pd.testing.assert_frame_equal(df, before)
    assert (diagnosis.zero_balance, diagnosis.negative_balance) == (1, 1)
    assert "Nothing was changed" in text
    assert diagnosis.columns == 24  # the base columns, not the derived ones


def test_the_report_never_prints_a_person_key():
    df = frame([raw(1, person_key="secret_person_key_1"), raw(2)])
    text = D.render(D.diagnose(df, SETTINGS), SETTINGS).render()
    assert "secret_person_key_1" not in text


@pytest.mark.parametrize("missing", [None, np.nan])
def test_missing_agent_fields_do_not_break_the_suspect_search(missing):
    df = frame([raw(1, agent_name=missing, agent_login=missing)])
    assert D.suspect_agents(df, SETTINGS.suspect_agent_pattern).empty


def test_cli_writes_the_report(tmp_path):
    path = dataset([raw(1), paid_on_time(2)]).save(tmp_path / "ds.json")
    code, out = run_cli("diagnose", "--dataset", str(path), "--out-dir", str(tmp_path / "r"))
    assert code == 0 and "rows=2" in out
    assert (tmp_path / "r" / "diagnose.md").read_text(encoding="utf-8").startswith("# Cleaning")
