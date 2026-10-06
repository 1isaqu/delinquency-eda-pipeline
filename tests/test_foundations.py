"""Bands, dates, settings, SQL helpers and the Markdown builder."""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal

import pandas as pd
import pytest

from delinquency_eda.bands import Bands
from delinquency_eda.config import Settings
from delinquency_eda.dates import iso, parse_date_arg, to_date, to_float, to_id
from delinquency_eda.errors import ConfigError, DatasetError, ReadOnlyViolation
from delinquency_eda.report import Doc, fmt_pct, fmt_rate, md_table
from delinquency_eda.sqlutil import chunked, ensure_read_only, placeholders

# ---- bands ---------------------------------------------------------------------------------


def test_int_bands_are_right_closed_and_labelled():
    bands = Bands((0, 2, 5))
    assert bands.labels == ["<=0", "1-2", "3-5", ">5"]
    assert [bands.label(v) for v in (-3, 0, 1, 2, 3, 5, 6)] == [
        "<=0", "<=0", "1-2", "1-2", "3-5", "3-5", ">5",
    ]  # fmt: skip


def test_single_value_band_label_is_not_a_range():
    assert Bands((1, 2, 6)).labels == ["<=1", "2", "3-6", ">6"]


def test_money_and_ratio_labels():
    assert Bands((100, 2500.5), "money").labels == ["<=100", "100-2,500.5", ">2,500.5"]
    ratio = Bands((0.05, 0.5), "ratio")
    assert ratio.labels == ["<=5%", "5-50%", ">50%"]
    assert ratio.label(0.5) == "5-50%"
    assert ratio.label(0.51) == ">50%"


def test_missing_values_have_no_band():
    bands = Bands((0, 2))
    assert bands.label(None) is None
    assert bands.label(float("nan")) is None
    assert bands.assign([1, None, 3]) == ["1-2", None, ">2"]


def test_edges_must_increase():
    with pytest.raises(ValueError):
        Bands((3, 3))
    with pytest.raises(ValueError):
        Bands(())


# ---- dates ---------------------------------------------------------------------------------


def test_to_date_accepts_every_driver_shape():
    assert to_date(datetime(2026, 3, 1, 23, 59)) == date(2026, 3, 1)
    assert to_date(date(2026, 3, 1)) == date(2026, 3, 1)
    assert to_date("2026-03-01 10:11:12") == date(2026, 3, 1)
    assert to_date("2026-03-01T10:11:12") == date(2026, 3, 1)
    assert to_date(pd.Timestamp("2026-03-01")) == date(2026, 3, 1)


@pytest.mark.parametrize("empty", [None, "", "  ", "nan", float("nan"), "NaT"])
def test_to_date_treats_nulls_as_none(empty):
    assert to_date(empty) is None
    assert iso(empty) is None


def test_to_date_rejects_garbage():
    with pytest.raises(DatasetError):
        to_date("not a date")


def test_numbers_and_ids():
    assert to_float(Decimal("12.50")) == 12.5
    assert to_float("3") == 3.0
    assert to_float(None) is None
    assert to_float(float("nan")) is None
    with pytest.raises(DatasetError):
        to_float("abc")
    assert to_id(Decimal("7")) == 7
    assert isinstance(to_id(Decimal("7")), int)
    assert to_id(" ab ") == "ab"


def test_parse_date_arg():
    assert parse_date_arg("2026-03-31") == date(2026, 3, 31)
    with pytest.raises(ConfigError):
        parse_date_arg("31/03/2026")


# ---- settings ------------------------------------------------------------------------------


def test_defaults_validate():
    settings = Settings().validate()
    assert settings.term_bands().labels[0] == "<=0"
    assert settings.grace_days >= 0


def test_settings_from_mapping_overrides_and_converts_lists():
    s = Settings.from_mapping({"grace_days": 7, "term_band_edges": [1, 4], "model": {"folds": 3}})
    assert s.grace_days == 7
    assert s.term_band_edges == (1, 4)
    assert s.model.folds == 3
    assert s.model.random_state == 42


def test_settings_reject_unknown_and_invalid_values():
    with pytest.raises(ConfigError, match="unknown setting"):
        Settings.from_mapping({"grace": 3})
    with pytest.raises(ConfigError, match="unknown model setting"):
        Settings.from_mapping({"model": {"fold": 3}})
    with pytest.raises(ConfigError, match="grace_days"):
        Settings.from_mapping({"grace_days": -1})
    with pytest.raises(ConfigError, match="strictly increasing"):
        Settings.from_mapping({"term_band_edges": [5, 5]})
    with pytest.raises(ConfigError, match="regular expression"):
        Settings.from_mapping({"portfolio_group_pattern": "("})
    with pytest.raises(ConfigError, match="min_n_weak"):
        Settings.from_mapping({"min_n_weak": 50, "min_n_reliable": 10})
    with pytest.raises(ConfigError, match=r"model\.folds"):
        Settings.from_mapping({"model": {"folds": 1}})


def test_settings_from_file(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"grace_days": 9}), encoding="utf-8")
    assert Settings.from_file(path).grace_days == 9
    path.write_text("[1]", encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON object"):
        Settings.from_file(path)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        Settings.from_file(path)
    with pytest.raises(ConfigError, match="cannot read"):
        Settings.from_file(tmp_path / "missing.json")


# ---- sql helpers ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "  with x as (select 1 as a) select a from x",
        "SELECT 'DELETE FROM t' AS note -- DROP TABLE x",
        "SELECT created_at, updated_by, into_account FROM t",
    ],
)
def test_reads_pass_the_guard(sql):
    ensure_read_only(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM t",
        "UPDATE t SET a = 1",
        "SELECT * INTO copy FROM t",
        "SELECT 1; SELECT 2",
        "WITH x AS (SELECT 1) DELETE FROM t",
        "EXEC sp_who",
        "DROP TABLE t",
        "",
    ],
)
def test_writes_are_refused(sql):
    with pytest.raises(ReadOnlyViolation):
        ensure_read_only(sql)


def test_placeholders_and_chunks():
    assert placeholders(3) == "?, ?, ?"
    with pytest.raises(ValueError):
        placeholders(0)
    assert list(chunked(range(5), 2)) == [[0, 1], [2, 3], [4]]
    assert list(chunked([], 3)) == []
    with pytest.raises(ValueError):
        list(chunked([1], 0))


# ---- markdown ------------------------------------------------------------------------------


def test_md_table_formats_cells():
    frame = pd.DataFrame({"name": ["a|b", None], "n": [1234, 5], "x": [1.5, float("nan")]})
    text = md_table(frame)
    lines = text.splitlines()
    assert lines[0] == "| name | n | x |"
    assert lines[2] == "| a\\|b | 1,234 | 1.50 |"
    assert lines[3] == "|  | 5 |  |"


def test_md_table_from_rows_and_alignment():
    text = md_table([[1, True]], ["a", "b"], align_right=["a"])
    assert text.splitlines()[1] == "|---:|---|"
    assert text.splitlines()[2] == "| 1 | yes |"


def test_percent_helpers():
    assert fmt_pct(1, 4) == "25.0%"
    assert fmt_pct(1, 0) == "-"
    assert fmt_rate(0.256) == "25.6%"
    assert fmt_rate(None) == "-"


def test_doc_renders_and_saves(tmp_path):
    doc = Doc().h1("Title").p("text").bullets(["x", "y"]).quote("note")
    text = doc.render()
    assert text.startswith("# Title\n\ntext\n\n- x\n- y\n\n> note")
    target = doc.save(tmp_path / "nested" / "out.md")
    assert target.read_text(encoding="utf-8") == text
