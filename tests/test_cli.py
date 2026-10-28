"""The command line and the stage registry."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from delinquency_eda import __version__
from delinquency_eda.cli import build_parser
from delinquency_eda.stages import STAGES, by_command

from clirun import run_cli, sqlite_args
from frames import dataset, paid_on_time, raw

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_registry_order():
    assert [s.number for s in STAGES] == list(range(11))
    assert len({s.command for s in STAGES}) == 11
    assert [s.module.split("_")[0] for s in STAGES] == [f"s{n:02d}" for n in range(11)]


def test_stage_entry_points():
    for stage in STAGES:
        module = stage.load()
        assert callable(module.add_arguments) and callable(module.run), stage.command
        assert module.__doc__ and f"Stage {stage.number:02d}" in module.__doc__, stage.command


def test_needs_database_flag():
    for stage in STAGES:
        opens = "ctx.databases()" in inspect.getsource(stage.load())
        assert opens == stage.needs_database, stage.command


def test_unknown_command():
    assert by_command("model").number == 10
    with pytest.raises(KeyError):
        by_command("nonsense")


@pytest.mark.parametrize("stage", [s.command for s in STAGES])
def test_every_stage_has_help(stage, capsys):
    with pytest.raises(SystemExit) as info:
        build_parser().parse_args([stage, "--help"])
    assert info.value.code == 0
    assert f"Stage {by_command(stage).number:02d}" in capsys.readouterr().out


def test_global_option_position():
    before = build_parser().parse_args(["--dataset", "x.json", "--out-dir", "o", "censoring"])
    after = build_parser().parse_args(["censoring", "--dataset", "x.json", "--out-dir", "o"])
    assert (before.dataset, before.out_dir) == (after.dataset, after.out_dir) == ("x.json", "o")
    default = build_parser().parse_args(["censoring"])
    assert (default.dataset, default.out_dir) == ("data/dataset.json", "reports")


def test_source_position():
    assert build_parser().parse_args(["--source", "a=1", "schema"]).source == ["a=1"]
    assert build_parser().parse_args(["schema", "--source", "b=2"]).source == ["b=2"]
    both = build_parser().parse_args(["--source", "a=1", "schema", "--source", "b=2"])
    assert both.source == ["b=2"]  # give the sources once, in one place


def test_no_stage():
    code, out = run_cli()
    assert code == 2 and "usage: delinquency-eda" in out


def test_unknown_stage():
    with pytest.raises(SystemExit) as info:
        build_parser().parse_args(["frobnicate"])
    assert info.value.code == 2


def test_error_format(tmp_path, capsys):
    code, _ = run_cli("analysis", "--dataset", str(tmp_path / "missing.json"))
    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and "Traceback" not in err


BAD_DATES = [
    ["period", "--as-of", "31/03/2026"],
    ["base", "--from", "nope", "--to", "2026-03-31"],
    ["base", "--from", "2026-01-01", "--to", "2026-03-31", "--as-of", "2026-13-40"],
    ["drift", "--reference-date", "2026-13-40"],
    ["card", "--as-of", "yesterday"],
]


@pytest.mark.parametrize("stage_args", BAD_DATES, ids=lambda a: f"{a[0]}-{a[1]}")
def test_bad_date(stage_args, capsys):
    with pytest.raises(SystemExit) as info:
        build_parser().parse_args(stage_args)
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert "expected a date as YYYY-MM-DD" in err and "Traceback" not in err


def test_bad_date_as_module():
    env = {**os.environ, "PYTHONPATH": SRC}
    done = subprocess.run(
        [sys.executable, "-m", "delinquency_eda", "drift", "--reference-date", "2026-13-40"],
        capture_output=True, text=True, env=env, check=False,
    )  # fmt: skip
    assert done.returncode == 2
    assert "Traceback" not in done.stderr
    assert "expected a date as YYYY-MM-DD" in done.stderr


def test_env_supplies_sources(world, tmp_path):
    env = {"DEA_BACKEND": "sqlite", "DEA_SOURCES": f"auto={world}"}
    code, out = run_cli("--out-dir", str(tmp_path / "r"), "schema", environ=env)
    assert code == 0 and "auto: ready" in out


def test_env_file_precedence(world, tmp_path):
    env_file = tmp_path / "dea.env"
    env_file.write_text(f"DEA_BACKEND=sqlite\nDEA_SOURCES=wrong={tmp_path / 'nope.db'}\n")
    code, out = run_cli(
        "--env-file", str(env_file), "--source", f"auto={world}", "--out-dir", str(tmp_path / "r"),
        "schema",
    )  # fmt: skip
    assert code == 0 and "auto: ready" in out and "wrong" not in out


def test_settings_file(world, tmp_path):
    good = tmp_path / "good.json"
    good.write_text('{"generated_status_labels": ["ACTIVE", "NOT_A_STATUS"]}')
    args = [*sqlite_args(("auto", world)), "--out-dir", str(tmp_path / "r"), "schema"]
    code, out = run_cli("--config", str(good), *args)
    assert code == 1 and "NOT_A_STATUS" in out
    bad = tmp_path / "bad.json"
    bad.write_text('{"no_such_setting": 1}')
    code, _ = run_cli("--config", str(bad), *args)
    assert code == 2


def test_run_as_module():
    env = {**os.environ, "PYTHONPATH": SRC}
    done = subprocess.run(
        [sys.executable, "-m", "delinquency_eda", "--version"],
        capture_output=True, text=True, env=env, check=False,
    )  # fmt: skip
    assert done.returncode == 0
    assert done.stdout.strip() == f"delinquency-eda {__version__}"


# ---- degenerate datasets -------------------------------------------------------------------

OFFLINE = [
    ["analysis"],
    ["censoring"],
    ["drift", "--reference-date", "2026-03-20"],
    ["eda"],
    ["diagnose"],
    ["clean"],
    ["card", "--from-dataset"],
    ["model"],
]


@pytest.mark.parametrize("stage_args", OFFLINE, ids=lambda a: a[0])
def test_empty_dataset(stage_args, tmp_path, capsys):
    path = dataset([]).save(tmp_path / "empty.json")
    code, _ = run_cli(*stage_args, "--dataset", str(path), "--out-dir", str(tmp_path / "r"))
    assert code == 2
    assert "no rows" in capsys.readouterr().err


@pytest.mark.parametrize("stage_args", OFFLINE[:-1], ids=lambda a: a[0])
def test_no_history_tables(stage_args, tmp_path):
    rows = [raw(1), paid_on_time(2), raw(3, person_key=None, due_date="2026-03-29")]
    path = dataset(rows).save(tmp_path / "bare.json")
    code, _ = run_cli(*stage_args, "--dataset", str(path), "--out-dir", str(tmp_path / "r"))
    assert code == 0


def test_single_slip_dataset(tmp_path):
    path = dataset([raw(1)]).save(tmp_path / "one.json")
    for stage_args in (["analysis"], ["censoring"], ["eda"], ["diagnose"], ["clean"]):
        code, _ = run_cli(*stage_args, "--dataset", str(path), "--out-dir", str(tmp_path / "r"))
        assert code == 0, stage_args


@pytest.mark.parametrize("stage_args", OFFLINE, ids=lambda a: a[0])
def test_stages_check_the_seal(stage_args, tmp_path, capsys):
    path = dataset([raw(1), paid_on_time(2)]).save(tmp_path / "ds.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["base"][0]["first_installment_value"] = 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    code, _ = run_cli(*stage_args, "--dataset", str(path), "--out-dir", str(tmp_path / "r"))
    assert code == 2
    err = capsys.readouterr().err
    assert "does not match its seal" in err and "rebuild it with the base command" in err
    assert not (tmp_path / "r").exists(), "a refused dataset produces no report"


def test_old_dataset_refused(tmp_path, capsys):
    path = dataset([raw(1)]).save(tmp_path / "ds.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["meta"]["schema_version"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    code, _ = run_cli("analysis", "--dataset", str(path), "--out-dir", str(tmp_path / "r"))
    assert code == 2
    assert "rebuild it with the base command" in capsys.readouterr().err
