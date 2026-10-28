"""Stage 02: the base build as a command, and what it promises about reproducibility."""

from __future__ import annotations

import json

from delinquency_eda.cli import build_parser
from delinquency_eda.dataset import Dataset
from delinquency_eda.stages.s02_base import render, summarize

from clirun import run_cli, sqlite_args
from frames import dataset, paid_on_time, raw
from world import damage

BUILD = ["base", "--from", "2026-03-01", "--to", "2026-03-31", "--as-of", "2026-03-31"]


def test_summary_counts_slips_people_and_buckets():
    ds = dataset(
        [paid_on_time(1), raw(2), raw(3, person_key=None), raw(4, due_date="2026-04-20")],
    )
    s = summarize(ds)
    assert s.slips == 4
    assert s.by_bucket == {"paid_on_time": 1, "overdue_unpaid": 2, "to_mature": 1}
    assert s.by_line == {"auto": 4}
    assert (s.people, s.without_person_key) == (3, 1)
    assert s.first_installment_value == 400.0


def test_the_report_warns_about_keeping_the_dataset_out_of_version_control():
    ds = dataset([raw(1)])
    text = render(ds, summarize(ds)).render()
    assert "version control" in text
    assert "2026-03-31" in text


def test_cli_extracts_the_world_and_saves_a_loadable_dataset(world, tmp_path):
    target = tmp_path / "data" / "dataset.json"
    code, out = run_cli(
        *sqlite_args(("auto", world)),
        "--dataset", str(target),
        "--out-dir", str(tmp_path / "r"),
        *BUILD,
    )  # fmt: skip
    assert code == 0
    assert "(5 slips)" in out
    loaded = Dataset.load(target)
    assert len(loaded.base) == 5
    assert loaded.meta.as_of == "2026-03-31"
    assert (tmp_path / "r" / "base.md").exists()


def test_the_same_command_and_date_give_a_byte_identical_dataset(world, tmp_path):
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    for target in (first, second):
        code, _ = run_cli(
            *sqlite_args(("auto", world)),
            "--dataset", str(target), "--out-dir", str(tmp_path / "r"), *BUILD,
        )  # fmt: skip
        assert code == 0
    assert first.read_bytes() == second.read_bytes()


def test_a_later_reference_date_changes_outcomes_not_membership(world, tmp_path):
    early, late = tmp_path / "early.json", tmp_path / "late.json"
    for target, as_of in ((early, "2026-03-11"), (late, "2026-03-31")):
        args = ["base", "--from", "2026-03-01", "--to", "2026-03-31", "--as-of", as_of]
        code, _ = run_cli(
            *sqlite_args(("auto", world)),
            "--dataset", str(target), "--out-dir", str(tmp_path / "r"), *args,
        )  # fmt: skip
        assert code == 0
    a, b = Dataset.load(early), Dataset.load(late)
    assert {r["agreement_id"] for r in a.base} == {r["agreement_id"] for r in b.base}
    assert [r["bucket"] for r in a.base] != [r["bucket"] for r in b.base]


def test_a_reversed_period_is_reported_not_raised(world, tmp_path):
    code, _ = run_cli(
        *sqlite_args(("auto", world)),
        "--dataset", str(tmp_path / "d.json"),
        "--out-dir", str(tmp_path / "r"),
        "base", "--from", "2026-03-31", "--to", "2026-03-01",
    )  # fmt: skip
    assert code == 2


# ---- the gate ------------------------------------------------------------------------------


def build(world, tmp_path, name="dataset.json"):
    target = tmp_path / name
    code, out = run_cli(
        *sqlite_args(("auto", world)),
        "--dataset", str(target), "--out-dir", str(tmp_path / "r"), *BUILD,
    )  # fmt: skip
    return code, out, target


def test_admitted_build(world, tmp_path):
    code, out, target = build(world, tmp_path)
    assert code == 0 and "entry contract: all 15 rules passed" in out
    assert (tmp_path / "r" / "base.md").exists()
    admission = (tmp_path / "r" / "admission.md").read_text(encoding="utf-8")
    assert "**Result: ADMITTED.**" in admission
    assert "15 of 15 rules passed" in (tmp_path / "r" / "base.md").read_text(encoding="utf-8")
    record = json.loads(target.read_text(encoding="utf-8"))["admission"]
    assert record["passed"] is True and len(record["rules"]) == 15


def test_rejected_build(world, tmp_path):
    damage(world, "UPDATE agreements SET paid_amount = 0 WHERE row_id = 1")
    code, out, target = build(world, tmp_path)
    assert code == 1
    assert not target.exists()
    assert "rejected: 1 entry rule(s) failed; no dataset was written" in out
    assert "row.payment: 1 violation(s)" in out
    assert "not judged, because an earlier rule failed" in out
    assert not (tmp_path / "r" / "base.md").exists(), "base.md describes a missing dataset"
    admission = (tmp_path / "r" / "admission.md").read_text(encoding="utf-8")
    assert "**Result: REJECTED.**" in admission and "row.payment" in admission


def test_rejection_keeps_old_file(world, tmp_path):
    code, _, target = build(world, tmp_path)
    assert code == 0
    before = target.read_bytes()
    damage(world, "DELETE FROM debtors WHERE debtor_id = 2")
    code, out, _ = build(world, tmp_path)
    assert code == 1
    assert target.read_bytes() == before
    assert f"the file at {target} was left as it was" in out
    assert "was left as it was" in (tmp_path / "r" / "admission.md").read_text(encoding="utf-8")


def test_bad_source_not_queried(world, tmp_path):
    damage(world, "DROP TABLE contact_results")
    code, out, target = build(world, tmp_path)
    assert code == 1 and not target.exists()
    assert "source.tables: 1 violation(s)" in out
    assert "auto.contact_results: table not found" in out


def test_no_bypass_option():
    """The options of ``base`` are its window and its date, and the options every stage shares."""
    parser = build_parser()
    base = next(
        action.choices["base"]
        for action in parser._actions
        if hasattr(action, "choices") and action.choices and "base" in action.choices
    )
    shared = {
        "-h", "--help", "--backend", "--source", "--env-file", "--config", "--dataset", "--out-dir",
    }  # fmt: skip
    flags = {opt for action in base._actions for opt in action.option_strings}
    assert flags - shared == {"--from", "--to", "--as-of"}
