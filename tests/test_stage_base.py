"""Stage 02: the base build as a command, and what it promises about reproducibility."""

from __future__ import annotations

from delinquency_eda.dataset import Dataset
from delinquency_eda.stages.s02_base import render, summarize

from clirun import run_cli, sqlite_args
from frames import dataset, paid_on_time, raw

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
