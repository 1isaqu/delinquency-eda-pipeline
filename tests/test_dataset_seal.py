"""The dataset file: only an admitted dataset is written, and only an intact one is read."""

from __future__ import annotations

import json

import pytest

from delinquency_eda.dataset import SCHEMA_VERSION, Dataset, content_digest, make_seal
from delinquency_eda.errors import DatasetError

from frames import FIXTURE_ADMISSION, dataset, paid_on_time, raw


def written(tmp_path, rows=None):
    ds = dataset(rows or [raw(1), paid_on_time(2), raw(3, due_date="2026-03-20")])
    return ds.save(tmp_path / "ds.json")


def edited(path, change):
    """Rewrite the file after ``change(payload)``; the seal is left as it was."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def resealed(path, change):
    """Rewrite the file after ``change(payload)`` and give it a seal that matches."""

    def apply(payload):
        change(payload)
        body = {k: v for k, v in payload.items() if k != "seal"}
        payload["seal"] = make_seal(body)

    return edited(path, apply)


def test_an_admitted_dataset_round_trips_with_its_record_and_seal(tmp_path):
    path = written(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["seal"]["algorithm"] == "sha256"
    assert payload["seal"]["rows"] == {"base": 3, "debtors": 0, "contacts": 0, "breaks": 0}
    assert payload["admission"] == FIXTURE_ADMISSION
    loaded = Dataset.load(path)
    assert loaded.admission == FIXTURE_ADMISSION
    assert [r["record_id"] for r in loaded.base] == [1, 2, 3]


@pytest.mark.parametrize(
    "admission",
    [None, {}, {"passed": False}, {"passed": "yes"}, {"passed": 1.0}],
    ids=["none", "empty", "failed", "text", "number"],
)
def test_a_dataset_that_was_not_admitted_cannot_be_written(tmp_path, admission):
    ds = dataset([raw(1)])
    ds.admission = admission
    with pytest.raises(DatasetError, match="did not pass the entry contract"):
        ds.save(tmp_path / "ds.json")
    assert not (tmp_path / "ds.json").exists()


def test_the_seal_does_not_depend_on_the_order_of_keys_in_the_file(tmp_path):
    path = written(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(dict(reversed(payload.items()))), encoding="utf-8")
    assert len(Dataset.load(path).base) == 3


def test_a_changed_value_breaks_the_seal(tmp_path):
    path = edited(written(tmp_path), lambda p: p["base"][0].update(first_installment_value=1.0))
    with pytest.raises(DatasetError, match="does not match its seal"):
        Dataset.load(path)


def test_a_changed_admission_record_breaks_the_seal(tmp_path):
    path = edited(written(tmp_path), lambda p: p["admission"].update(note="looks fine"))
    with pytest.raises(DatasetError, match="does not match its seal"):
        Dataset.load(path)


def test_a_cut_table_is_reported_by_name_and_count(tmp_path):
    path = edited(written(tmp_path), lambda p: p["base"].pop())
    with pytest.raises(DatasetError, match=r"'base' has 2 rows and its seal says 3"):
        Dataset.load(path)


def test_an_added_row_is_caught_too(tmp_path):
    path = edited(written(tmp_path), lambda p: p["contacts"].append({}))
    with pytest.raises(DatasetError, match=r"'contacts' has 1 rows and its seal says 0"):
        Dataset.load(path)


@pytest.mark.parametrize("seal", [None, {}, {"algorithm": "md5", "digest": "x"}, "sealed"])
def test_a_file_without_a_usable_seal_is_refused(tmp_path, seal):
    def change(payload):
        payload.pop("seal")
        if seal is not None:
            payload["seal"] = seal

    path = edited(written(tmp_path), change)
    with pytest.raises(DatasetError, match="has no seal"):
        Dataset.load(path)


def test_a_file_whose_admission_failed_is_refused_even_when_it_is_sealed(tmp_path):
    path = resealed(written(tmp_path), lambda p: p["admission"].update(passed=False))
    with pytest.raises(DatasetError, match="did not pass the entry contract"):
        Dataset.load(path)


def test_a_file_without_an_admission_record_is_refused_even_when_it_is_sealed(tmp_path):
    path = resealed(written(tmp_path), lambda p: p.pop("admission"))
    with pytest.raises(DatasetError, match="did not pass the entry contract"):
        Dataset.load(path)


def test_a_file_from_before_the_contract_is_refused_as_an_old_version(tmp_path):
    path = edited(written(tmp_path), lambda p: p["meta"].update(schema_version=1))
    assert SCHEMA_VERSION == 2
    with pytest.raises(DatasetError, match="schema version 1 is not 2"):
        Dataset.load(path)


def test_the_digest_covers_the_meta_block(tmp_path):
    path = edited(written(tmp_path), lambda p: p["meta"].update(grace_days=30))
    with pytest.raises(DatasetError, match="does not match its seal"):
        Dataset.load(path)


def test_the_digest_is_a_function_of_content_only():
    body = {"a": [1, 2], "b": {"x": 1.5, "y": None}}
    assert content_digest(body) == content_digest({"b": {"y": None, "x": 1.5}, "a": [1, 2]})
    assert content_digest(body) != content_digest({**body, "a": [2, 1]})


def test_saving_twice_gives_the_same_bytes(tmp_path):
    first = written(tmp_path)
    second = dataset([raw(1), paid_on_time(2), raw(3, due_date="2026-03-20")]).save(
        tmp_path / "again.json"
    )
    assert first.read_bytes() == second.read_bytes()
