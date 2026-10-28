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


def test_round_trip(tmp_path):
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
def test_save_needs_admission(tmp_path, admission):
    ds = dataset([raw(1)])
    ds.admission = admission
    with pytest.raises(DatasetError, match="did not pass the entry contract"):
        ds.save(tmp_path / "ds.json")
    assert not (tmp_path / "ds.json").exists()


def test_key_order(tmp_path):
    path = written(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(dict(reversed(payload.items()))), encoding="utf-8")
    assert len(Dataset.load(path).base) == 3


def test_edited_value(tmp_path):
    path = edited(written(tmp_path), lambda p: p["base"][0].update(first_installment_value=1.0))
    with pytest.raises(DatasetError, match="does not match its seal"):
        Dataset.load(path)


def test_edited_record(tmp_path):
    path = edited(written(tmp_path), lambda p: p["admission"].update(note="looks fine"))
    with pytest.raises(DatasetError, match="does not match its seal"):
        Dataset.load(path)


def test_cut_table(tmp_path):
    path = edited(written(tmp_path), lambda p: p["base"].pop())
    with pytest.raises(DatasetError, match=r"'base' has 2 rows and its seal says 3"):
        Dataset.load(path)


def test_added_row(tmp_path):
    path = edited(written(tmp_path), lambda p: p["contacts"].append({}))
    with pytest.raises(DatasetError, match=r"'contacts' has 1 rows and its seal says 0"):
        Dataset.load(path)


@pytest.mark.parametrize("seal", [None, {}, {"algorithm": "md5", "digest": "x"}, "sealed"])
def test_no_seal(tmp_path, seal):
    def change(payload):
        payload.pop("seal")
        if seal is not None:
            payload["seal"] = seal

    path = edited(written(tmp_path), change)
    with pytest.raises(DatasetError, match="has no seal"):
        Dataset.load(path)


def test_failed_admission(tmp_path):
    path = resealed(written(tmp_path), lambda p: p["admission"].update(passed=False))
    with pytest.raises(DatasetError, match="did not pass the entry contract"):
        Dataset.load(path)


def test_no_admission(tmp_path):
    path = resealed(written(tmp_path), lambda p: p.pop("admission"))
    with pytest.raises(DatasetError, match="did not pass the entry contract"):
        Dataset.load(path)


def test_old_version(tmp_path):
    path = edited(written(tmp_path), lambda p: p["meta"].update(schema_version=1))
    assert SCHEMA_VERSION == 2
    with pytest.raises(DatasetError, match="schema version 1 is not 2"):
        Dataset.load(path)


def test_meta_is_sealed(tmp_path):
    path = edited(written(tmp_path), lambda p: p["meta"].update(grace_days=30))
    with pytest.raises(DatasetError, match="does not match its seal"):
        Dataset.load(path)


def test_digest():
    body = {"a": [1, 2], "b": {"x": 1.5, "y": None}}
    assert content_digest(body) == content_digest({"b": {"y": None, "x": 1.5}, "a": [1, 2]})
    assert content_digest(body) != content_digest({**body, "a": [2, 1]})


def test_deterministic_save(tmp_path):
    first = written(tmp_path)
    second = dataset([raw(1), paid_on_time(2), raw(3, due_date="2026-03-20")]).save(
        tmp_path / "again.json"
    )
    assert first.read_bytes() == second.read_bytes()
