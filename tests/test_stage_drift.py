"""Stage 04: replaying the buckets at an earlier date."""

from __future__ import annotations

from datetime import date

import pytest

from delinquency_eda import derive
from delinquency_eda.errors import ConfigError
from delinquency_eda.stages.s04_drift import (
    NOT_YET,
    bucket_at,
    closest_measure,
    measure_sums,
    render,
    replay,
)

from clirun import run_cli
from frames import AS_OF, GRACE, dataset, raw

REFERENCE = date(2026, 3, 29)

ROWS = [
    # due 03-28, paid 03-30: unpaid on 03-29, paid inside the grace window by 03-31 -> a mover
    raw(1, due_date="2026-03-28", paid_amount=100.0, paid_date="2026-03-30", total_value=300.0),
    # the agreement did not exist yet on 03-29
    raw(2, agreement_date="2026-03-30", due_date="2026-04-02"),
    # unpaid on both dates
    raw(3, due_date="2026-03-05", total_value=700.0),
    # not due on 03-29, overdue and unpaid by 03-31 -> entered the bucket
    raw(4, due_date="2026-03-30"),
]


@pytest.fixture
def frame():
    return dataset(ROWS).base_frame()


def test_payment_after_reference(frame):
    then = bucket_at(frame, REFERENCE, GRACE)
    now = bucket_at(frame, AS_OF, GRACE)
    assert then.iloc[0] == derive.OVERDUE_UNPAID
    assert now.iloc[0] == derive.PAID_ON_TIME


def test_agreement_after_reference(frame):
    assert bucket_at(frame, REFERENCE, GRACE).iloc[1] == NOT_YET
    assert bucket_at(frame, AS_OF, GRACE).iloc[1] == derive.TO_MATURE


def test_replay_present(frame):
    same = replay(frame, AS_OF, AS_OF, GRACE)
    assert same.movers.empty and same.entered == 0
    assert (same.then == same.current).all()


def test_movers_and_entrants(frame):
    result = replay(frame, REFERENCE, AS_OF, GRACE)
    assert result.movers["agreement_id"].tolist() == [1]
    assert result.entered == 1  # slip 4
    assert result.unpaid_now() - result.unpaid_then() == result.entered - len(result.movers)


def test_transition_matrix(frame):
    result = replay(frame, REFERENCE, AS_OF, GRACE)
    assert int(result.transitions.to_numpy().sum()) == len(frame)
    assert result.transitions.loc[derive.OVERDUE_UNPAID, derive.PAID_ON_TIME] == 1
    assert result.transitions.loc[NOT_YET, derive.TO_MATURE] == 1


def test_reference_after_dataset(frame):
    with pytest.raises(ConfigError, match="after the dataset reference date"):
        replay(frame, date(2026, 4, 1), AS_OF, GRACE)


def test_value_measures(frame):
    result = replay(frame, REFERENCE, AS_OF, GRACE)
    sums = measure_sums(frame, result.current)
    # unpaid now: slips 3 and 4. first installment 100 + 100, whole agreement 700 + 100.
    assert sums["first installment"] == pytest.approx(200.0)
    assert sums["whole agreement"] == pytest.approx(800.0)
    ranked = closest_measure(805.0, sums)
    assert ranked[0][0] == "whole agreement"
    assert ranked[0][2] == pytest.approx(5 / 805)
    assert closest_measure(0.0, sums)[0][2] == float("inf")


def test_headline_measure(frame):
    result = replay(frame, REFERENCE, AS_OF, GRACE)
    text = render(frame, result, headline_value=800.0, headline_count=2).render()
    assert "Closest: **whole agreement**" in text
    assert "Headline count given: 2" in text
    assert "Movers" in text
    assert "left or are not rewound" not in text  # no stray template text


def test_cli_replays_a_dataset_file(tmp_path):
    path = dataset(ROWS).save(tmp_path / "ds.json")
    base = ["drift", "--dataset", str(path), "--out-dir", str(tmp_path / "r")]
    code, out = run_cli(*base, "--reference-date", "2026-03-29", "--headline-value", "800")
    assert code == 0
    assert "left: 1, entered: 1" in out
    assert "whole agreement" in (tmp_path / "r" / "drift.md").read_text(encoding="utf-8")
    code, _ = run_cli(*base, "--reference-date", "2026-05-01")
    assert code == 2
