"""The numbered stages.

Each stage module exposes ``add_arguments(parser)`` for its own options and ``run(ctx, args)``
returning a process exit code. Pure computation lives in plain functions next to ``run`` so the
tests can call it without a database or a command line.

The registry below is the single list the command line, the README table and the tests read.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType


@dataclass(frozen=True)
class Stage:
    number: int
    command: str
    module: str
    needs_database: bool
    summary: str

    def load(self) -> ModuleType:
        return import_module(f"delinquency_eda.stages.{self.module}")


STAGES: tuple[Stage, ...] = (
    Stage(
        0, "period", "s00_period", True, "Find the date window that reproduces a reference funnel"
    ),
    Stage(1, "schema", "s01_schema", True, "Check the database against the data contract"),
    Stage(
        2, "base", "s02_base", True, "Extract the dataset bundle (slips, people, contacts, breaks)"
    ),
    Stage(
        3, "analysis", "s03_analysis", False, "Concentration and discrimination tables per factor"
    ),
    Stage(4, "drift", "s04_drift", False, "Replay the buckets at an earlier date and reconcile"),
    Stage(
        5, "censoring", "s05_censoring", False, "Split the unpaid bucket into still-open and lost"
    ),
    Stage(6, "eda", "s06_eda", False, "Exploratory report: profile, quality, leakage, outliers"),
    Stage(7, "diagnose", "s07_diagnose", False, "Cleaning diagnostics, report only"),
    Stage(8, "clean", "s08_clean", False, "Apply the additive cleaning rules and write the tables"),
    Stage(9, "card", "s09_card", True, "Recompute the summary card with a maturity cut"),
    Stage(
        10,
        "model",
        "s10_term_model",
        False,
        "Days-to-pay model: effect of the term, mix controlled",
    ),
)


def by_command(command: str) -> Stage:
    for stage in STAGES:
        if stage.command == command:
            return stage
    raise KeyError(command)
