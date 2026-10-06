"""Pipeline settings.

Every number here that a real business would calibrate (grace period, band edges, status labels,
agent patterns, reliability floors) has an invented default. The defaults exist so the demo and the
tests run; they are not recommendations. Calibrate them on your own data and keep them in a JSON
file passed with ``--config``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from delinquency_eda.bands import Bands
from delinquency_eda.errors import ConfigError


@dataclass(frozen=True)
class ModelSettings:
    """Hyperparameters and split rules of the payment-term model (stage 10)."""

    folds: int = 5
    random_state: int = 42
    logistic_c: float = 1.0
    logistic_max_iter: int = 2000
    boosting_max_iter: int = 300
    boosting_learning_rate: float = 0.06
    min_category_count: int = 20
    holdout_quantile: float = 0.75
    permutation_repeats: int = 20

    def validate(self) -> None:
        if self.folds < 2:
            raise ConfigError("model.folds must be at least 2")
        if not 0.0 < self.holdout_quantile < 1.0:
            raise ConfigError("model.holdout_quantile must be between 0 and 1 (exclusive)")
        if self.logistic_c <= 0 or self.boosting_learning_rate <= 0:
            raise ConfigError("model.logistic_c and model.boosting_learning_rate must be positive")
        if min(self.logistic_max_iter, self.boosting_max_iter, self.permutation_repeats) < 1:
            raise ConfigError("model iteration and repeat counts must be at least 1")
        if self.min_category_count < 1:
            raise ConfigError("model.min_category_count must be at least 1")


@dataclass(frozen=True)
class Settings:
    """Rules and thresholds shared by the stages. See docs/data-contract.md for the vocabulary."""

    # --- what counts as paid on time, and which agreements are in scope -------------------
    grace_days: int = 3
    generated_status_labels: tuple[str, ...] = (
        "ACTIVE",
        "BROKEN",
        "PAID",
        "AUTO_BROKEN",
        "PAID_MANUAL",
    )
    broken_status_labels: tuple[str, ...] = ("BROKEN", "AUTO_BROKEN")
    manual_break_label: str = "BROKEN"
    excluded_agent_patterns: tuple[str, ...] = ("system%", "test%")
    system_agents: tuple[str, ...] = ()
    id_chunk_size: int = 500

    # --- bands (right-closed upper edges) --------------------------------------------------
    term_band_edges: tuple[int, ...] = (0, 2, 5, 10, 20)
    value_band_edges: tuple[float, ...] = (100, 500, 2000, 10000)
    ratio_band_edges: tuple[float, ...] = (0.05, 0.2, 0.5)
    installment_band_edges: tuple[int, ...] = (1, 3, 6, 12, 24)

    # --- reporting and diagnostics ---------------------------------------------------------
    min_n_reliable: int = 30
    min_n_weak: int = 10
    chronic_break_threshold: int = 4
    mnar_gap_pp: float = 10.0
    rare_category_min_n: int = 25
    outlier_iqr_k: float = 3.0
    portfolio_group_pattern: str = r"[\s-]+(?:[IVXLCDM]+|\d+)$"
    suspect_agent_pattern: str = r"system|robot|bot|batch|auto|test|import"

    model: ModelSettings = field(default_factory=ModelSettings)

    # ----------------------------------------------------------------------------------------
    def term_bands(self) -> Bands:
        return Bands(tuple(self.term_band_edges), "int")

    def value_bands(self) -> Bands:
        return Bands(tuple(self.value_band_edges), "money")

    def ratio_bands(self) -> Bands:
        return Bands(tuple(self.ratio_band_edges), "ratio")

    def installment_bands(self) -> Bands:
        return Bands(tuple(self.installment_band_edges), "int")

    def validate(self) -> Settings:
        """Raise ``ConfigError`` for values that make no sense; return ``self`` for chaining."""
        if self.grace_days < 0:
            raise ConfigError("grace_days must be zero or positive")
        if not self.generated_status_labels:
            raise ConfigError("generated_status_labels must not be empty")
        if not 1 <= self.id_chunk_size <= 1000:
            raise ConfigError("id_chunk_size must be between 1 and 1000 (driver parameter limits)")
        if self.min_n_weak > self.min_n_reliable:
            raise ConfigError("min_n_weak must not exceed min_n_reliable")
        if self.chronic_break_threshold < 2:
            raise ConfigError("chronic_break_threshold must be at least 2")
        for name, kind in (
            ("term_band_edges", "int"),
            ("value_band_edges", "money"),
            ("ratio_band_edges", "ratio"),
            ("installment_band_edges", "int"),
        ):
            try:
                Bands(tuple(getattr(self, name)), kind)  # type: ignore[arg-type]
            except ValueError as exc:
                raise ConfigError(f"{name}: {exc}") from None
        for name in ("portfolio_group_pattern", "suspect_agent_pattern"):
            try:
                re.compile(getattr(self, name))
            except re.error as exc:
                raise ConfigError(f"{name} is not a valid regular expression: {exc}") from None
        self.model.validate()
        return self

    # ----------------------------------------------------------------------------------------
    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> Settings:
        """Build settings from parsed JSON. Unknown keys are an error, not silently ignored."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ConfigError(f"unknown setting(s): {', '.join(unknown)}")
        values: dict[str, Any] = {}
        for name, value in raw.items():
            if name == "model":
                values[name] = _model_from_mapping(value)
            elif isinstance(value, list):
                values[name] = tuple(value)
            else:
                values[name] = value
        try:
            return replace(cls(), **values).validate()
        except TypeError as exc:
            raise ConfigError(f"invalid settings: {exc}") from None

    @classmethod
    def from_file(cls, path: str | Path) -> Settings:
        target = Path(path)
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except OSError as exc:
            reason = exc.strerror or exc
            raise ConfigError(f"cannot read settings file {target}: {reason}") from None
        except json.JSONDecodeError as exc:
            raise ConfigError(f"settings file {target} is not valid JSON: {exc}") from None
        if not isinstance(raw, dict):
            raise ConfigError(f"settings file {target} must hold a JSON object")
        return cls.from_mapping(raw)


def _model_from_mapping(raw: Any) -> ModelSettings:
    if not isinstance(raw, dict):
        raise ConfigError("'model' must be a JSON object")
    known = {f.name for f in fields(ModelSettings)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"unknown model setting(s): {', '.join(unknown)}")
    return replace(ModelSettings(), **raw)
