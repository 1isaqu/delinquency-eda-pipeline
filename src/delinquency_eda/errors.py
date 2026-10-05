"""Exception types. Everything the pipeline raises on purpose derives from ``PipelineError``."""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for errors the command line reports as a plain message, not a traceback."""


class ConfigError(PipelineError):
    """Invalid settings, environment variables or command line values."""


class AdapterError(PipelineError):
    """A database adapter could not connect or run a statement."""


class AdapterUnavailable(AdapterError):
    """An optional driver is not installed."""


class ReadOnlyViolation(PipelineError, ValueError):
    """SQL handed to an adapter is not a plain read."""


class DatasetError(PipelineError):
    """A dataset file is missing, malformed or inconsistent."""


class ModelError(PipelineError):
    """The model stage cannot run on the data it was given."""
