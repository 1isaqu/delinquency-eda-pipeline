"""Run the command line in-process and capture what it prints."""

from __future__ import annotations

import io
from collections.abc import Mapping

from delinquency_eda.cli import main


def run_cli(*argv: str, environ: Mapping[str, str] | None = None) -> tuple[int, str]:
    """``(exit code, stdout text)``. The environment is empty unless given."""
    out = io.StringIO()
    code = main(list(argv), environ=dict(environ or {}), out=out)
    return code, out.getvalue()


def sqlite_args(*pairs: tuple[str, object]) -> list[str]:
    """``--backend sqlite --source line=path ...`` for the given product lines."""
    args = ["--backend", "sqlite"]
    for line, path in pairs:
        args += ["--source", f"{line}={path}"]
    return args
