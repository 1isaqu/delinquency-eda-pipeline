"""Command line: one subcommand per numbered stage.

Global options (``--backend``, ``--source``, ``--env-file``, ``--config``, ``--dataset``,
``--out-dir``) may be written before or after the subcommand. Connection settings never appear on
the command line: they come from ``DEA_DB_*`` environment variables (see ``.env.example``).

Exit codes: 0 success, 1 a stage ran and reported a failed check (``schema``; ``base`` when the
data breaks an entry rule), 2 usage or configuration error, including a dataset file that fails
its seal.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from delinquency_eda import __version__
from delinquency_eda.config import Settings
from delinquency_eda.context import Connector, Context
from delinquency_eda.db.sources import BACKENDS, parse_sources, sources_from_env
from delinquency_eda.env import load_env_file
from delinquency_eda.errors import PipelineError
from delinquency_eda.stages import STAGES, by_command

DEFAULT_DATASET = "data/dataset.json"
DEFAULT_OUT_DIR = "reports"


def _add_global_options(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Global options. Repeated on every subcommand with ``SUPPRESS`` so position never matters."""

    def default(value: Any) -> Any:
        return argparse.SUPPRESS if suppress else value

    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=default(None),
        help="database adapter (default: $DEA_BACKEND, else sqlserver)",
    )
    parser.add_argument(
        "--source",
        action="append",
        metavar="LINE=TARGET",
        default=default(None),
        help="a product line and its database (repeatable; default: $DEA_SOURCES)",
    )
    parser.add_argument(
        "--env-file", default=default(None), help="read DEA_* variables from a file"
    )
    parser.add_argument(
        "--config", default=default(None), help="JSON settings file (default: $DEA_CONFIG)"
    )
    parser.add_argument(
        "--dataset",
        default=default(DEFAULT_DATASET),
        help=f"dataset bundle written by 'base' (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--out-dir",
        default=default(DEFAULT_OUT_DIR),
        help=f"where reports are written (default: {DEFAULT_OUT_DIR})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="delinquency-eda",
        description="Numbered analysis pipeline for overdue-unpaid receivables.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    _add_global_options(parser, suppress=False)
    common = argparse.ArgumentParser(add_help=False)
    _add_global_options(common, suppress=True)
    sub = parser.add_subparsers(dest="command", metavar="<stage>")
    for stage in STAGES:
        child = sub.add_parser(
            stage.command,
            parents=[common],
            help=f"{stage.number:02d}  {stage.summary}",
            description=f"Stage {stage.number:02d}. {stage.summary}.",
        )
        stage.load().add_arguments(child)
    return parser


def _settings(args: argparse.Namespace, env: Mapping[str, str]) -> Settings:
    path = args.config or env.get("DEA_CONFIG", "").strip()
    return Settings.from_file(path) if path else Settings().validate()


def build_context(
    args: argparse.Namespace,
    env: dict[str, str],
    *,
    connector: Connector | None = None,
    out: TextIO | None = None,
) -> Context:
    if args.env_file:
        load_env_file(args.env_file, env)
    sources = parse_sources(args.source) if args.source else sources_from_env(env)
    return Context(
        settings=_settings(args, env),
        environ=env,
        backend=args.backend or env.get("DEA_BACKEND", "").strip() or "sqlserver",
        sources=sources,
        out_dir=Path(args.out_dir),
        dataset_path=Path(args.dataset),
        out=out or sys.stdout,
        connector=connector,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    connector: Connector | None = None,
    out: TextIO | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(out or sys.stdout)
        return 2
    env = dict(os.environ if environ is None else environ)
    try:
        ctx = build_context(args, env, connector=connector, out=out)
        return int(by_command(args.command).load().run(ctx, args) or 0)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
