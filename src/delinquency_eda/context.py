"""What every stage receives: settings, where the data lives, where reports go, and a printer.

A stage never reads the process environment or ``sys.argv`` itself. The command line builds one
``Context`` and hands it over, which is what lets the tests drive every stage with a fake
connector, a temporary directory and a captured output stream.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from delinquency_eda.config import Settings
from delinquency_eda.dataset import Dataset
from delinquency_eda.db.base import Database
from delinquency_eda.db.sources import Source, open_source
from delinquency_eda.errors import ConfigError, DatasetError
from delinquency_eda.report import Doc

Connector = Callable[[Source, str], Database]


@dataclass
class Context:
    settings: Settings
    environ: Mapping[str, str] = field(default_factory=dict)
    backend: str = "sqlserver"
    sources: list[Source] = field(default_factory=list)
    out_dir: Path = Path("reports")
    dataset_path: Path = Path("data/dataset.json")
    out: TextIO = field(default_factory=lambda: sys.stdout)
    connector: Connector | None = None

    def say(self, text: str = "") -> None:
        print(text, file=self.out)

    @contextmanager
    def databases(self) -> Iterator[dict[str, Database]]:
        """Open every configured product line and close them all on the way out."""
        if not self.sources:
            raise ConfigError(
                "no sources configured: pass --source <line>=<target> or set DEA_SOURCES"
            )
        opened: dict[str, Database] = {}
        try:
            for source in self.sources:
                opened[source.product_line] = open_source(
                    source, self.backend, self.environ, connector=self.connector
                )
            yield opened
        finally:
            for db in opened.values():
                db.close()

    def load_dataset(self, *, require_rows: bool = True) -> Dataset:
        dataset = Dataset.load(self.dataset_path)
        if require_rows and not dataset.base:
            raise DatasetError(
                f"dataset {self.dataset_path} has no rows; rebuild it with a wider period"
            )
        return dataset

    def write_report(self, name: str, doc: Doc) -> Path:
        path = doc.save(self.out_dir / name)
        self.say(f"report written: {path}")
        return path
