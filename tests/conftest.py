"""Fixtures shared by the extraction, stage and end-to-end tests."""

from __future__ import annotations

import pytest

from delinquency_eda.db import SQLiteDatabase

from demo_loader import build_demo
from world import make_second_line, make_world


@pytest.fixture
def world(tmp_path):
    """A small hand-built database where every extraction rule has one proving row."""
    return make_world(tmp_path / "auto.db")


@pytest.fixture
def two_lines(tmp_path):
    """The same world plus a second product line that reuses numeric ids."""
    databases = {
        "auto": SQLiteDatabase.open(make_world(tmp_path / "auto.db")),
        "consumer": SQLiteDatabase.open(make_second_line(tmp_path / "consumer.db")),
    }
    yield databases
    for db in databases.values():
        db.close()


@pytest.fixture(scope="session")
def demo_dir(tmp_path_factory):
    """The seeded demo databases, built once per test session."""
    directory = tmp_path_factory.mktemp("demo")
    build_demo(directory)
    return directory
