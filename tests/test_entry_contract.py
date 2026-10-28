"""The entry contract as data: the rule catalogue, the column kinds, and the declared types."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from delinquency_eda.contract import (
    ACCEPTED_FAMILIES,
    COLUMN_KINDS,
    EXTRACTED,
    RULE_IDS,
    RULES,
    TABLES,
    accepts,
    family_of,
    rule_for,
)

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "data-contract.md").read_text(encoding="utf-8")


def test_rule_ids_are_unique_and_every_rule_says_what_to_do():
    assert len(set(RULE_IDS)) == len(RULES)
    for rule in RULES:
        assert rule.scope in {"source", "rows", "dataset"}
        assert rule.text.endswith(".") and rule.fix.endswith(".")
        assert rule_for(rule.id) is rule
    with pytest.raises(KeyError):
        rule_for("no.such.rule")


def test_every_rule_is_documented_and_every_documented_rule_exists():
    documented = set(re.findall(r"^\| `([a-z]+\.[a-z_]+)` \|", DOC, flags=re.MULTILINE))
    assert documented == set(RULE_IDS)


def test_the_documented_text_of_every_rule_is_the_text_the_code_enforces():
    for rule in RULES:
        assert rule.text in DOC, rule.id


def test_every_required_column_has_a_kind_and_nothing_else_does():
    for table in TABLES:
        assert set(COLUMN_KINDS[table.name]) == set(table.required), table.name
    assert {kind for kinds in COLUMN_KINDS.values() for kind in kinds.values()} <= set(
        ACCEPTED_FAMILIES
    )


def test_extracted_columns_are_the_columns_the_dataset_holds():
    from delinquency_eda.dataset import BASE_COLUMNS, BREAK_COLUMNS, CONTACT_COLUMNS, DEBTOR_COLUMNS

    held = {
        "slips": BASE_COLUMNS,
        "contacts": CONTACT_COLUMNS,
        "breaks": BREAK_COLUMNS,
        "debtors": DEBTOR_COLUMNS,
    }
    for table in EXTRACTED:
        named = {
            *table.required,
            *table.dates,
            *table.numbers,
            *table.counts,
            *table.flags,
        }
        assert named <= set(held[table.name]), table.name


@pytest.mark.parametrize(
    ("declared", "family"),
    [
        ("int", "number"),
        ("integer", "number"),
        ("bigint", "number"),
        ("decimal(12,2)", "number"),
        ("numeric(18,0)", "number"),
        ("float", "number"),
        ("real", "number"),
        ("money", "number"),
        ("bit", "number"),
        ("varchar(40)", "text"),
        ("varchar(max)", "text"),
        ("nvarchar(20)", "text"),
        ("char(3)", "text"),
        ("nchar(3)", "text"),
        ("text", "text"),
        ("ntext", "text"),
        ("uniqueidentifier", "text"),
        ("date", "date"),
        ("datetime", "date"),
        ("datetime2", "date"),
        ("smalldatetime", "date"),
        ("datetimeoffset", "date"),
        ("varbinary(8)", None),
        ("xml", None),
        ("interval", None),
    ],
)
def test_a_declared_type_belongs_to_a_family(declared, family):
    assert family_of(declared) == family
    assert family_of(declared.upper()) == family


@pytest.mark.parametrize(
    ("kind", "declared", "fits"),
    [
        ("integer", "int", True),
        ("integer", "decimal(18,0)", True),
        ("integer", "varchar(20)", False),
        ("integer", "datetime", False),
        ("number", "decimal(12,2)", True),
        ("number", "varchar(20)", False),
        ("text", "varchar(20)", True),
        ("text", "int", False),
        ("id", "int", True),
        ("id", "varchar(20)", True),
        ("id", "datetime", False),
        ("date", "date", True),
        ("date", "datetime2", True),
        ("date", "text", True),  # SQLite stores ISO text; the values are read and checked anyway
        ("date", "integer", False),
        ("date", "decimal(8,0)", False),
        ("flag", "bit", True),
        ("flag", "integer", True),
        ("flag", "varchar(1)", False),  # 'N' would read as true
        ("flag", "char(1)", False),
    ],
)
def test_a_kind_accepts_only_its_families(kind, declared, fits):
    assert accepts(kind, declared) is fits


def test_an_engine_that_declares_no_type_is_not_judged():
    for kind in ACCEPTED_FAMILIES:
        assert accepts(kind, "") and accepts(kind, "  ")
