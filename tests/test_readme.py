"""The README makes claims about the code. These tests keep them true."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import fields
from pathlib import Path

import pytest

from delinquency_eda.cli import build_parser
from delinquency_eda.config import ModelSettings, Settings
from delinquency_eda.stages import STAGES

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def fenced_blocks(text: str) -> list[str]:
    return re.findall(r"```[a-z]*\n(.*?)```", text, flags=re.DOTALL)


def table_rows(text: str, header_start: str) -> list[list[str]]:
    """Cells of the markdown table whose header row starts with ``header_start``."""
    lines = [line.strip() for line in text.splitlines()]
    start = next(i for i, line in enumerate(lines) if line.startswith(header_start))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
        rows.append([c.strip() for c in cells])
    return rows


def test_every_command_in_a_code_block_parses():
    commands = [
        line.strip()
        for block in fenced_blocks(README)
        for line in block.splitlines()
        if line.strip().startswith("delinquency-eda ")
    ]
    assert len(commands) >= 10
    for command in commands:
        argv = shlex.split(command)[1:]
        build_parser().parse_args(argv)  # raises SystemExit on an unknown stage or option


def test_the_quick_start_runs_every_stage_except_the_ones_noted():
    block = next(b for b in fenced_blocks(README) if "build_demo_db.py" in b)
    used = {
        shlex.split(line)[1]
        for line in block.splitlines()
        if line.strip().startswith("delinquency-eda ")
    }
    assert used == {s.command for s in STAGES}


def test_the_stage_table_matches_the_registry():
    rows = table_rows(README, "| # | Command")
    listed = [(int(r[0]), r[1].strip("`")) for r in rows]
    assert listed == [(s.number, s.command) for s in STAGES]


def _default_text(value: object) -> str:
    if isinstance(value, tuple):
        return ", ".join(str(v) for v in value) if value else "none"
    return str(value)


def test_every_setting_is_documented_with_its_real_default():
    rows = {
        r[0].strip("`"): r[1].strip("`").replace("\\|", "|")
        for r in table_rows(README, "| Setting")
    }
    defaults = Settings()
    for field in fields(Settings):
        if field.name == "model":
            continue
        assert field.name in rows, f"setting {field.name} is not in the README"
        assert rows[field.name] == _default_text(getattr(defaults, field.name)), field.name
    for field in fields(ModelSettings):
        key = f"model.{field.name}"
        assert key in rows, f"setting {key} is not in the README"
        assert rows[key] == _default_text(getattr(ModelSettings(), field.name)), key
    documented = set(rows)
    assert documented <= {f.name for f in fields(Settings)} | {
        f"model.{f.name}" for f in fields(ModelSettings)
    }


def test_every_environment_variable_in_the_example_file_is_documented():
    names = re.findall(r"^#?\s*(DEA_[A-Z_]+)=", (ROOT / ".env.example").read_text(), re.MULTILINE)
    assert len(names) >= 10
    for name in names:
        assert f"`{name}`" in README, name


def test_example_settings_file_is_valid_and_matches_the_defaults():
    loaded = Settings.from_file(ROOT / "examples" / "settings.example.json")
    defaults = Settings()
    for field in fields(Settings):
        if field.name in {"system_agents", "model"}:
            continue  # the example names the demo's pseudo-agent on purpose
        assert getattr(loaded, field.name) == getattr(defaults, field.name), field.name
    assert loaded.model == defaults.model
    assert loaded.system_agents == ("batch_loader",)


def test_example_settings_file_names_every_setting():
    raw = json.loads((ROOT / "examples" / "settings.example.json").read_text(encoding="utf-8"))
    assert set(raw) == {f.name for f in fields(Settings)}
    assert set(raw["model"]) == {f.name for f in fields(ModelSettings)}


def test_relative_links_point_at_files_that_exist():
    links = re.findall(r"\]\(([^)#]+?)\)", README)
    local = [target for target in links if not target.startswith(("http://", "https://"))]
    assert {
        "docs/architecture.md",
        "docs/data-contract.md",
        "docs/data-sources.md",
        "docs/stages.md",
    } <= set(local)
    for target in local:
        assert (ROOT / target).exists(), target


def test_files_named_in_backticks_exist_unless_the_pipeline_generates_them():
    generated = {"data/dataset.json", "reports/", "data/demo", ".env"}
    tokens = set(re.findall(r"`([A-Za-z0-9_./-]+)`", README))
    paths = {t for t in tokens if "/" in t or t.endswith((".py", ".json", ".md", ".example"))}
    paths -= generated
    paths = {p for p in paths if not p.startswith("--")}
    assert "examples/settings.example.json" in paths
    for path in sorted(paths):
        assert (ROOT / path).exists(), path


def test_the_layout_block_only_names_files_and_folders_that_exist():
    block = next(b for b in fenced_blocks(README) if b.lstrip().startswith("src/delinquency_eda/"))
    names = set(re.findall(r"[\w.-]+\.(?:py|json|md|yml)|[\w.-]+/(?=\s|$)", block))
    assert {"cli.py", "stages/", "db/", "build_demo_db.py", "ci.yml", "data-contract.md"} <= names
    existing = {p.name for p in ROOT.rglob("*") if ".git" not in p.parts and ".venv" not in p.parts}
    for name in names:
        assert name.rstrip("/") in existing, name


MARKDOWN_PAGES = ["README.md", *sorted(f"docs/{p.name}" for p in (ROOT / "docs").glob("*.md"))]


@pytest.mark.parametrize("page", MARKDOWN_PAGES)
def test_every_table_row_has_as_many_cells_as_its_header(page):
    """GitHub splits a cell at every unescaped pipe, inside backticks too."""
    header, fenced = None, False
    for number, line in enumerate((ROOT / page).read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        fenced ^= text.startswith("```")
        if fenced or not text.startswith("|"):
            header = None
            continue
        cells = len(re.split(r"(?<!\\)\|", text.strip("|")))
        header = header or cells
        assert cells == header, f"{page}:{number} has {cells} cells, its header {header}"


@pytest.mark.parametrize(
    "page", ["architecture", "data-contract", "data-sources", "stages", "decisions", "glossary"]
)
def test_every_doc_page_exists_and_has_a_title(page):
    text = (ROOT / "docs" / f"{page}.md").read_text(encoding="utf-8")
    assert text.startswith("# ")


def test_the_readme_has_the_required_sections():
    for heading in (
        "## Quick start",
        "## Connect your own data",
        "## Configuration",
        "## Tests",
        "## Project layout",
        "## Key design decisions",
        "## Limitations",
    ):
        assert heading in README, heading
    assert "rewritten on synthetic data" in README
    assert "contains no" in README and "real data" in README


def test_ci_runs_the_quick_start_stage_by_stage():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for stage in STAGES:
        assert f"delinquency-eda {stage.command}" in workflow, stage.command
    assert "python -m pytest" in workflow and "ruff check ." in workflow
    assert "DEA_CONFIG: examples/settings.example.json" in workflow


def test_the_contract_table_in_the_readme_matches_the_contract():
    from delinquency_eda.contract import TABLES

    rows = table_rows(README, "| Table | One row per")
    documented = {r[0].strip("`"): set(re.findall(r"`([a-z_]+)`", r[2])) for r in rows}
    assert set(documented) == {t.name for t in TABLES}
    for table in TABLES:
        assert documented[table.name] >= set(table.required), table.name
    assert "never a name or a national id" in README
