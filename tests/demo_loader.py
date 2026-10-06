"""Import ``examples/build_demo_db.py`` (a script, not a package) so tests can reuse it."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "examples" / "build_demo_db.py"
_SPEC = importlib.util.spec_from_file_location("build_demo_db", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
module = importlib.util.module_from_spec(_SPEC)
sys.modules["build_demo_db"] = module  # dataclasses look their module up here
_SPEC.loader.exec_module(module)

DDL = module.DDL
build_demo = module.build_demo
