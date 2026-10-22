"""Helpers to import the repo's script-style modules (they use flat, sys.path-relative imports)."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def load_module(relpath: str, name: str):
    """Import REPO/relpath under a unique module name, with its directory on sys.path."""
    path = REPO / relpath
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO
