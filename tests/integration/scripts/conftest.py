"""Shared fixtures for the CLI integration tests"""

import shutil
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

from src.paths import CHECKPOINTS_DIR


@pytest.fixture
def isolated_checkpoints_dir() -> Generator[Path, None, None]:
    """Yield a throwaway directory nested inside CHECKPOINTS_DIR"""
    path = CHECKPOINTS_DIR / f"_test_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True)

    yield path

    if path.exists():
        shutil.rmtree(path)
