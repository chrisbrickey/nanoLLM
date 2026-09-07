"""Shared fixtures for the training CLI integration tests."""

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

from src.paths import DATA_DIR

# Enough stories for at least one batch with batch_size=2
FAKE_STORIES = "\n".join(
    f"Once upon a time story number {i} ended here.<|endoftext|>" for i in range(6)
)


@pytest.fixture
def data_file() -> Generator[Path, None, None]:
    """Writes a throwaway data file for a CLI run.

    Creates the gitignored data directory so tests can run even on a fresh clone without training data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"cli_test_{uuid.uuid4().hex[:8]}.txt"
    path.write_text(FAKE_STORIES, encoding="utf-8")
    yield path
    path.unlink(missing_ok=True)
