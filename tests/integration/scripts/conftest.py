"""Shared fixtures and helpers for the CLI integration tests"""

import dataclasses
import shutil
import uuid
from collections.abc import Callable, Generator, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from src.paths import CHECKPOINTS_DIR
from tests.conftest import RunCli, SaveTinyCheckpoint, assert_error_exit


@dataclasses.dataclass
class RecordedCall:
    """What a patched callable was asked to do and what it handed back."""

    kwargs: dict[str, object] = dataclasses.field(default_factory=dict)
    result: object = None


CaptureCall = Callable[..., AbstractContextManager[RecordedCall]]


@pytest.fixture
def isolated_checkpoints_dir() -> Generator[Path, None, None]:
    """Yield a throwaway directory nested inside CHECKPOINTS_DIR"""
    path = CHECKPOINTS_DIR / f"_test_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True)

    yield path

    if path.exists():
        shutil.rmtree(path)


@pytest.fixture
def checkpoint_bundle(
    isolated_checkpoints_dir: Path, save_tiny_checkpoint: SaveTinyCheckpoint
) -> Path:
    """A single tiny-model checkpoint bundle with metadata."""
    return save_tiny_checkpoint(isolated_checkpoints_dir / "bundle")


@pytest.fixture
def capture_call() -> CaptureCall:
    """Patch a target and record the keyword arguments it received.

    Pass `returning` to stub the return value, or `wrapping` to delegate to the
    real callable and keep what it produced.
    """

    @contextmanager
    def _capture(
        target: str,
        *,
        returning: object = None,
        wrapping: Callable[..., object] | None = None,
    ) -> Iterator[RecordedCall]:
        recorded = RecordedCall()

        def _fake(**kwargs: object) -> object:
            recorded.kwargs.update(kwargs)
            recorded.result = wrapping(**kwargs) if wrapping is not None else returning
            return recorded.result

        with patch(target, side_effect=_fake):
            yield recorded

    return _capture


class CheckpointResolutionErrorTests:
    """Failure paths every checkpoint-loading CLI must handle the same way.

    A subclass supplies its own `run_cli` fixture and sets CAPTURED_LOGGER to
    its script's logger so the error records are captured.
    """

    def test_no_checkpoints_anywhere_exits_1(
        self,
        run_cli: RunCli,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with patch("src.cli.get_latest_checkpoint", return_value=None):
            with pytest.raises(SystemExit) as exc_info:
                run_cli()
        assert_error_exit(exc_info, caplog)

    def test_nonexistent_checkpoint_source_exits_1(
        self,
        run_cli: RunCli,
        missing_checkpoint_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_cli(["--checkpoint-source", str(missing_checkpoint_path)])
        assert_error_exit(exc_info, caplog)
