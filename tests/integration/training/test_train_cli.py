"""Integration tests for the CLI entry point (scripts/train.py)."""

import logging
import shutil
import uuid
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.train import main
from src.config import TrainingConfig
from src.paths import CHECKPOINTS_DIR, DATA_DIR

# Enough stories for at least one batch with batch_size=2
_FAKE_STORIES = "\n".join(
    f"Once upon a time story number {i} ended here.<|endoftext|>" for i in range(6)
)


@pytest.fixture
def data_file() -> Generator[Path, None, None]:
    path = DATA_DIR / f"cli_test_{uuid.uuid4().hex[:8]}.txt"
    path.write_text(_FAKE_STORIES, encoding="utf-8")
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def checkpoint_path() -> Generator[Path, None, None]:
    path = CHECKPOINTS_DIR / f"cli_test_{uuid.uuid4().hex[:8]}.orbax"
    yield path
    if path.exists():
        shutil.rmtree(path)


@pytest.fixture
def run_cli(data_file: Path):
    """Factory fixture: runs main() with runner.run patched out and returns the kwargs
    that the CLI assembled for it (training_config, checkpoint_destination, etc.)."""

    def _run(*extra_args: str) -> dict:
        base_argv = [
            "nanollm-train",
            "--data-file", str(data_file),
            "--epochs", "1",
            "--batch-size", "2",
        ]
        argv = base_argv + list(extra_args)

        with patch("scripts.train.run") as mock_run:
            mock_run.return_value = None
            with patch("sys.argv", argv):
                main()
            return mock_run.call_args.kwargs

    return _run


class TestCliHappyPath:
    def test_exits_cleanly_and_writes_checkpoint(
        self, data_file: Path, checkpoint_path: Path
    ) -> None:
        argv = [
            "nanollm-train",
            "--data-file", str(data_file),
            "--max-stories", "6",
            "--epochs", "1",
            "--batch-size", "2",
            "--checkpoint-destination", str(checkpoint_path),
        ]
        with patch("sys.argv", argv):
            main()
        assert checkpoint_path.exists()


class TestCliArguments:
    def test_batch_size(self, run_cli) -> None:
        kwargs = run_cli("--batch-size", "8")
        assert kwargs["training_config"].batch_size == 8

    def test_epochs(self, run_cli) -> None:
        kwargs = run_cli("--epochs", "7")
        assert kwargs["training_config"].epochs == 7

    def test_max_stories(self, run_cli) -> None:
        kwargs = run_cli("--max-stories", "5")
        assert kwargs["training_config"].max_stories == 5

    def test_seed(self, run_cli) -> None:
        kwargs = run_cli("--seed", "123")
        assert kwargs["training_config"].seed == 123

    def test_shuffle_flag(self, run_cli) -> None:
        kwargs = run_cli("--shuffle")
        assert kwargs["training_config"].shuffle is True

    def test_no_shuffle_flag(self, run_cli) -> None:
        kwargs = run_cli("--no-shuffle")
        assert kwargs["training_config"].shuffle is False

    def test_default_shuffle_matches_training_config(self, run_cli) -> None:
        kwargs = run_cli()
        assert kwargs["training_config"].shuffle == TrainingConfig().shuffle

    def test_checkpoint_path_passed_to_trainer(
        self, run_cli, checkpoint_path: Path
    ) -> None:
        kwargs = run_cli("--checkpoint-destination", str(checkpoint_path))
        assert kwargs["checkpoint_destination"] == checkpoint_path

    def test_default_checkpoint_path(self, run_cli) -> None:
        fixed_dt = datetime(2026, 1, 15, 10, 30, 45)
        with patch("src.checkpoint.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_dt
            kwargs = run_cli()
        expected = CHECKPOINTS_DIR / "NanoLLM_20260115_103045"
        assert kwargs["checkpoint_destination"] == expected


class TestCliErrorHandling:
    @pytest.fixture(autouse=True)
    def _capture_logs(self, caplog: pytest.LogCaptureFixture) -> Generator[None, None, None]:
        with caplog.at_level(logging.ERROR, logger="scripts.train"):
            yield

    def test_missing_data_file_exits_1_with_error_message(self, caplog) -> None:
        argv = ["nanollm-train", "--data-file", str(DATA_DIR / "nonexistent.txt")]
        with patch("sys.argv", argv):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_data_file_outside_project_exits_1_with_error_message(self, caplog) -> None:
        argv = ["nanollm-train", "--data-file", "/tmp/outside_project.txt"]
        with patch("sys.argv", argv):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 1
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_oserror_during_training_exits_1_with_error_message(
        self, caplog, data_file: Path
    ) -> None:
        checkpoint = CHECKPOINTS_DIR / "cli_test_oserror.orbax"
        argv = [
            "nanollm-train",
            "--data-file", str(data_file),
            "--epochs", "1",
            "--batch-size", "2",
            "--checkpoint-destination", str(checkpoint),
        ]
        with patch("scripts.train.run") as mock_run:
            mock_run.side_effect = OSError("disk full")
            with patch("sys.argv", argv):
                with pytest.raises(SystemExit) as exc_info:
                    main()
        assert exc_info.value.code == 1
        assert any(r.levelno == logging.ERROR for r in caplog.records)
