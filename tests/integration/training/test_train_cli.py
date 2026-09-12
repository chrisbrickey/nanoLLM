"""Integration tests for the CLI entry point (scripts/train.py)."""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.train import main
from src.config import TrainingConfig
from src.paths import CHECKPOINTS_DIR, DATA_DIR
from tests.conftest import MakeRunCli, RunCli, assert_error_exit

PROG = "nanollm-train"
LOGGER_NAME = "scripts.train"

RunCliForRunnerKwargs = Callable[..., dict]


@pytest.fixture
def run_cli(make_run_cli: MakeRunCli) -> RunCli:
    """Run the train CLI and return everything it printed."""
    return make_run_cli(main, PROG)


@pytest.fixture
def run_cli_for_runner_kwargs(
    data_file: Path, run_cli: RunCli
) -> RunCliForRunnerKwargs:
    """Run main() with Runner patched out and return the kwargs the CLI passed to
    Runner(...) (training_config, checkpoint_destination, etc.)."""

    def _run(*extra_args: str) -> dict:
        args = [
            "--data-file", str(data_file),
            "--epochs", "1",
            "--batch-size", "2",
            *extra_args,
        ]
        with patch("scripts.train.Runner") as mock_runner_cls:
            mock_runner_cls.return_value.run.return_value = None
            run_cli(args)
            return mock_runner_cls.call_args.kwargs

    return _run


class TestCliHappyPath:
    def test_exits_cleanly_and_writes_checkpoint(
        self, data_file: Path, checkpoint_path: Path, run_cli: RunCli
    ) -> None:
        run_cli([
            "--data-file", str(data_file),
            "--max-stories", "6",
            "--epochs", "1",
            "--batch-size", "2",
            "--checkpoint-destination", str(checkpoint_path),
        ])
        assert checkpoint_path.exists()


class TestCliArguments:
    def test_batch_size(self, run_cli_for_runner_kwargs: RunCliForRunnerKwargs) -> None:
        kwargs = run_cli_for_runner_kwargs("--batch-size", "8")
        assert kwargs["training_config"].batch_size == 8

    def test_epochs(self, run_cli_for_runner_kwargs: RunCliForRunnerKwargs) -> None:
        kwargs = run_cli_for_runner_kwargs("--epochs", "7")
        assert kwargs["training_config"].epochs == 7

    def test_max_stories(self, run_cli_for_runner_kwargs: RunCliForRunnerKwargs) -> None:
        kwargs = run_cli_for_runner_kwargs("--max-stories", "5")
        assert kwargs["training_config"].max_stories == 5

    def test_seed(self, run_cli_for_runner_kwargs: RunCliForRunnerKwargs) -> None:
        kwargs = run_cli_for_runner_kwargs("--seed", "123")
        assert kwargs["training_config"].seed == 123

    def test_shuffle_flag(self, run_cli_for_runner_kwargs: RunCliForRunnerKwargs) -> None:
        kwargs = run_cli_for_runner_kwargs("--shuffle")
        assert kwargs["training_config"].shuffle is True

    def test_no_shuffle_flag(self, run_cli_for_runner_kwargs: RunCliForRunnerKwargs) -> None:
        kwargs = run_cli_for_runner_kwargs("--no-shuffle")
        assert kwargs["training_config"].shuffle is False

    def test_default_shuffle_matches_training_config(
        self, run_cli_for_runner_kwargs: RunCliForRunnerKwargs
    ) -> None:
        kwargs = run_cli_for_runner_kwargs()
        assert kwargs["training_config"].shuffle == TrainingConfig().shuffle

    def test_checkpoint_path_passed_to_trainer(
        self, run_cli_for_runner_kwargs: RunCliForRunnerKwargs, checkpoint_path: Path
    ) -> None:
        kwargs = run_cli_for_runner_kwargs("--checkpoint-destination", str(checkpoint_path))
        assert kwargs["checkpoint_destination"] == checkpoint_path

    def test_default_checkpoint_path(
        self, run_cli_for_runner_kwargs: RunCliForRunnerKwargs
    ) -> None:
        fixed_dt = datetime(2026, 1, 15, 10, 30, 45)
        with patch("src.training.checkpoint.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_dt
            kwargs = run_cli_for_runner_kwargs()
        expected = CHECKPOINTS_DIR / "NanoLLM_20260115_103045"
        assert kwargs["checkpoint_destination"] == expected


class TestCliErrorHandling:
    CAPTURED_LOGGER = LOGGER_NAME

    @pytest.mark.parametrize(
        "data_file_arg",
        [str(DATA_DIR / "nonexistent.txt"), "/tmp/outside_project.txt"],
        ids=["missing_file", "outside_project"],
    )
    def test_unusable_data_file_exits_1_with_error_message(
        self,
        run_cli: RunCli,
        caplog: pytest.LogCaptureFixture,
        data_file_arg: str,
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            run_cli(["--data-file", data_file_arg])
        assert_error_exit(exc_info, caplog)

    def test_oserror_during_training_exits_1_with_error_message(
        self,
        run_cli: RunCli,
        caplog: pytest.LogCaptureFixture,
        data_file: Path,
    ) -> None:
        checkpoint = CHECKPOINTS_DIR / "cli_test_oserror.orbax"
        with patch("scripts.train.Runner") as mock_runner_cls:
            mock_runner_cls.return_value.run.side_effect = OSError("disk full")
            with pytest.raises(SystemExit) as exc_info:
                run_cli([
                    "--data-file", str(data_file),
                    "--epochs", "1",
                    "--batch-size", "2",
                    "--checkpoint-destination", str(checkpoint),
                ])
        assert_error_exit(exc_info, caplog)
