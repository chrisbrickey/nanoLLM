"""Integration tests for scripts/generate.py CLI"""

import logging
import uuid
from collections.abc import Callable, Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.generate import main
from src.config import InferenceConfig
from src.paths import CHECKPOINTS_DIR
from tests.conftest import SaveTinyCheckpoint

PROG = "nanollm-generate"
LOGGER_NAME = "scripts.generate"

SAMPLE_PROMPT = "sample-text"
FIXED_COMPLETION = f"{SAMPLE_PROMPT} and more generated words"

SAMPLE_MAX_NEW_TOKENS = 7
SAMPLE_TEMPERATURE = 0.5
SAMPLE_SEED = 42

RunCli = Callable[[list[str]], str]


def _argv(*extra: str, prompt: str = SAMPLE_PROMPT) -> list[str]:
    """Build argv for a CLI run. Every invocation needs a prompt, so it is the default."""
    return [PROG, "--prompt", prompt, *extra]


def _e2e_argv(checkpoint_bundle: Path) -> list[str]:
    """Argv for an unmocked run: a small token budget and fixed seed keep it fast and repeatable."""
    return _argv(
        "--checkpoint-source", str(checkpoint_bundle),
        "--max-new-tokens", "3",
        "--seed", "0",
    )


def _assert_error_exit(
    exc_info: pytest.ExceptionInfo[SystemExit], caplog: pytest.LogCaptureFixture
) -> None:
    """A handled failure exits 1 and explains itself in the log."""
    assert exc_info.value.code == 1
    assert any(r.levelno == logging.ERROR for r in caplog.records)


@pytest.fixture
def checkpoint_bundle(
    isolated_checkpoints_dir: Path, save_tiny_checkpoint: SaveTinyCheckpoint
) -> Path:
    """A single tiny-model checkpoint bundle with metadata."""
    return save_tiny_checkpoint(isolated_checkpoints_dir / "bundle")


@pytest.fixture
def run_cli(capsys: pytest.CaptureFixture[str]) -> RunCli:
    """Run the CLI with the given argv and return everything it printed."""

    def _run(argv: list[str]) -> str:
        with patch("sys.argv", argv):
            main()
        return capsys.readouterr().out

    return _run


class TestCliHappyPath:
    def test_explicit_checkpoint_source_prints_completion(
        self,
        checkpoint_bundle: Path,
        run_cli: RunCli,
    ) -> None:
        with patch("scripts.generate.complete_prompt", return_value=FIXED_COMPLETION):
            out = run_cli(_argv("--checkpoint-source", str(checkpoint_bundle)))

        assert FIXED_COMPLETION in out

    def test_falls_back_to_latest_checkpoint_when_source_omitted(
        self,
        checkpoint_bundle: Path,
        run_cli: RunCli,
    ) -> None:
        with patch("scripts.generate.complete_prompt", return_value=FIXED_COMPLETION), \
             patch("src.inference.cli.get_latest_checkpoint", return_value=checkpoint_bundle):
            out = run_cli(_argv())

        assert FIXED_COMPLETION in out


class TestCliEndToEnd:
    """Run the full pipeline (checkpoint restore, encode, generate, decode) with no mocks."""

    def test_generates_and_prints_completion_containing_prompt(
        self,
        checkpoint_bundle: Path,
        run_cli: RunCli,
    ) -> None:
        out = run_cli(_e2e_argv(checkpoint_bundle))
        assert "GENERATED TEXT" in out
        assert SAMPLE_PROMPT in out

    def test_same_seed_produces_identical_output(
        self,
        checkpoint_bundle: Path,
        run_cli: RunCli,
    ) -> None:
        argv = _e2e_argv(checkpoint_bundle)
        first = run_cli(argv)
        second = run_cli(argv)
        assert first == second


class TestCliErrors:
    @pytest.fixture(autouse=True)
    def _capture_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> Generator[None, None, None]:
        with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
            yield

    def test_no_checkpoints_anywhere_exits_1(
        self,
        run_cli: RunCli,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with patch("src.inference.cli.get_latest_checkpoint", return_value=None):
            with pytest.raises(SystemExit) as exc_info:
                run_cli(_argv())
        _assert_error_exit(exc_info, caplog)

    def test_nonexistent_checkpoint_source_exits_1(
        self,
        run_cli: RunCli,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        missing_path = CHECKPOINTS_DIR / f"missing_{uuid.uuid4().hex[:8]}"
        with pytest.raises(SystemExit) as exc_info:
            run_cli(_argv("--checkpoint-source", str(missing_path)))
        _assert_error_exit(exc_info, caplog)

    @pytest.mark.parametrize("prompt", ["", "   "])
    def test_empty_prompt_exits_2_without_loading_a_checkpoint(
        self,
        run_cli: RunCli,
        prompt: str,
    ) -> None:
        """An unusable prompt is a usage error, so argparse rejects it before the model loads."""
        with patch("scripts.generate.restore_from_checkpoint") as mock_restore:
            with pytest.raises(SystemExit) as exc_info:
                run_cli(_argv(prompt=prompt))
        assert exc_info.value.code == 2
        mock_restore.assert_not_called()

    @pytest.mark.parametrize(
        "flag,value",
        [
            ("--max-new-tokens", "0"),
            ("--temperature", "0"),
        ],
    )
    def test_invalid_flag_values_exit_1(
        self,
        run_cli: RunCli,
        caplog: pytest.LogCaptureFixture,
        flag: str,
        value: str,
    ) -> None:
        """Config validation fails before any checkpoint is read, so none is needed here."""
        with pytest.raises(SystemExit) as exc_info:
            run_cli(_argv(flag, value))
        _assert_error_exit(exc_info, caplog)


class TestCliFlags:
    def test_flags_and_prompt_propagate(
        self,
        checkpoint_bundle: Path,
        run_cli: RunCli,
    ) -> None:
        captured_kwargs: dict[str, object] = {}

        def _fake_complete_prompt(**kwargs: object) -> str:
            captured_kwargs.update(kwargs)
            return FIXED_COMPLETION

        with patch("scripts.generate.complete_prompt", side_effect=_fake_complete_prompt):
            out = run_cli(
                _argv(
                    "--checkpoint-source", str(checkpoint_bundle),
                    "--max-new-tokens", str(SAMPLE_MAX_NEW_TOKENS),
                    "--temperature", str(SAMPLE_TEMPERATURE),
                    "--seed", str(SAMPLE_SEED),
                )
            )

        assert captured_kwargs["prompt"] == SAMPLE_PROMPT
        config = captured_kwargs["inference_config"]
        assert isinstance(config, InferenceConfig)
        assert config.max_new_tokens == SAMPLE_MAX_NEW_TOKENS
        assert config.temperature == SAMPLE_TEMPERATURE
        assert config.seed == SAMPLE_SEED

        assert FIXED_COMPLETION in out
