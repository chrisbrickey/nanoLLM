"""Shared fixtures for unit and integration tests"""

import dataclasses
import logging
import os
import shutil
import uuid

# Prevent HTTP call when gradio is imported in tests
# Without this intervention, as soon as gradio is imported it creates an HTTP client and calls out for analytics.
# We use gradio in the tests but neither unit nor integration test suites support external HTTP calls.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
from collections.abc import Callable, Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import (
    RECOMMENDED_NEW_TOKENS,
    RECOMMENDED_TEMPERATURE,
    ModelConfig,
    TokenizerConfig,
)
from src.model.model import NanoLLM
from src.paths import CHECKPOINTS_DIR
from src.training.checkpoint import save_checkpoint
from src.training.schema import CheckpointMetadata

TINY_MAXLEN = 4
TINY_VOCAB_SIZE = 50
TINY_EMBED_DIM = 12
TINY_NUM_HEADS = 3
TINY_FF_DIM = 16
TINY_NUM_BLOCKS = 1

RunCli = Callable[..., str]
MakeRunCli = Callable[[Callable[[], None], str], RunCli]
MakeTinyModelWithConfig = Callable[..., tuple[NanoLLM, ModelConfig]]
MakeTinyModel = Callable[..., NanoLLM]
CheckpointPathFactory = Callable[..., Path]
SaveTinyCheckpoint = Callable[..., Path]


def make_tiny_model_config(seed: int = 0) -> ModelConfig:
    """Build the tiny model config shared by every test module."""
    return ModelConfig(
        maxlen=TINY_MAXLEN,
        vocab_size=TINY_VOCAB_SIZE,
        embed_dim=TINY_EMBED_DIM,
        num_heads=TINY_NUM_HEADS,
        feed_forward_dim=TINY_FF_DIM,
        num_transformer_blocks=TINY_NUM_BLOCKS,
        model_seed=seed,
    )


# Serialized forms of the tiny configs for tests that assert on metadata.json payloads instead of building a model
SAMPLE_MODEL_CONFIG_DICT = dataclasses.asdict(make_tiny_model_config())
SAMPLE_TOKENIZER_CONFIG = TokenizerConfig()
SAMPLE_TOKENIZER_CONFIG_DICT = dataclasses.asdict(SAMPLE_TOKENIZER_CONFIG)

# Sample inference values shared by the unit and integration suites
SAMPLE_PROMPT = "sample-text"
SAMPLE_COMPLETION = f"{SAMPLE_PROMPT} and then more words"

# Temperatures placed relative to the recommended band, which is advisory rather than enforced.
# All three stay above zero so InferenceConfig accepts them.
BELOW_RANGE_TEMPERATURE = RECOMMENDED_TEMPERATURE.minimum / 2
ABOVE_RANGE_TEMPERATURE = RECOMMENDED_TEMPERATURE.maximum + 1.0
IN_RANGE_TEMPERATURE = (RECOMMENDED_TEMPERATURE.minimum + RECOMMENDED_TEMPERATURE.maximum) / 2

# Token counts relative to the recommended band. There is no below-band case to test because
# the band's minimum is the hard limit, so anything lower is rejected outright.
ABOVE_RANGE_NEW_TOKENS = int(RECOMMENDED_NEW_TOKENS.maximum) + 1
IN_RANGE_NEW_TOKENS = int(RECOMMENDED_NEW_TOKENS.maximum) // 2


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    return make_tiny_model_config()


@pytest.fixture
def make_tiny_model_with_config() -> MakeTinyModelWithConfig:
    """Build a tiny model plus the config it was built from.

    Callers that persist a checkpoint need the config to write into metadata.
    """
    def _make(seed: int = 0) -> tuple[NanoLLM, ModelConfig]:
        config = make_tiny_model_config(seed)
        return NanoLLM(config), config

    return _make


@pytest.fixture
def make_tiny_model(
    make_tiny_model_with_config: MakeTinyModelWithConfig,
) -> MakeTinyModel:
    def _make(seed: int = 0) -> NanoLLM:
        model, _ = make_tiny_model_with_config(seed)
        return model

    return _make


@pytest.fixture
def checkpoint_path_factory() -> Generator[CheckpointPathFactory, None, None]:
    """Return a factory for unique bundle paths, each removed at teardown.

    Paths are direct children of CHECKPOINTS_DIR because anything outside the project root is rejected.
    """
    created: list[Path] = []

    def _new(prefix: str = "test") -> Path:
        path = CHECKPOINTS_DIR / f"{prefix}_{uuid.uuid4().hex[:8]}"
        created.append(path)
        return path

    yield _new

    for path in created:
        if path.exists():
            shutil.rmtree(path)


@pytest.fixture
def checkpoint_path(checkpoint_path_factory: CheckpointPathFactory) -> Path:
    """A single unique checkpoint bundle path, removed at teardown."""
    return checkpoint_path_factory()


@pytest.fixture
def save_tiny_checkpoint(
    make_tiny_model_with_config: MakeTinyModelWithConfig,
) -> SaveTinyCheckpoint:
    """Return a factory that persists a tiny-model checkpoint bundle at a given path."""

    def _save(
        path: Path,
        *,
        seed: int = 0,
        cumulative_epochs_completed: int = 1,
        with_metadata: bool = True,
        model_config: ModelConfig | None = None,
    ) -> Path:
        """Save a bundle. Pass model_config when the weights must match a model
        built from something other than the tiny config."""
        if model_config is None:
            model, config = make_tiny_model_with_config(seed)
        else:
            config = model_config
            model = NanoLLM(config)

        metadata = None
        if with_metadata:
            metadata = CheckpointMetadata(
                cumulative_epochs_completed=cumulative_epochs_completed,
                model_config=dataclasses.asdict(config),
                tokenizer_config=SAMPLE_TOKENIZER_CONFIG_DICT,
            )
        save_checkpoint(model, path, metadata=metadata)
        return path

    return _save


def assert_error_exit(
    exc_info: pytest.ExceptionInfo[SystemExit], caplog: pytest.LogCaptureFixture
) -> None:
    """A handled failure exits 1 and explains itself in the log."""
    assert exc_info.value.code == 1
    assert any(r.levelno == logging.ERROR for r in caplog.records)


@pytest.fixture
def missing_checkpoint_path() -> Path:
    """A path under CHECKPOINTS_DIR that is guaranteed not to exist."""
    return CHECKPOINTS_DIR / f"missing_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def make_run_cli(capsys: pytest.CaptureFixture[str]) -> MakeRunCli:
    """Return a factory that wraps one CLI entry point in a callable.

    The callable patches sys.argv, invokes main(), and returns whatever was printed.
    """

    def _make(main: Callable[[], None], prog: str) -> RunCli:
        def _run(args: list[str] | None = None) -> str:
            with patch("sys.argv", [prog, *(args or [])]):
                main()
            return capsys.readouterr().out

        return _run

    return _make


@pytest.fixture(autouse=True)
def capture_logs(
    request: pytest.FixtureRequest, caplog: pytest.LogCaptureFixture
) -> Generator[None, None, None]:
    """Capture one logger's records for test classes that ask for it.

    A class opts in by setting CAPTURED_LOGGER (and optionally CAPTURED_LEVEL);
    everything else runs untouched.
    """
    logger_name = getattr(request.cls, "CAPTURED_LOGGER", None)
    if logger_name is None:
        yield
        return

    level = getattr(request.cls, "CAPTURED_LEVEL", logging.ERROR)
    with caplog.at_level(level, logger=logger_name):
        yield
