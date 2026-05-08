"""Unit tests for src/training/trainer.py"""

import dataclasses
import json
import logging
import shutil
import uuid
from collections.abc import Generator, Iterator
from pathlib import Path
from unittest.mock import patch

import numpy as np
import flax.nnx as nnx
import pytest

from src.config import ModelConfig, TrainingConfig, TokenizerConfig
from src.model.model import NanoLLM
from src.paths import CHECKPOINTS_DIR
from src.training.trainer import Trainer

MAXLEN = 4
VOCAB_SIZE = 50
EMBED_DIM = 12
NUM_HEADS = 3
FF_DIM = 16
NUM_BLOCKS = 1
BATCH_SIZE = 2
N_BATCHES = 4  # yields 2 log entries with default log_every_n_steps=2


class _FakeDataLoader:
    """Mimics grain's batch format: each batch is shape (MAXLEN, BATCH_SIZE)."""

    def __init__(self, n_batches: int, maxlen: int, batch_size: int) -> None:
        self.n_batches = n_batches
        self.maxlen = maxlen
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[np.ndarray]:
        for _ in range(self.n_batches):
            yield np.ones((self.maxlen, self.batch_size), dtype=np.int32)


def _make_model_config() -> ModelConfig:
    return ModelConfig(
        maxlen=MAXLEN,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        feed_forward_dim=FF_DIM,
        num_transformer_blocks=NUM_BLOCKS,
    )


def _make_model() -> NanoLLM:
    return NanoLLM(_make_model_config())


def _make_config(**overrides: object) -> TrainingConfig:
    defaults = dict(
        epochs=1,
        batch_size=BATCH_SIZE,
        log_every_n_steps=2,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)  # type: ignore[arg-type]


def _make_dataloader() -> _FakeDataLoader:
    return _FakeDataLoader(n_batches=N_BATCHES, maxlen=MAXLEN, batch_size=BATCH_SIZE)


@pytest.fixture
def project_checkpoint_path() -> Generator[Path, None, None]:
    path = CHECKPOINTS_DIR / f"trainer_test_{uuid.uuid4().hex[:8]}.orbax"
    yield path
    if path.exists():
        shutil.rmtree(path)


class TestTrainerInit:
    def test_wires_optimizer_and_metrics_and_step(self) -> None:
        trainer = Trainer(
            model=_make_model(),
            training_config=_make_config(),
            dataloader=_make_dataloader(),
            batches_per_epoch=N_BATCHES,
        )
        assert isinstance(trainer.optimizer, nnx.ModelAndOptimizer)
        assert isinstance(trainer.metrics, nnx.MultiMetric)
        assert callable(trainer.train_step)

    @pytest.mark.parametrize("batches_per_epoch", [0, -1])
    def test_rejects_non_positive_batches_per_epoch(self, batches_per_epoch: int) -> None:
        """Defense-in-depth check at the Trainer's public boundary."""
        with pytest.raises(ValueError, match="batches_per_epoch"):
            Trainer(
                model=_make_model(),
                training_config=_make_config(),
                dataloader=_make_dataloader(),
                batches_per_epoch=batches_per_epoch,
            )


class TestTrainerTrain:
    def test_returns_populated_metrics_history(self, caplog: pytest.LogCaptureFixture) -> None:
        trainer = Trainer(
            model=_make_model(),
            training_config=_make_config(),
            dataloader=_make_dataloader(),
            batches_per_epoch=N_BATCHES,
        )
        with caplog.at_level(logging.INFO, logger="src.training.trainer"):
            history = trainer.train()
        assert "train_loss" in history
        assert len(history["train_loss"]) > 0
        assert "loss=" in caplog.text
        assert "Epoch 1/1 complete" in caplog.text

    def test_log_every_n_steps_controls_history_length(self, caplog: pytest.LogCaptureFixture) -> None:
        config = _make_config(log_every_n_steps=2)
        trainer = Trainer(
            model=_make_model(),
            training_config=config,
            dataloader=_make_dataloader(),  # N_BATCHES=4 batches
            batches_per_epoch=N_BATCHES,
        )
        with caplog.at_level(logging.INFO, logger="src.training.trainer"):
            history = trainer.train()
        # 4 batches / log every 2 steps = 2 entries
        assert len(history["train_loss"]) == N_BATCHES // config.log_every_n_steps
        assert caplog.text.count("loss=") == N_BATCHES // config.log_every_n_steps

    def test_empty_dataloader_returns_empty_history(self, caplog: pytest.LogCaptureFixture) -> None:
        trainer = Trainer(
            model=_make_model(),
            training_config=_make_config(),
            dataloader=_FakeDataLoader(n_batches=0, maxlen=MAXLEN, batch_size=BATCH_SIZE),
            batches_per_epoch=10,  # enough for a valid schedule; dataloader yields nothing
        )
        with caplog.at_level(logging.INFO, logger="src.training.trainer"):
            history = trainer.train()
        assert history == {"train_loss": []}
        assert "loss=" not in caplog.text

    def test_checkpoint_path_none_skips_save(self) -> None:
        trainer = Trainer(
            model=_make_model(),
            training_config=_make_config(),
            dataloader=_make_dataloader(),
            batches_per_epoch=N_BATCHES,
            checkpoint_path=None,
        )
        with patch("src.training.trainer.save_checkpoint") as mock_save:
            trainer.train()
            mock_save.assert_not_called()

    def test_checkpoint_path_writes_file(self, project_checkpoint_path: Path) -> None:
        trainer = Trainer(
            model=_make_model(),
            training_config=_make_config(),
            dataloader=_make_dataloader(),
            batches_per_epoch=N_BATCHES,
            checkpoint_path=project_checkpoint_path,
        )
        trainer.train()
        assert project_checkpoint_path.exists()


SAMPLE_TOKENIZER_CONFIG = TokenizerConfig(
    delimiter="<|endoftext|>",
    name="gpt2",
    pad_token_id=0,
)


class TestTrainerTokenizerConfig:
    def test_tokenizer_config_written_to_metadata_json(
        self, project_checkpoint_path: Path
    ) -> None:
        """When Trainer receives tokenizer_config and train() saves a checkpoint,
        the metadata.json on disk contains a tokenizer_config key matching the config."""
        trainer = Trainer(
            model=_make_model(),
            training_config=_make_config(),
            dataloader=_make_dataloader(),
            batches_per_epoch=N_BATCHES,
            checkpoint_path=project_checkpoint_path,
            tokenizer_config=SAMPLE_TOKENIZER_CONFIG,
        )
        trainer.train()

        metadata_file = project_checkpoint_path / "metadata.json"
        assert metadata_file.exists()
        saved = json.loads(metadata_file.read_text(encoding="utf-8"))
        assert saved["tokenizer_config"] == dataclasses.asdict(SAMPLE_TOKENIZER_CONFIG)

    def test_trainer_without_tokenizer_config_does_not_raise(
        self, project_checkpoint_path: Path
    ) -> None:
        """Omitting tokenizer_config is backward-compatible — train() must not raise,
        and the checkpoint must be written."""
        trainer = Trainer(
            model=_make_model(),
            training_config=_make_config(),
            dataloader=_make_dataloader(),
            batches_per_epoch=N_BATCHES,
            checkpoint_path=project_checkpoint_path,
        )
        trainer.train()  # must not raise

        assert project_checkpoint_path.exists()
        metadata_file = project_checkpoint_path / "metadata.json"
        assert metadata_file.exists()
        saved = json.loads(metadata_file.read_text(encoding="utf-8"))
        # tokenizer_config is either absent or null — either is acceptable
        assert saved.get("tokenizer_config") is None
