"""Unit tests for src/training/trainer.py — Trainer.train_step is stubbed
so each test exercises the orchestration loop only (no JIT compile, no real
gradient computation, no disk I/O). Real-training behavior and disk side
effects are covered by tests/integration/training/."""

import dataclasses
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import pytest

from src.config import TrainingConfig, TokenizerConfig
from src.model.model import NanoLLM
from src.training.trainer import Trainer

EPOCH_COUNT = 1
BATCH_SIZE = 2
N_BATCHES = 4  # yields 2 log entries with default log_every_n_steps=2
MAXLEN = 4  # matches conftest.TINY_MAXLEN
STUB_LOSS = 0.5

SAMPLE_TOKENIZER_CONFIG = TokenizerConfig(
    delimiter="<|endoftext|>",
    name="gpt2",
    pad_token_id=0,
)


class _FakeDataLoader:
    """Mimics grain's batch format: each batch is shape (MAXLEN, BATCH_SIZE)."""

    def __init__(self, n_batches: int, maxlen: int, batch_size: int) -> None:
        self.n_batches = n_batches
        self.maxlen = maxlen
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[np.ndarray]:
        for _ in range(self.n_batches):
            yield np.ones((self.maxlen, self.batch_size), dtype=np.int32)


def _make_config(**overrides: object) -> TrainingConfig:
    defaults = dict(epochs=EPOCH_COUNT, batch_size=BATCH_SIZE, log_every_n_steps=2)
    defaults.update(overrides)
    return TrainingConfig(**defaults)  # type: ignore[arg-type]


def _make_dataloader() -> _FakeDataLoader:
    return _FakeDataLoader(n_batches=N_BATCHES, maxlen=MAXLEN, batch_size=BATCH_SIZE)


def _stub_train_step(
    model: nnx.Module,
    optimizer: nnx.ModelAndOptimizer,
    metrics: nnx.MultiMetric,
    batch: tuple[jnp.ndarray, jnp.ndarray],
) -> None:
    """Lightweight replacement for the JIT-compiled train_step. Updates
    metrics with a fixed loss so the orchestration loop still has a value
    to log and accumulate, but skips the gradient computation."""
    metrics.update(loss=jnp.array(STUB_LOSS))


def _build_trainer(
    model: NanoLLM, *, training_config: TrainingConfig | None = None, **kwargs: object
) -> Trainer:
    trainer = Trainer(
        model=model,
        training_config=training_config or _make_config(),
        dataloader=_make_dataloader(),
        batches_per_epoch=N_BATCHES,
        **kwargs,  # type: ignore[arg-type]
    )
    trainer.train_step = _stub_train_step  # type: ignore[assignment]
    return trainer


class TestTrainerInit:
    def test_wires_optimizer_and_metrics_and_step(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        trainer = Trainer(
            model=make_tiny_model(),
            training_config=_make_config(),
            dataloader=_make_dataloader(),
            batches_per_epoch=N_BATCHES,
            previous_epochs_completed=0,
        )
        assert isinstance(trainer.optimizer, nnx.ModelAndOptimizer)
        assert isinstance(trainer.metrics, nnx.MultiMetric)
        assert callable(trainer.train_step)

    @pytest.mark.parametrize("batches_per_epoch", [0, -1])
    def test_rejects_non_positive_batches_per_epoch(
        self, make_tiny_model: Callable[..., NanoLLM], batches_per_epoch: int
    ) -> None:
        """Defense-in-depth check at the Trainer's public boundary."""
        with pytest.raises(ValueError, match="batches_per_epoch"):
            Trainer(
                model=make_tiny_model(),
                training_config=_make_config(),
                dataloader=_make_dataloader(),
                batches_per_epoch=batches_per_epoch,
                previous_epochs_completed=0,
            )


class TestTrainerTrain:
    def test_logs_progress_and_loss(
            self, make_tiny_model: Callable[..., NanoLLM], caplog: pytest.LogCaptureFixture
    ) -> None:
        config = _make_config()
        previous_epochs_completed = 0
        trainer = _build_trainer(make_tiny_model(), training_config=config, previous_epochs_completed=previous_epochs_completed)
        with caplog.at_level(logging.INFO, logger="src.training.trainer"):
            trainer.train()

        assert "Epoch 1 commenced" in caplog.text
        assert f"All {EPOCH_COUNT} epochs completed" in caplog.text
        assert f"in addition to {previous_epochs_completed} epochs accumulated during previous trainings" in caplog.text
        assert "Loss=" in caplog.text

    def test_returns_populated_metrics_history(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        trainer = _build_trainer(make_tiny_model(), previous_epochs_completed=0)
        history = trainer.train()

        assert "train_loss" in history
        assert len(history["train_loss"]) > 0

    def test_log_every_n_steps_controls_history_length(
        self, make_tiny_model: Callable[..., NanoLLM], caplog: pytest.LogCaptureFixture
    ) -> None:
        config = _make_config(log_every_n_steps=2)
        trainer = _build_trainer(make_tiny_model(), training_config=config, previous_epochs_completed=0)
        with caplog.at_level(logging.INFO, logger="src.training.trainer"):
            history = trainer.train()
        # 4 batches / log every 2 steps = 2 entries
        assert len(history["train_loss"]) == N_BATCHES // config.log_every_n_steps
        assert caplog.text.count("Loss=") == N_BATCHES // config.log_every_n_steps

    def test_empty_dataloader_returns_empty_history(
        self, make_tiny_model: Callable[..., NanoLLM], caplog: pytest.LogCaptureFixture
    ) -> None:
        trainer = Trainer(
            model=make_tiny_model(),
            training_config=_make_config(),
            dataloader=_FakeDataLoader(n_batches=0, maxlen=MAXLEN, batch_size=BATCH_SIZE),
            batches_per_epoch=10,  # enough for a valid schedule; dataloader yields nothing
            previous_epochs_completed=0,
        )
        trainer.train_step = _stub_train_step  # type: ignore[assignment]
        with caplog.at_level(logging.INFO, logger="src.training.trainer"):
            history = trainer.train()
        assert history == {"train_loss": []}
        assert "loss=" not in caplog.text

    def test_checkpoint_path_none_skips_save(
        self, make_tiny_model: Callable[..., NanoLLM]
    ) -> None:
        trainer = _build_trainer(make_tiny_model(), checkpoint_path=None, previous_epochs_completed=0)
        with patch("src.training.trainer.save_checkpoint") as mock_save:
            trainer.train()
            mock_save.assert_not_called()

    def test_checkpoint_path_set_invokes_save_with_path(
        self, make_tiny_model: Callable[..., NanoLLM], tmp_path: Path
    ) -> None:
        target = tmp_path / "ckpt_bundle"
        trainer = _build_trainer(make_tiny_model(), checkpoint_path=target, previous_epochs_completed=0)
        with patch("src.training.trainer.save_checkpoint") as mock_save:
            trainer.train()
            mock_save.assert_called_once()
            args, kwargs = mock_save.call_args
            assert args[1] == target


class TestTrainerTokenizerConfig:
    def test_metadata_includes_tokenizer_config_when_provided(
        self, make_tiny_model: Callable[..., NanoLLM], tmp_path: Path
    ) -> None:
        """When Trainer is given a tokenizer_config, the CheckpointMetadata
        passed to save_checkpoint contains the matching dict."""
        trainer = _build_trainer(
            make_tiny_model(),
            checkpoint_path=tmp_path / "bundle",
            tokenizer_config=SAMPLE_TOKENIZER_CONFIG,
            previous_epochs_completed=0,
        )
        with patch("src.training.trainer.save_checkpoint") as mock_save:
            trainer.train()
            metadata = mock_save.call_args.kwargs["metadata"]
            assert metadata.tokenizer_config == dataclasses.asdict(SAMPLE_TOKENIZER_CONFIG)

    def test_metadata_tokenizer_config_is_none_when_omitted(
        self, make_tiny_model: Callable[..., NanoLLM], tmp_path: Path
    ) -> None:
        """Omitting tokenizer_config is backward-compatible — train() must
        still call save_checkpoint, with metadata.tokenizer_config = None."""
        trainer = _build_trainer(
            make_tiny_model(), checkpoint_path=tmp_path / "bundle", previous_epochs_completed=0
        )
        with patch("src.training.trainer.save_checkpoint") as mock_save:
            trainer.train()
            metadata = mock_save.call_args.kwargs["metadata"]
            assert metadata.tokenizer_config is None


class TestTrainerTotalEpochsMetadata:
    def test_metadata_total_epochs_accumulates_prior(
        self, make_tiny_model: Callable[..., NanoLLM], tmp_path: Path
    ) -> None:
        """To accurately reflect cumulative training history, checkpoint
        metadata must record cumulative_epochs_completed as the sum of epochs
        from previous training (loaded checkpoint) and epochs from this
        current round of training (specified on training_config)."""

        previous_epochs_completed, current_epochs = 5, 3
        config = _make_config(epochs=current_epochs)
        trainer = _build_trainer(
            make_tiny_model(),
            training_config=config,
            checkpoint_path=tmp_path / "bundle",
            previous_epochs_completed=previous_epochs_completed,
        )

        with patch("src.training.trainer.save_checkpoint") as mock_save:
            trainer.train()
            metadata = mock_save.call_args.kwargs["metadata"]
            assert metadata.cumulative_epochs_completed == (previous_epochs_completed + current_epochs)

    def test_metadata_total_epochs_with_zero_prior(
        self, make_tiny_model: Callable[..., NanoLLM], tmp_path: Path
    ) -> None:

        previous_epochs_completed, current_epochs = 0, 3
        config = _make_config(epochs=current_epochs)
        trainer = _build_trainer(
            make_tiny_model(),
            training_config=config,
            checkpoint_path=tmp_path / "bundle",
            previous_epochs_completed=previous_epochs_completed,
        )
        with patch("src.training.trainer.save_checkpoint") as mock_save:
            trainer.train()
            metadata = mock_save.call_args.kwargs["metadata"]
            assert metadata.cumulative_epochs_completed == (previous_epochs_completed + current_epochs)
