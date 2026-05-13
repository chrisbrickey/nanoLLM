"""Integration tests for a full training run (src/training/)."""

import dataclasses
import json
import shutil
import uuid
from collections.abc import Generator, Iterator
from pathlib import Path

import numpy as np
import pytest

from src.config import ModelConfig, TokenizerConfig, TrainingConfig
from src.model.model import NanoLLM
from src.paths import CHECKPOINTS_DIR
from src.training.trainer import Trainer

SAMPLE_DATA_SOURCE = Path("/fake/data/stories.txt")
SAMPLE_TOKENIZER_CONFIG = TokenizerConfig(
    delimiter="<|endoftext|>",
    name="gpt2",
    pad_token_id=0,
)

# Small enough to run fast; large enough that loss reliably trends down.
MAXLEN = 8
VOCAB_SIZE = 50
EMBED_DIM = 12
NUM_HEADS = 3
FF_DIM = 16
NUM_BLOCKS = 1
BATCH_SIZE = 4
N_BATCHES = 8
EPOCHS = 2


class _FakeDataLoader:
    """Mimics grain's batch format: yields arrays shaped (MAXLEN, BATCH_SIZE)."""

    def __init__(self, n_batches: int, maxlen: int, batch_size: int) -> None:
        self.n_batches = n_batches
        self.maxlen = maxlen
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[np.ndarray]:
        for _ in range(self.n_batches):
            yield np.ones((self.maxlen, self.batch_size), dtype=np.int32)


@pytest.fixture
def project_checkpoint_path() -> Generator[Path, None, None]:
    path = CHECKPOINTS_DIR / f"integration_test_{uuid.uuid4().hex[:8]}"
    yield path
    if path.exists():
        shutil.rmtree(path)


def _make_trainer(
    checkpoint_path: Path | None = None,
    *,
    tokenizer_config: TokenizerConfig | None = None,
    previous_epochs_completed: int,
) -> Trainer:
    model_config = ModelConfig(
        maxlen=MAXLEN,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        feed_forward_dim=FF_DIM,
        num_transformer_blocks=NUM_BLOCKS,
    )
    model = NanoLLM(model_config)
    config = TrainingConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        log_every_n_steps=4,
    )
    dataloader = _FakeDataLoader(
        n_batches=N_BATCHES, maxlen=MAXLEN, batch_size=BATCH_SIZE
    )
    return Trainer(
        model=model,
        training_config=config,
        dataloader=dataloader,
        batches_per_epoch=N_BATCHES,
        data_source=SAMPLE_DATA_SOURCE,
        previous_epochs_completed=previous_epochs_completed,
        checkpoint_path=checkpoint_path,
        tokenizer_config=tokenizer_config,
    )


class TestTrainLoop:
    def test_metrics_history_is_populated(self) -> None:
        history = _make_trainer(previous_epochs_completed=0).train()
        assert "train_loss" in history
        assert len(history["train_loss"]) > 0

    def test_all_losses_are_finite_and_positive(self) -> None:
        history = _make_trainer(previous_epochs_completed=0).train()
        for loss in history["train_loss"]:
            assert loss > 0.0
            assert loss < float("inf")

    def test_loss_decreases_over_training(self) -> None:
        history = _make_trainer(previous_epochs_completed=0).train()
        losses = history["train_loss"]
        assert losses[0] > losses[-1], (
            f"Expected loss to decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_checkpoint_written_to_disk(self, project_checkpoint_path: Path) -> None:
        _make_trainer(checkpoint_path=project_checkpoint_path, previous_epochs_completed=0).train()
        assert project_checkpoint_path.exists()
        assert (project_checkpoint_path / "weights.orbax").exists()
        assert (project_checkpoint_path / "metadata.json").exists()


class TestTrainerTokenizerConfigOnDisk:
    """End-to-end verification that the metadata.json on disk contains the
    tokenizer_config supplied to the Trainer (or null when omitted). The
    in-memory wiring is unit-tested in tests/unit/training/test_trainer.py."""

    def test_tokenizer_config_written_to_metadata_json(
        self, project_checkpoint_path: Path
    ) -> None:
        _make_trainer(
            checkpoint_path=project_checkpoint_path,
            tokenizer_config=SAMPLE_TOKENIZER_CONFIG,
            previous_epochs_completed=0,
        ).train()

        metadata_file = project_checkpoint_path / "metadata.json"
        assert metadata_file.exists()
        saved = json.loads(metadata_file.read_text(encoding="utf-8"))
        assert saved["tokenizer_config"] == dataclasses.asdict(SAMPLE_TOKENIZER_CONFIG)

    def test_omitting_tokenizer_config_writes_null(
        self, project_checkpoint_path: Path
    ) -> None:
        _make_trainer(checkpoint_path=project_checkpoint_path, previous_epochs_completed=0).train()

        metadata_file = project_checkpoint_path / "metadata.json"
        assert metadata_file.exists()
        saved = json.loads(metadata_file.read_text(encoding="utf-8"))
        assert saved.get("tokenizer_config") is None


class TestTrainerPriorEpochsOnDisk:
    """End-to-end verification that previous_epochs_completed is correctly added to
    training_config.epochs and written to metadata.json as cumulative_epochs_completed."""

    def test_prior_epochs_written_to_metadata_json(
        self, project_checkpoint_path: Path
    ) -> None:
        """When previous_epochs_completed=10 and training_config.epochs=EPOCHS,
        metadata.json on disk must record cumulative_epochs_completed=10+EPOCHS so
        that each checkpoint gives an unambiguous cumulative training history."""
        prior = 10
        _make_trainer(
            checkpoint_path=project_checkpoint_path,
            previous_epochs_completed=prior,
        ).train()

        metadata_file = project_checkpoint_path / "metadata.json"
        assert metadata_file.exists()
        saved = json.loads(metadata_file.read_text(encoding="utf-8"))
        assert saved["cumulative_epochs_completed"] == prior + EPOCHS
