import dataclasses
import logging
from collections.abc import Iterable
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from src.checkpoint import CheckpointMetadata, save_checkpoint
from src.config import TrainingConfig, TokenizerConfig
from src.model.model import NanoLLM
from src.training.schedule import build_learning_rate_schedule, compute_step_counts
from src.training.step import make_train_step

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(
        self,
        model: NanoLLM,
        dataloader: Iterable,
        batches_per_epoch: int,
        training_config: TrainingConfig,
        previous_epochs_completed: int,
        *,
        checkpoint_path: Path | None = None,                # falls back to default within method
        tokenizer_config: TokenizerConfig | None = None,    # falls back to default within method
    ) -> None:

        if batches_per_epoch <= 0:
            raise ValueError(
                f"batches_per_epoch must be > 0, got {batches_per_epoch}"
            )

        self.model = model
        self.training_config = training_config
        self.dataloader = dataloader
        self.checkpoint_path = checkpoint_path
        self.tokenizer_config = tokenizer_config
        self.previous_epochs_completed = previous_epochs_completed

        total_steps, warmup_steps = compute_step_counts(training_config, batches_per_epoch)
        self.schedule = build_learning_rate_schedule(training_config, total_steps, warmup_steps)

        self.optimizer = nnx.ModelAndOptimizer(
            model,
            optax.adamw(learning_rate=self.schedule, weight_decay=training_config.weight_decay),
        )
        self.metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
        self.train_step = make_train_step()

    def train(self) -> dict[str, list[float]]:
        """Orchestrates the full training loop.

        Returns metrics_history, such as the history of the loss across all training steps."""

        metrics_history: dict[str, list[float]] = {"train_loss": []}

        # Slide inputs over by one index so we are always comparing the inputs to the next token (the target)
        prep_target_batch = jax.vmap(
            lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0])))
        )

        # value of 'epoch' is only used for logging within the training loop so it is 1-indexed
        previous_epochs_completed, current_epochs = self.previous_epochs_completed, self.training_config.epochs
        for epoch in range(1, current_epochs + 1):
            logger.info(f"Epoch {epoch} commenced.")

            step = 0 # step is zero-indexed
            epoch_losses: list[float] = []
            for batch in self.dataloader:
                input_batch = jnp.array(jnp.array(batch).T).astype(jnp.int32)
                target_batch = prep_target_batch(
                    jnp.array(jnp.array(batch).T)
                ).astype(jnp.int32)

                self.train_step(self.model, self.optimizer, self.metrics, (input_batch, target_batch))

                if (step + 1) % self.training_config.log_every_n_steps == 0:
                    for metric, value in self.metrics.compute().items():
                        metrics_history.setdefault(f"train_{metric}", []).append(float(value))
                    self.metrics.reset()

                    current_learning_rate = self.schedule(step)
                    loss_val = metrics_history["train_loss"][-1]
                    epoch_losses.append(loss_val)
                    logger.info(
                        "Epoch %d/%d: Loss=%.4f, LearningRate=%.2e",
                        epoch, current_epochs, loss_val, float(current_learning_rate),
                    )

                step += 1

            logger.info(f"Epoch {epoch}/{current_epochs} completed.")
            if epoch_losses:
                logger.info(
                    "Average Loss Overall: %.4f",
                    sum(epoch_losses) / len(epoch_losses),
                )

        logger.info(
            f"All {current_epochs} epochs completed. This is in addition to {previous_epochs_completed} epochs accumulated during previous trainings."
        )

        if self.checkpoint_path:
            final_loss = metrics_history["train_loss"][-1] if metrics_history["train_loss"] else None
            metadata = CheckpointMetadata(
                cumulative_epochs_completed=(previous_epochs_completed + current_epochs),
                final_loss=final_loss,
                model_config=dataclasses.asdict(self.model.config),
                training_config=dataclasses.asdict(self.training_config),
                tokenizer_config=dataclasses.asdict(self.tokenizer_config) if self.tokenizer_config is not None else None,
            )
            save_checkpoint(self.model, self.checkpoint_path, metadata=metadata)
        else:
            logger.info("Checkpoint path undefined. No checkpoint persisted.")

        return metrics_history
