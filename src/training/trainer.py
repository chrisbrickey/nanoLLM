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
        *,
        model: NanoLLM,
        dataloader: Iterable,
        data_source: Path,                                # used for the start-of-run header log
        batches_per_epoch: int,
        training_config: TrainingConfig,
        tokenizer_config: TokenizerConfig | None = None,  # if None, omitted from checkpoint metadata
        checkpoint_destination: Path | None,              # if None, no checkpoint is persisted!
        checkpoint_source: Path | None = None,            # only passed as parameter when resuming training
        previous_epochs_completed: int = 0,               # only passed as parameter when resuming training
    ) -> None:
        """Configure and initialize the training session.

        Args:
            model: The NanoLLM model to train
            dataloader: Iterable that yields batches for each epoch
            data_source: Path to the training data file; used in the run header log
            batches_per_epoch: Number of batches per epoch; used to build the LR schedule
            training_config: Training hyperparameters
            tokenizer_config: Tokenizer settings; if None, omitted from checkpoint metadata
            checkpoint_destination: Destination for the saved checkpoint; if None, no checkpoint is written
            checkpoint_source: Source checkpoint path; only set when resuming a prior run
            previous_epochs_completed: Epochs completed before this run; added to the
                cumulative epoch count saved in the checkpoint
        """

        self.model = model
        self.dataloader = dataloader
        self.data_source = data_source
        self.training_config = training_config
        self.tokenizer_config = tokenizer_config

        self.checkpoint_destination = checkpoint_destination
        if checkpoint_destination is None:
            logger.warning("No checkpoint_destination path provided so no checkpoint will be persisted on this run.")

        self.checkpoint_source = checkpoint_source
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

        Returns:
            Metrics history dict keyed by metric name (e.g. ``"train_loss"``),
            with values recorded every ``log_every_n_steps`` steps
        """

        # Log initialization data
        header_lines = self._build_training_header_lines(
            data_source=self.data_source,
            training_config=self.training_config,
            checkpoint_destination=self.checkpoint_destination,
            checkpoint_source=self.checkpoint_source,
            previous_epochs=self.previous_epochs_completed,
        )
        logger.info("\n\n%s\n\n", "\n".join(header_lines))

        # Slide inputs over by one index so we are always comparing the inputs to the next token (the target)
        prep_target_batch = jax.vmap(
            lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0])))
        )

        metrics_history: dict[str, list[float]] = {"train_loss": []}
        previous_epochs_completed, current_epochs = self.previous_epochs_completed, self.training_config.epochs

        # Value of 'epoch' is only used for logging within the training loop so it is 1-indexed
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
                logger.info("Average Loss Overall: %.4f",(sum(epoch_losses) / len(epoch_losses)))

        cumulative_epochs_completed = previous_epochs_completed + current_epochs
        logger.info(
            f"\n\tAll {current_epochs} epochs completed.\n"
            f"\tThis is in addition to {previous_epochs_completed} epochs accumulated during previous trainings.\n"
            f"\tCumulative epochs completed: {cumulative_epochs_completed}.\n"
        )

        # Persist checkpoint
        if self.checkpoint_destination:
            final_loss = metrics_history["train_loss"][-1] if metrics_history["train_loss"] else None
            metadata = CheckpointMetadata(
                cumulative_epochs_completed=cumulative_epochs_completed,
                final_loss=final_loss,
                model_config=dataclasses.asdict(self.model.config),
                training_config=dataclasses.asdict(self.training_config),
                tokenizer_config=dataclasses.asdict(self.tokenizer_config) if self.tokenizer_config is not None else None,
            )
            save_checkpoint(self.model, self.checkpoint_destination, metadata=metadata)
        else:
            logger.info("Checkpoint path undefined. No checkpoint persisted.")

        return metrics_history

    @staticmethod
    def _build_training_header_lines(
            *,
            data_source: Path | None,
            training_config: TrainingConfig,
            checkpoint_destination: Path | None,

            # only applicable when resuming training
            checkpoint_source: Path | None = None,
            previous_epochs: int = 0,
    ) -> list[str]:
        """Build a per-invocation summary header as a list of lines.
        Callers join and log the return value.

        Args:
            data_source: Path to the training data file
            training_config: Training hyperparameters for this run
            checkpoint_destination: Where the checkpoint will be written
            checkpoint_source: Source checkpoint path; if provided, resume-related
                lines are prepended to the header
            previous_epochs: Number of epochs completed before this run
        """

        lines = []
        if checkpoint_source is not None:
            lines.append(f"\tprevious epochs trained: {previous_epochs}")
            lines.append(f"\tcheckpoint source:       {checkpoint_source}")
            lines.append("")

        lines.extend(
            [
                f"\tepochs (this run):      {training_config.epochs}",
                f"\tdata source:            {data_source}",
                f"\tmax stories:            {training_config.max_stories}",
                f"\tbatch size:             {training_config.batch_size}",
                f"\tshuffle:                {training_config.shuffle}",
                f"\tseed:                   {training_config.seed}",
                "",
                f"\tcheckpoint destination: {checkpoint_destination}",
            ]
        )
        return lines
