import logging
from collections.abc import Iterable

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from src.config import TrainingConfig
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
        batches_per_epoch: int,
        training_config: TrainingConfig,
    ) -> None:
        """Configure and initialize the training session.

        Args:
            model: the model to train
            dataloader: iterable that yields batches for each epoch
            batches_per_epoch: count of batches per epoch; used to build the schedule
            training_config: training parameters
        """

        self.model = model
        self.dataloader = dataloader
        self.training_config = training_config

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
            Metrics history dict keyed by metric name (e.g. "train_loss"),
            with values recorded every log_every_n_steps steps
        """

        # Slide inputs over by one index so we are always comparing the inputs to the next token (the target)
        prep_target_batch = jax.vmap(
            lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0])))
        )

        metrics_history: dict[str, list[float]] = {"train_loss": []}
        current_epochs = self.training_config.epochs

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
                logger.info("Average Loss Overall: %.4f", (sum(epoch_losses) / len(epoch_losses)))

        logger.info(f"All {current_epochs} epochs completed.")
        return metrics_history
