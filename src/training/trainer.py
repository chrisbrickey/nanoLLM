import logging
from collections.abc import Callable, Iterable

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from src.config import TrainingConfig
from src.loss import cross_entropy_loss
from src.model.model import NanoLLM
from src.training.schema import MetricsHistory

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

        # --- caller-supplied dependencies (retained for use by train()) ---
        self.model = model
        self.dataloader = dataloader
        self.training_config = training_config
        self.batches_per_epoch = batches_per_epoch

        # --- derived schedule (public for inspection by logs and notebooks) ---
        self.total_steps, self.warmup_steps = self._compute_step_counts(training_config, batches_per_epoch)
        self.schedule = self._build_learning_rate_schedule(training_config, self.total_steps, self.warmup_steps)

        # --- internal training apparatus (built from inputs, mutated during training) ---
        self.optimizer = nnx.ModelAndOptimizer(
            model,
            optax.adamw(learning_rate=self.schedule, weight_decay=training_config.weight_decay),
        )
        self.metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
        self.train_step = self._make_train_step()

    def train(self) -> MetricsHistory:
        """Orchestrates the full training loop.

        Returns:
            MetricsHistory with values recorded every log_every_n_steps steps.
        """

        # Slide inputs over by one index so we are always comparing the inputs to the next token (the target)
        prep_target_batch = jax.vmap(
            lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0])))
        )

        metrics_history = MetricsHistory()
        current_epochs = self.training_config.epochs

        logger.info(
            f"Training plan: {self.total_steps} total steps "
            f"({self.warmup_steps} warmup) across {current_epochs} epochs "
            f"of {self.batches_per_epoch} batches"
        )

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
                        metrics_history.record(metric, float(value))
                    self.metrics.reset()

                    current_learning_rate = self.schedule(step)
                    loss_val = metrics_history.train_loss[-1]
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

    @staticmethod
    def _compute_step_counts(
        training_config: TrainingConfig, batches_per_epoch: int
    ) -> tuple[int, int]:
        """Calculate total training steps and warmup steps from a TrainingConfig.

        Args:
            training_config: TrainingConfig providing epochs and warmup_rate.
            batches_per_epoch: Number of batches per epoch.

        Returns:
            Tuple of (total_steps, warmup_steps). warmup_steps is at least 1.

        Raises:
            ValueError: If batches_per_epoch <= 0. A zero-batch schedule degenerates
                silently in optax; reject it at the boundary with a hint about the
                likely cause (dataset smaller than batch_size with drop_remainder=True).
        """
        if batches_per_epoch <= 0:
            raise ValueError(
                f"batches_per_epoch must be > 0, got {batches_per_epoch}; "
                "check that dataset_size >= batch_size with drop_remainder=True"
            )
        total_steps = batches_per_epoch * training_config.epochs
        warmup_steps = max(1, int(total_steps * training_config.warmup_rate))
        return total_steps, warmup_steps

    @staticmethod
    def _build_learning_rate_schedule(
        training_config: TrainingConfig,
        total_steps: int,
        warmup_steps: int,
    ) -> optax.Schedule:
        """Warmup + cosine-decay schedule built from input parameters
        and default values that are defined in configs.

        The learning rate schedule determines how the learning rate
        will change during a training run.
        """
        return optax.warmup_cosine_decay_schedule(
            init_value=training_config.lr_init_value,
            peak_value=training_config.lr_peak_value,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=training_config.lr_end_value,
        )

    @staticmethod
    def _make_train_step() -> Callable[
        [nnx.Module, nnx.ModelAndOptimizer, nnx.MultiMetric, tuple[jnp.ndarray, jnp.ndarray]],
        None,
    ]:
        """JIT-compiled training step factory that returns a compiled function,
        specifically an @nnx.jit-compiled train_step function.

        Calling the returned function will execute a single gradient update
        for the batch passed in as a parameter to the factory.

        NB: Returning a factory keeps the @nnx.jit decoration explicit and makes it easy
        to swap in a non-jit version for debugging or a pmap version for multi-device.
        """

        @nnx.jit
        def train_step(
            model: nnx.Module,
            optimizer: nnx.ModelAndOptimizer,
            metrics: nnx.MultiMetric,
            batch: tuple[jnp.ndarray, jnp.ndarray],
        ) -> None:
            grad_fn = nnx.value_and_grad(cross_entropy_loss, has_aux=True)
            (loss, _logits), grads = grad_fn(model, batch)
            metrics.update(loss=loss)
            optimizer.update(grads)

        return train_step
