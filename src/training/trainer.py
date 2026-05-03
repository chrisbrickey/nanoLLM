from collections.abc import Iterable
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax

from src.checkpoint import save_checkpoint
from src.config import TrainingConfig
from src.model.model import NanoLLM
from src.training.schedule import build_learning_rate_schedule
from src.training.step import make_train_step

class Trainer:

    def __init__(
        self,
        model: NanoLLM,
        training_config: TrainingConfig,
        dataloader: Iterable,
        batches_per_epoch: int,
        *,
        checkpoint_path: Path | None = None,
    ) -> None:

        self.model = model
        self.training_config = training_config
        self.dataloader = dataloader
        self.checkpoint_path = checkpoint_path

        total_steps, warmup_steps = training_config.calculate_training_steps(batches_per_epoch)
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

        for epoch in range(self.training_config.num_epochs):
            step = 0
            for batch in self.dataloader:
                input_batch = jnp.array(jnp.array(batch).T).astype(jnp.int32)
                target_batch = prep_target_batch(
                    jnp.array(jnp.array(batch).T)
                ).astype(jnp.int32)

                self.train_step(self.model, self.optimizer, self.metrics, (input_batch, target_batch))

                if (step + 1) % self.training_config.log_every_n_steps == 0:
                    for metric, value in self.metrics.compute().items():
                        metrics_history[f"train_{metric}"].append(float(value))
                    self.metrics.reset()

                    current_learning_rate = self.schedule(step)
                    print(
                        f"\nEpoch: {epoch + 1}, Step {step + 1}, "
                        f"Loss: {metrics_history['train_loss'][-1]:.4f}, "
                        f"LR: {current_learning_rate:.2e}"
                    )

                step += 1

        if self.checkpoint_path is not None:
            print(f"\nSaving checkpoint to {self.checkpoint_path} ...")
            save_checkpoint(self.model, self.checkpoint_path)
            print("Checkpoint saved.")

        return metrics_history
