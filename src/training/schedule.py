import optax

from src.config import TrainingConfig


def compute_step_counts(
    training_config: TrainingConfig, batches_per_epoch: int
) -> tuple[int, int]:
    """Calculate total training steps and warmup steps from a TrainingConfig.

    Args:
        training_config: TrainingConfig providing epochs and warmup_rate.
        batches_per_epoch: Number of batches per epoch.

    Returns:
        Tuple of (total_steps, warmup_steps). warmup_steps is at least 1.
    """
    total_steps = batches_per_epoch * training_config.epochs
    warmup_steps = max(1, int(total_steps * training_config.warmup_rate))
    return total_steps, warmup_steps


def build_learning_rate_schedule(
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
