from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training hyperparameters."""

    # Data loading
    batch_size: int = 32
    max_stories: int = 100
    shuffle: bool = False
    seed: int = 42
    num_workers: int = 0  # Single process for experimentation

    # Training loop
    epochs: int = 3 # how many times the Trainer's outer loop iterates

    # Learning rate schedule
    lr_init_value: float = 0.0
    lr_peak_value: float = 3e-4
    lr_end_value: float = 1e-5
    warmup_rate: float = 0.1  # 10% of total steps

    # Optimizer
    weight_decay: float = 0.01

    # Logging
    log_every_n_steps: int = 2
