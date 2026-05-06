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

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.max_stories <= 0:
            raise ValueError(f"max_stories must be > 0, got {self.max_stories}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if not 0.0 <= self.warmup_rate <= 1.0:
            raise ValueError(f"warmup_rate must be in [0.0, 1.0], got {self.warmup_rate}")
        if self.lr_init_value < 0.0:
            raise ValueError(f"lr_init_value must be >= 0.0, got {self.lr_init_value}")
        if self.lr_init_value > self.lr_peak_value:
            raise ValueError(
                f"lr_init_value ({self.lr_init_value}) must be <= lr_peak_value ({self.lr_peak_value})"
            )
        if self.lr_end_value < 0.0:
            raise ValueError(f"lr_end_value must be >= 0.0, got {self.lr_end_value}")
        if self.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0.0, got {self.weight_decay}")
        if self.log_every_n_steps <= 0:
            raise ValueError(f"log_every_n_steps must be > 0, got {self.log_every_n_steps}")
