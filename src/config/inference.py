from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for text generation / inference behavior."""

    max_new_tokens: int = 50
    temperature: float = 1.0
