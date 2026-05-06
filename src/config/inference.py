from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for text generation / inference behavior.

    - max_new_tokens: caps how many tokens the model generates before stopping. Prevents runaway generation and controls output length.
    - temperature: tunes the randomness of the generated text by scaling the logits before the softmax.
        The value controls how "sharp" or "flat" the probability distribution is:
            1.0 (default): unchanged distribution — standard sampling
            < 1.0 (e.g. 0.5): more focused/conservative output (sharpens the distribution, which makes high-probability tokens more likely)
            > 1.0 (e.g. 1.5): more random/creative output (flattens the distribution, which gives lower-probability tokens more chance)
            → 0: approaches greedy decoding (always picks the most likely token)

    - seed: gets passed to a random number generator before sampling to control reproducibility of the token sampling process during inference.
        In this app, the seed is used when temperature-scaled logits are converted to a probability distribution and a token is drawn.
            None (default): sampling is non-deterministic; each run produces different output, which is the normal behavior for text generation.
            An integer value: sampling is deterministic; the same prompt with the same seed will always produce the same output. Useful for debugging, testing, or reproducible demos.
    """

    max_new_tokens: int = 50
    temperature: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {self.max_new_tokens}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0.0, got {self.temperature}")
