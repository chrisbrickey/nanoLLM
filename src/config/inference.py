from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for text generation

    - max_new_tokens: caps how many tokens the model generates before stopping. Prevents runaway generation and controls output length.
    - temperature: tunes the randomness of the generated text by scaling the logits before the softmax.
        The value controls how "sharp" or "flat" the probability distribution is. It must be greater
        than 0 because the logits are divided by it:
            1.0: unchanged distribution, standard sampling
            below 1.0 (e.g. 0.5, the default): more focused/conservative output (sharpens the distribution, which makes high-probability tokens more likely)
            above 1.0 (e.g. 1.5): more random/creative output (flattens the distribution, which gives lower-probability tokens more chance)
            very small (e.g. 0.01): nearly greedy decoding, almost always picking the most likely token

    - seed: passed to a random number generator before sampling to control reproducibility of the token sampling process during inference.
            None (default): sampling is non-deterministic; each run produces different output, which is the normal behavior for text generation.
            An integer value: sampling is deterministic; the same prompt with the same seed will always produce the same output. Useful for debugging, testing, or reproducible demos.
    """

    max_new_tokens: int = 30
    temperature: float = 0.5
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {self.max_new_tokens}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0.0, got {self.temperature}")
