"""
nanoLLM/src/config/inference.py

Configuration for text generation.

Hard limits live here and are enforced in __post_init__
to reject values that are mathematically or structurally impossible.

Recommended bands are declared here but enforced nowhere.
They describe where this model produces useful output, so each surface
decides for itself whether to warn, clamp to, or ignore them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ParamRange:
    """Accepted range for a tunable inference parameter."""

    minimum: float
    maximum: float

    def contains(self, value: float) -> bool:
        """True if value falls within the recommended range, inclusive."""
        return self.minimum <= value <= self.maximum


# Hard limits, enforced below in __post_init__.
MIN_NEW_TOKENS: int = 1 # generation requires at least one new token
MIN_TEMPERATURE_EXCLUSIVE: float = 0.0  # generation divides raw scores by temperature so zero must be rejected

# Recommended bands, advisory only. See the module docstring.
RECOMMENDED_TEMPERATURE = ParamRange(minimum=0.1, maximum=2.0)
RECOMMENDED_NEW_TOKENS = ParamRange(minimum=MIN_NEW_TOKENS, maximum=200)

@dataclass(frozen=True)
class InferenceConfig:
    """Configuration for text generation

    - max_new_tokens: Caps how many tokens the model generates before stopping.
                      Prevents runaway generation and controls output length.
                      Must be greater than 0.0. Otherwise no response will be generated.

    - temperature:
         Tunes the randomness of the generated text by scaling the raw scores (dividing by temperature) before converting to probabilities.
         The temperature controls how "sharp" or "flat" is the probability distribution. e.g.,
             Low temperature (e.g. 0.2):   scores get stretched apart → softmax picks the top choice much more confidently → output is more predictable/greedy
             High temperature (e.g. 1.5):  scores get squeezed together → probabilities flatten out → output is more random/creative.

         Temperature = 0:   forbidden because we divide the raw scores by this number
         Temperature = 1:   no change, use raw scores as-is.
         Temperature = 2:   conventional upper boundary, but there is no mathematical limit

    - seed: passed to a random number generator before sampling to control reproducibility of the token sampling process during inference.
    """

    max_new_tokens: int = 30
    temperature: float = 0.5
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens < MIN_NEW_TOKENS:
            raise ValueError(
                f"max_new_tokens set to {self.max_new_tokens}, but must be >= {MIN_NEW_TOKENS}"
            )
        if self.temperature <= MIN_TEMPERATURE_EXCLUSIVE:
            raise ValueError(
                f"temperature set to {self.temperature}, but must be > {MIN_TEMPERATURE_EXCLUSIVE}"
            )
