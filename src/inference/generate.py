"""Text generation and inference utilities."""

import logging

import jax.numpy as jnp

from src.config import InferenceConfig, TokenizerConfig
from src.model.model import NanoLLM

logger = logging.getLogger(__name__)


def generate_text(
    model: NanoLLM,
    tokenizer_config: TokenizerConfig,
    inference_config: InferenceConfig,
    start_tokens: list[int],
) -> str:
    """
    Generate text using the model autoregressively.

    Args:
        model: Trained NanoLLM model
        tokenizer_config: Tokenizer and delimiter (from the loaded checkpoint)
        inference_config: Decoding behavior (max_new_tokens, temperature)
        start_tokens: Initial token IDs to condition generation

    Returns:
        Generated text string including the start tokens
    """

    tokenizer = tokenizer_config.tokenizer
    delimiter = tokenizer_config.delimiter
    end_token_id = tokenizer.encode(delimiter, allowed_special={delimiter})[0]
    max_new_tokens = inference_config.max_new_tokens
    temperature = inference_config.temperature

    # Make a defensive copy of the input to avoid mutating the caller's list
    tokens = list(start_tokens)

    logger.info(
        "Generating text: context=%d tokens, max_new_tokens=%d, temperature=%.2f",
        len(start_tokens), max_new_tokens, temperature,
    )

    for _ in range(max_new_tokens):
        context = tokens[-model.maxlen:]

        # RIGHT-pad to match training (not left-pad!)
        actual_len = len(context)
        if actual_len < model.maxlen:
            context = context + [0] * (model.maxlen - actual_len)

        context_array = jnp.array(context)[None, :]
        logits = model(context_array)

        next_token_logits = logits[0, actual_len - 1, :] / temperature

        # TODO: Replace greedy choice with sampling from the probability distribution
        #   Current state (greedy): Always selects the single most probable token.
        #   Deterministic, tends toward repetitive or degenerate output, and can loop.
        #   Makes the temperature parameter effectively meaningless.
        #
        #   With sampling: Temperature becomes meaningful (higher = more random, lower = more conservative),
        #   output is varied, and the model behaves closer to the expectations of a real LLM.
        #      key, subkey = jax.random.split(key)
        #      next_token = int(jax.random.categorical(subkey, next_token_logits))
        next_token = int(jnp.argmax(next_token_logits))

        if next_token == end_token_id:
            logger.info(
                "Generation stopped at end token after %d new tokens",
                len(tokens) - len(start_tokens),
            )
            break

        tokens.append(next_token)

    logger.info("Generation complete: %d tokens total", len(tokens))
    return tokenizer.decode(tokens)
