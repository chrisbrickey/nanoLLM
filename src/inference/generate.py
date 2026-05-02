"""Text generation and inference utilities."""

import jax.numpy as jnp
import tiktoken

from src.model.model import NanoLLM


def generate_text(
    model: NanoLLM,
    tokenizer: tiktoken.Encoding,
    delimiter: str,
    start_tokens: list[int],
    max_new_tokens: int = 50,
    temperature: float = 1.0
) -> str:
    """
    Generate text using the model autoregressively.

    Args:
        model: Trained NanoLLM model
        tokenizer: Tokenizer instance for decoding
        delimiter: Delimiter string that marks end of text
        start_tokens: Initial token IDs to condition generation
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)

        NB: The delimiter serves different purposes during training vs inference
            During training: Marks boundaries between stories in your dataset
            During inference: Acts as a stop condition. If the model generates the end token,
                it is signaling "I think this pararaph is complete".

    Returns:
        Generated text string including the start tokens
    """
    tokens = list(start_tokens)
    end_token_id = tokenizer.encode(delimiter, allowed_special={delimiter})[0]

    for _ in range(max_new_tokens):
        context = tokens[-model.maxlen:]

        # RIGHT-pad to match training (not left-pad!)
        actual_len = len(context)
        if actual_len < model.maxlen:
            context = context + [0] * (model.maxlen - actual_len)

        context_array = jnp.array(context)[None, :]
        logits = model(context_array)

        next_token_logits = logits[0, actual_len - 1, :] / temperature

        # TODO: Replace below with sample from the probability distribution
        #   probs = jax.nn.softmax(next_token_logits)
        #   next_token = int(jax.random.categorical(jax.random.PRNGKey(0), next_token_logits))
        next_token = int(jnp.argmax(next_token_logits))

        if next_token == end_token_id:
            break

        tokens.append(next_token)

    return tokenizer.decode(tokens)
