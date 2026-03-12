# miniLLM

A simple LLM built from scratch using JAX, which predicts the next words in a phrase.

## Technology
This project uses [UV](https://docs.astral.sh/uv/) as the package manager.
All dependencies are managed via `pyproject.toml` and installed using UV.

| Package         | Purpose                                                              |
|-----------------|----------------------------------------------------------------------|
| **python 3.11** | runtime                                             | 
| **jupyter**     | Interactive notebook environment for development and experimentation |
| **jax**         | High-performance numerical computing and automatic differentiation   |
| **grain**       | Efficient data loading and preprocessing library for JAX             |
| **numpy**       | Numerical computing library for array operations                     |
| **matplotlib**  | Data visualization and plotting                                      |
| **tiktoken**    | Converts input text into tokens (array of ints)                      |

## Major Components
- **Token + Position Embedding** tokenizes the text and embeds the tokens; I added position information within this embedding layer - otherwise the transformer would not know the word order
- **Transformer Blocks** I started with 6 transformer blocks. We could see how performace varies as number of blocks varies. Each transformer block has 6 attention heads each of which focus on a different aspect of the input (e.g. chars, time).
- **Output Layers** The last output layer will (for each word in a dictionary) calculate the probability of a word being the next word in the phrase. This is how we will choose the next word.* I used a causal attention mask to prevent the model from seeing future tokens when predicting the next one.

_* The next word might not be the one with the highest probability, depending on the temperature setting._

## Setup

### Install Dependencies

```
uv sync
```

### Run Jupyter Notebooks

```
uv run jupyter notebook

# alternative interface
uv run jupyter lab
```

This will open Jupyter in your default browser at `http://localhost:8888`.

_NB: I added a git filter that cleans up notebooks (e.g., removes outputs and execution counts) prior to committing._

### Run Test Suite

```
uv run pytest
```

## Project structure
I'm using notebooks for development and will extract code to `src/` as it stabilizes.

- **notebooks/**: Interactive notebooks for building out components
- **src/**: Reusable Python modules extracted from notebooks
- **tests/**: Unit and integration tests for src/ code
- **data/**: Training data and processed datasets (gitignored)
- **checkpoints/**: Under development; Saved model weights
