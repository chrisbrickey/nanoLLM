# nanoLLM

A simple LLM built using JAX, which predicts the next words in a phrase.

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

### 1. Install Dependencies

```
uv sync
```

### 2. Install Package in Editable Mode

This permits import of modules from `src/` in notebooks and other modules like `from src.utils import some_function`.

```
uv pip install -e .
```

### 3. Run Jupyter Notebooks

```
uv run jupyter notebook

# alternative interface
uv run jupyter lab
```

This will open Jupyter in your default browser at `http://localhost:8888`.

### Run Test Suite

```
uv run pytest
```

## Project structure
I'm using notebooks for development and am extracting code to `src/` (with test coverage) as code stabilizes.

```
nanoLLM/
│
├── pyproject.toml       # UV dependencies and package configuration
│ 
├── checkpoints/         # saved model weights (gitignored)
├── data/                # training data and processed datasets (gitignored)
│ 
├── notebooks/           # interactive notebooks for building out components*
├── src/                 # reusable python modules extracted from notebooks
└── tests/               # unit and integration tests for src/ code
```
_*I added a git filter to clean the notebooks prior to committing (e.g., removes outputs and execution counts)._