# nanoLLM

A transformer-based language model capable of text generation and completion. 
Built with JAX and Flax NNX, this program covers the entire pipelime from embeddings to inference.
nanoLLM supports saving and loading training checkpoints, which enables model persistence, fine-tuning, and inference with pre-trained weights.
I developed the code iteratively, trying out implementations in Jupyter notebooks and extracting to modules with test coverage as code stabilized.

## Key Features & Capabilities

### Transformer Architecture
- **Dual embedding system** (token + position): Maps each of the vocabulary tokens and 128 sequence positions to vectors. Position is required to understand order. 
- **Multi-head self-attention** (starting with 6 heads per block): Allows the model to focus on different aspects of the input simultaneously.
- **Causal masking**: A triangular attention mask ensures that the model uses only previous context when predicting the next token.

### Data Pipeline
- **Tokenization**: Uses tiktoken tokenizer to convert raw text to integer sequences.
- **Custom dataset class**: Handles variable-length sequences with intelligent padding and truncation to fixed 128-token windows.
- **Efficient data loading**: Grain-based pipeline with configurable shuffling, batching, and multi-worker support for efficient CPU-GPU transfer.

### Training Infrastructure
- **AdamW optimizer** Uses decoupled weight decay for better generalization than standard Adam.
- **Learning rate scheduling**: Warmup configuration prevents training instability. Cosine decay enables fine-grained convergence.
- **Checkpoint management**: Orbax-based model persistence for experiment reproducibility and training resumption.

### Model Persistence & Deployment
- **Save and load checkpoints**: Export trained model weights to disk and restore them later for continued training, comparison, or inference.
- **Cross-device compatibility**: Load checkpoints trained on different devices (CPU/GPU/multi-device) for local inference or experimentation.
- **Integrity verification**: Automatic state comparison ensures loaded checkpoints match saved weights (using L2 norm checks).

### Text Generation Inference Engine
- **Token-by-token generation**: Iteratively generates one token at a time, feeding each prediction back as input for the next step.
- **Temperature-based sampling**: User can set temperature to tune randomness in the probability distribution. e.g. Lower temperatures produces focused, deterministic outputs. Higher temperatures enable creative, diverse generation.
- **Configurable generation length**: User can set maximum token counts to control output length.

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

### 3. Launch Jupyter Notebooks

```
uv run jupyter notebook

# alternative interface
uv run jupyter lab
```

This will open Jupyter in your default browser at `http://localhost:8888`.

### 4. Run Test Suite

```
uv run pytest
```

## Troubleshooting

### Environment / PATH issues

```
# Delete the old venv
rm -rf .venv

# Recreate it with the correct paths
uv sync
```