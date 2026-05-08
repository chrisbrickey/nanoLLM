# nanoLLM

A transformer-based language model capable of text generation and completion. 
Built with JAX and Flax NNX, this program covers the entire pipelime from embeddings to inference.
nanoLLM supports saving and loading training checkpoints, which enables model persistence, fine-tuning, and inference with pre-trained weights.


## Key Features & Capabilities
I developed the code iteratively, trying out implementations in Jupyter notebooks and extracting to modules with test coverage as code stabilized.

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
- **Checkpoint management**: Orbax-based model persistence for experiment reproducibility and training resumption. Each checkpoint is saved as a bundled directory with tensor data (`weights.orbax/`) and a human-readable json sidecar (`metadata.json`) of training parameters like epoch count, final loss, and model config.

### Model Persistence & Deployment
- **Save and load checkpoints**: Export trained model weights and training metadata to disk as a bundle directory (`weights.orbax/` + `metadata.json`), and restore them later for continued training, comparison, or inference.
- **Cross-device compatibility**: Load checkpoints trained on different devices (CPU/GPU/multi-device) for local inference or experimentation.
- **Integrity verification**: Automatic state comparison ensures loaded checkpoints match saved weights (using L2 norm checks).

### Text Generation Inference Engine
- **Token-by-token generation**: Iteratively generates one token at a time, feeding each prediction back as input for the next step.
- **Temperature-based sampling**: User can set temperature to tune randomness in the probability distribution. e.g. Lower temperatures produces focused, deterministic outputs. Higher temperatures enable creative, diverse generation.
- **Configurable generation length**: User can set maximum token counts to control output length.


## Architecture

### Technology
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
| **orbax-checkpoint** | Model checkpoint saving, loading, and restoration              |


### Project Structure
I'm using notebooks for development and am extracting code to `src/` (with test coverage) as code stabilizes.

```
nanoLLM/
│
├── pyproject.toml       # UV dependencies and package configuration
│ 
├── checkpoints/         # saved model weights (gitignored)
├── data/                # raw data and processed datasets (gitignored)
│ 
├── notebooks/           # interactive notebooks for building out components*
├── scripts/             # CLI entrypoints for training, inference, etc. 
│ 
├── src/   
│    ├── config/              
│    ├── data/           # processeses the training data and datasets
│    ├── inference/      # orchestrates text generation
│    ├── model/          # defines the transformer architecture
│    └── training/       # orchestrates training of the model                                                                                
│ 
└── tests/               # unit and integration testing on python modules and scripts
```
_*I added a git filter to clean the notebooks prior to committing (e.g., removes outputs and execution counts)._


## Setup

### 1. Install dependencies

```
uv sync
```

### 2. Train the model

```
# run with default configuration
uv run nanollm-train

# run with some overrides
uv run nanollm-train --epochs 5 --batch-size 64 --checkpoint {desired_checkpoint_directory}/{training_run_name}/
```

#### Optional Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--batch-size` | Number of samples per training batch | `32` |
| `--epochs` | Number of full passes through the training data | `3` |
| `--max-stories` | Maximum number of stories to load from the data file | `100` |
| `--seed` | Random seed for reproducibility | `42` |
| `--shuffle` / `--no-shuffle` | Enable or disable dataset shuffling | `False` |
| `--data-file` | Path to the training data file | `data/TinyStories-1000.txt` |
| `--checkpoint` | Path to save the checkpoint bundle directory | `checkpoints/NanoLLM_{timestamp}/` |



### 3. [Optional] Launch jupyter notebooks

```
uv run jupyter notebook

# alternative interface
uv run jupyter lab
```

This will open Jupyter in your default browser at `http://localhost:8888`.


## Troubleshooting

### Test Suite

```
uv run pytest
```

### Environment / PATH issues

```
# Delete the old venv
rm -rf .venv

# Recreate it with the correct paths
uv sync
```