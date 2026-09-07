# nanoLLM

A transformer-based language model capable of text generation and completion. 
Built with JAX and Flax NNX, this program covers the entire pipelime from embedding to inference.
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
- **Checkpoint management**: Orbax-based model persistence for experiment reproducibility and training resumption. Each checkpoint is saved as a bundled directory with both tensor weights and a human-readable json sidecar of training metadata.

### Model Persistence & Deployment
- **Save and load checkpoints**: Export trained model weights and training metadata to disk as a bundle directory and restore them later for continued training, comparison, or inference.
- **Cross-device compatibility**: Load checkpoints trained on different devices (CPU/GPU/multi-device) for local inference or experimentation.
- **Integrity verification**: Automatic state comparison ensures loaded checkpoints match saved weights. CLI script enables diff analysis of any two checkpoint bundles with L2-norm ratios and per-parameter change statistics.

### Text Generation Inference Engine
- **Token-by-token generation**: Iteratively generates one token at a time, feeding each prediction back as input for the next step.
- **Temperature-based sampling**: User can set temperature to tune randomness in the probability distribution. e.g. Lower temperatures produces focused, deterministic outputs. Higher temperatures enable creative, diverse generation.
- **Configurable generation length**: User can set maximum token counts to control output length.


## Architecture

### Technology
This project uses [UV](https://docs.astral.sh/uv/) as the package manager.
All dependencies are managed via `pyproject.toml` and installed using UV.

| Major Packages       | Purpose                                                            |
|----------------------|--------------------------------------------------------------------|
| **python 3.11**      | runtime                                                            |
| **jax**              | High-performance numerical computing and automatic differentiation |
| **grain**            | Efficient data loading and preprocessing library for JAX           |
| **tiktoken**         | Converts input text into tokens (array of ints)                    |
| **orbax-checkpoint** | Model checkpoint saving, loading, and restoration                  |


### Project Structure
I'm using notebooks for development. As code stabilizes, I extract code to `src/` with test coverage.

```
nanoLLM/
│
├── pyproject.toml       # UV dependencies and package configuration
│ 
├── checkpoints/         # model weights and training metadata (gitignored)
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

### 2. Download training data
As of 2026, this project defaults to training on a small subset (1000 stories) of a commonly used Hugging Face dataset.
This allows users to work though the training steps relatively quickly with low resource consumption.
Outcomes will improve greatly if you use the optional `--data-file` flag on the subsequent CLI commands to train the model on an entire dataset.

Download the entire dataset to a git-ignored `data/` directory.
```
mkdir -p data/raw && wget -P data/raw/ https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt
```

Extract a subset of the first 1000 stories from the dataset to a separate file for expedient training.                                                                                                                                                                        
```
awk 'BEGIN{c=0} /<|endoftext|>/{c++} c<1000' data/raw/TinyStories-train.txt > data/raw/TinyStories-1000.txt 
```


## Usage

### Train the model
Training is a resource-intensive process that is best accomplished across multiple, well-documented phases. 
NanoLLM training processes are fully integrated with checkpoint persistence to support multi-phase training and experiment reproducibility.

Two CLI scripts are provided:
- `nanollm-train` for a fresh start: builds an untrained `NanoLLM(ModelConfig())`, trains for default or specified number of epochs, and persists a new checkpoint bundle.
- `nanollm-resume` for continued training: loads weights and configs from an existing checkpoint, trains for additional epochs, and persists a new checkpoint bundle.

Total Epochs:
- The default epoch count per each training run is very small to facilitate quick runs with minimal resources so users can get feedback quickly and inexpensively.
- However, hundreds or thousands of cumulative epochs are required to generate even extremely basic sentences if using the small 1000-story default training subset.
- The total epochs trained across all training sessions is recorded as `cumulative_epochs_completed` in the checkpoint metadata, regardless of which script is used.

#### Train from scratch
Use this script the first time you train the model or at the beginning of an experiment. 

```
# fresh training with default configuration
uv run nanollm-train

# fresh training with example overrides
uv run nanollm-train --epochs 5 --batch-size 64 --checkpoint-destination path/to/new
```

#### Resume training
Use this script for subsequent training sessions. If not specified, it uses a utility to discover the most recent checkpoint.

_NB: This project stores a `metadata.json` file within each checkpoint directory that summarizes the training session, including the cumulative epoch count._

```
# resume training with default configuration
uv run nanollm-resume

# resume training with example overrides
uv run nanollm-resume --epochs 5 --checkpoint-source {checkpoint_directory}/{prior_run_name}/ 
```

#### Optional Flags

| Flag                         | Used by      | Description                                                     | Default                            |
|------------------------------|--------------|-----------------------------------------------------------------|------------------------------------|
| `--epochs`                   | both scripts | Number of epochs to run **in this invocation** (not cumulative) | `3`                                |
| `--batch-size`               | both scripts | Number of samples per training batch                            | `32`                               |
| `--data-file`                | both scripts | Path to the training data file                                  | `data/raw/TinyStories-1000.txt`    |
| `--max-stories`              | both scripts | Maximum number of stories to load from the data file            | `100`                              |
| `--seed`                     | both scripts | Random seed for reproducibility                                 | `42`                               |
| `--shuffle` / `--no-shuffle` | both scripts | Enable or disable dataset shuffling                             | `False`                            |
| `--checkpoint-destination`   | both scripts | Path to save the new checkpoint bundle directory                | `checkpoints/NanoLLM_{timestamp}/` |
| `--checkpoint-source`        | resume only  | Path to the checkpoint bundle to load weights from              | latest bundle in `checkpoints/`    |


### Run inference
The inference flow loads a pre-trained model (via checkpoints) and runs inference to complete a phrase provided by the user. 

#### Inference via CLI

Enter the input phrase as part of the CLI command (`--prompt` required). If not specified, the command builds the model using the most recent checkpoint.
```
# generate text using defaults (most recent checkpoint)
uv run nanollm-generate --prompt "I was eating a bowl of raspberries when I noticed that"

# generate text using overrides
uv run nanollm-generate --prompt "I was eating a bowl of raspberries when I noticed that" --checkpoint-source checkpoints/bundle_X/ --max-new-tokens 100 --temperature 0.8 --seed 42
```

#### Flags

| Flag                   | Description                             | Default                           |
|------------------------|-----------------------------------------|-----------------------------------|
| `--prompt`             | Text prompt to complete (required)      | none                              |
| `--checkpoint-source`  | Path to checkpoint bundle to load       | latest bundle in `checkpoints/`   |
| `--max-new-tokens`     | Maximum tokens to generate              | `30`                              |
| `--temperature`        | Sampling temperature (> 0)              | `0.5`                             |
| `--seed`               | Random seed for reproducible sampling   | none (non-deterministic)          |


#### Inference via UI
See the last section of `07_run_inference.ipynb`.

## Development

### Test Suite

```
uv run pytest

# unit tests only for quick sanity check
uv run pytest tests/unit/
```

### Jupyter Notebooks
Existing notebooks document code development and previous experiments.

Open Jupyter in your default browser at `http://localhost:8888`:
```
uv run jupyter lab

# alternative interface
uv run jupyter notebook
```

Activate the notebook git filters to strip unnecessary information from notebooks before committing:
```
# strip outputs and execution counts
uv run nbstripout --install

# verify the filters are configured
git config --get filter.nbstripout.clean
```

### Checkpoint Diffs
A custom CLI tool enables detailed comparison of any two training checkpoints to understand parameter changes. 
The tool selects the two most recent checkpoints by default, but you can pass arguments `--before` and `--after` to compare any two checkpoints.

```
# compare the two most recent checkpoints automatically
uv run nanollm-compare

# compare specific bundles
uv run nanollm-compare --before checkpoints/bundle_A/ --after checkpoints/bundle_B/
```

#### Optional Flags

| Flag | Description | Default                |
|------|-------------|------------------------|
| `--before` | Path to the "before" checkpoint bundle | penultimate checkpoint |
| `--after` | Path to the "after" checkpoint bundle | most recent checkpoint |
| `--threshold` | Minimum absolute difference to count a parameter as changed | `1e-8`                 |

### Benign JAX warnings

#### intrinsics log warning

Training runs log a warning for `cpp_gen_intrinsics.cc`. This issue is relatively harmless and I chose to neither address nor suppress the warning.

The message comes from JAX's C++ layer, which writes straight to stderr and bypasses Python logging. 
No logging or warnings filter in this codebase intercepts this warning. 
Resolving the issue would require setting `JAX_PLATFORMS` or `TF_CPP_MIN_LOG_LEVEL` env variables before JAX is imported. 
That would couple the import order to a single line of output. For now, I've determined that this would be more problematic than the log noise.

If you dislike this noise in the logs, you can silence it in your own shell without changing the codebase:
```
export TF_CPP_MIN_LOG_LEVEL=2
```

#### os.fork warning from notebooks

Some notebook cells execute a CLI script through shell magic (e.g., `!uv run nanollm-resume`)
to demonstrate how the extracted functionality aligns with the original, experimental code in the notebooks.
This triggers the following warning: 
`os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock. pid, fd = os.forkpty()`

Regardless of the warning, the script should run to completion and save a valid checkpoint.
 
The deadlock that the warning describes happens when a forked child keeps running Python and touches JAX state.
But when applying shell magic to run the scripts from the notebooks, the forked child immediately execs `uv`, which replaces the process image.
So no JAX threads or locks survive into live Python code.

### Environment / PATH tips

Dependency issues can often be corrected by running the following:
```
# Delete the old venv
rm -rf .venv

# Recreate it with the correct paths
uv sync
```
