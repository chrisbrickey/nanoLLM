# miniLLM

## Technology
This project uses [UV](https://docs.astral.sh/uv/) as the package manager.
All dependencies are managed via `pyproject.toml` and installed using UV.

| Package        | Version  | Purpose                                                              |
|----------------|----------|----------------------------------------------------------------------|
| **python**     | >=3.11   | runtime                                                              |
| **jupyter**    | >=1.1.1  | Interactive notebook environment for development and experimentation |
| **jax**        | >=0.9.1  | High-performance numerical computing and automatic differentiation   |
| **numpy**      | >=2.4.3  | Numerical computing library for array operations                     |
| **matplotlib** | >=3.10.8 | Data visualization and plotting                                      |

## Setup

### Install Dependencies

```
uv sync
```

### Run Jupyter

```
# If virtual environment is activated:
jupyter notebook

# Or run directly with UV:
uv run jupyter notebook
```

This will open Jupyter in your default browser at `http://localhost:8888`.

### Run JupyterLab (Alternative Interface)

```
# If virtual environment is activated:
jupyter lab

# Or run directly with UV:
uv run jupyter lab
```

## Anticipated project structure
I'm using notebooks for development and will extract code to `src/` as it stabilizes.

- **notebooks/**: Interactive notebooks for building out components
- **src/**: Reusable Python modules extracted from notebooks
- **tests/**: Unit and integration tests for src/ code
- **data/**: Training data and processed datasets
- **checkpoints/**: Saved model weights
