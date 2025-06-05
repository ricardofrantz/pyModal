# Repository guidelines for codex agent

This file provides coding and contribution guidelines for the **pyModal** repository. Follow these directions whenever you modify or add files.

## Project overview

The repository contains Python scripts that implement several modal decomposition techniques:

- `pod.py`  – Proper Orthogonal Decomposition
- `spod.py` – Spectral Proper Orthogonal Decomposition
- `bmsd.py` – Bispectral Mode Decomposition
- `utils.py` and `configs.py` – shared utilities and configuration
- `data_interface.py` – extensible data loaders for multiple file formats
- `fft/` – backend-specific FFT helpers
- `tests/` – unit tests executed with `pytest`

Sample data sets live in `data/`. Intermediate caches and figures are written to `results_*` and `figs_*` directories, which are ignored by git.

## Development workflow

1. Edit Python files using **PEP 8** style with a maximum line length of **120** characters.
2. Prefer single quotes for strings.
3. Run `python indent.py` to format code with **ruff** and sort imports.
4. Add descriptive docstrings and follow `snake_case` for functions and `PascalCase` for classes.
5. Keep platform independence in mind—use `os.path.join` for paths.

## Testing

Run the unit tests before each commit:

```bash
pytest
```

All tests must pass. The project has minimal dependencies (`h5py`, `matplotlib`, `numpy`, `scipy`, and `tqdm`) and targets **Python 3.13**.

## Running the scripts

The analysis scripts support staged execution with the flags `--prep`, `--compute`, and `--plot`. OMP-based parallelism is controlled via `OMP_NUM_THREADS`. You can check the detected optimizations with:

```bash
python -m parallel_utils
```

## Commit messages

Write clear, concise commit messages that describe the motivation for the change. Small, focused commits are preferred.
