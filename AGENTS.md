# AGENTS

This file provides context and guidelines for AI agents (e.g., Codex CLI) working within this repository. AI assistants should locate the nearest `AGENTS.md` in the directory hierarchy and follow its instructions when modifying or submitting code.

## Project Overview

`modal-decomp` is a pure‑Python collection of scripts for extracting coherent flow structures via modal decompositions:
- Spectral Proper Orthogonal Decomposition (SPOD)
- Bispectral Mode Decomposition (BSMD)
- Space‑Time Proper Orthogonal Decomposition (ST-POD)

See `README.md` for full details on features, installation, and usage.

## Repository Structure

```text
.                   # project root
├── bmsd.py          # Bispectral mode decomposition scripts
├── pod.py           # Proper Orthogonal Decomposition (POD)
├── spod.py          # Spectral POD implementation
├── spod_old.py      # Legacy SPOD code
├── utils.py         # Common utility functions
├── parallel_utils.py# Parallel helper routines
├── fft/             # Spectral analysis examples
├── data/            # Example datasets (MAT, HDF5)
├── preprocess/      # Data preprocessing pipelines
├── pyspod_upstream/ # Upstream PySPOD reference code
├── figs/            # Generated figures
├── pod_figs/        # POD-specific figures
├── pod_results/     # POD output data
├── DOCUMENTATION.md # High-level project docs
├── HARMONIZATION_PLAN.md
├── PARALLELIZATION_SUMMARY.md
├── refactoring_analysis.md
├── refactoring_results.md
└── README.md        # Project overview and quickstart
``` 

## Coding Guidelines

- Follow the existing code style (PEP8-compatible).
- Keep changes minimal and focused on the requested task.
- Do not modify code outside the scope of the current request.
- Preserve existing formatting and indentation.
- Update or add documentation when introducing new functionality.

## Testing & Validation

- Test changes by running the relevant scripts manually.
- Ensure generated figures or outputs match expected physical behavior.
- There is no automated test suite at this time.

## Commit & Review Workflow

- Use `apply_patch` to create minimal diffs for any file edits.
- Provide a concise summary of changes in the commit message.
- After applying patches, run any project-specific scripts to verify behavior.

> **Note:** If multiple `AGENTS.md` files exist in nested directories, AI agents should use the most deeply nested one relative to the working directory.