# pyModal — lightweight modal decompositions in Python

pyModal is a collection of zero-MPI, minimal-dependency scripts for uncovering coherent structures in complex data.  It currently implements Proper Orthogonal Decomposition (POD), Spectral POD (SPOD), and the recently introduced Bispectral Mode Decomposition (BSMD), with full Space-Time POD (ST-POD) support on the roadmap.
The focus is on clarity over boilerplate: every algorithm fits in a few hundred readable lines so you can study, tweak, or extend the maths without fighting a framework.

- Why another modal toolbox?
pyModal is the only open-source package to combine POD + SPOD and BSMD in a single, laptop-friendly codebase—something you won’t find in established libraries like PySPOD (SPOD only)  ￼, PyDMD (DMD variants), or modred (POD/BPOD/DMD).

- **Proper Orthogonal Decomposition (POD)**
  Performs a weighted singular value decomposition of mean-subtracted snapshots
  to recover energy-ranked spatial modes and their temporal coefficients.

- **Spectral Proper Orthogonal Decomposition (SPOD)**
  Solves the cross-spectral density eigenvalue problem to yield energy-ranked,
  harmonic spatial modes under the assumption of wide-sense stationarity.
  _Reference:_ [Towne, Schmidt & Colonius (2018)](https://arxiv.org/abs/1708.04393)

- **Bispectral Mode Decomposition (BSMD)**
  Extracts third-order phase-coupled spatial modes by diagonalizing an estimated
  bispectral density tensor, revealing the triadic interactions that drive
  nonlinear energy transfer.
  _Reference:_ [Nekkanti, Pickering, Schmidt & Colonius (2025)](https://arxiv.org/abs/2502.15091)
  If FFT blocks are already present in `results_spod/`, BSMD automatically reuses
  them (printing "Reusing cached FFT blocks...") and writes new output to
  `results_bsmd/`.

- **Space-Time Proper Orthogonal Decomposition (ST-POD)**
  Generalizes POD to a full space–time framework by solving an eigenproblem of
  the space-time correlation tensor, capturing arbitrary nonstationary and
  transient dynamics over finite windows.
  Support for ST-POD is planned but **not yet implemented**.
  _Reference:_ [Yeung & Schmidt (2025)](https://arxiv.org/abs/2502.09746)

---

## 🔑 Key Features

- **Pure-Python & portable** – runs out-of-the-box on CPython ≥3.9; no MPI or GPU prerequisites.
- **Unified CLI workflow** – --prep, --compute, --plot stages for each method; caches FFT blocks automatically.
- **Minimal install footprint** – only numpy, scipy, matplotlib, h5py, tqdm.
- **Flexible I/O** – HDF5, NetCDF, MATLAB .mat, raw NumPy arrays.
- **Built-in visualisation** - Quick mode shapes, spectra, and bispectral-energy maps straight from the CLI.

---

## 💾 Getting the Code

```bash
git clone https://github.com/ricardofrantz/pyModal.git
cd pyModal
```

## Installation

Install the required Python packages with:

```bash
pip install h5py matplotlib numpy scipy tqdm
```

These scripts were tested on **Python 3.13** running on **Ubuntu 24.04** and **macOS**.

### Performance notes

On Apple silicon (M‑series) Macs we recommend installing Python via
[Miniforge](https://github.com/conda-forge/miniforge) and using the packages
provided by conda-forge:

```bash
conda install numpy scipy matplotlib
```

For Intel workstations with the Intel compiler stack you can take advantage of
Intel's optimized libraries:

```bash
conda install intel-openmp
conda install numpy[mkl]
```

## Running Tests

Install the dependencies and run the unit tests with:

```bash
pytest
```

### Parallel Execution

FFT and matrix operations rely on NumPy's multithreaded BLAS libraries.
The number of threads is taken from the `OMP_NUM_THREADS` environment
variable when present; otherwise all available CPU cores are used.

Check which optimizations are active by running:

```bash
python -m parallel_utils
```

### Script Usage

The analysis scripts (`pod.py`, `spod.py`, and `bmsd.py`) can now be executed in
separate stages. Each accepts the flags `--prep`, `--compute`, and `--plot` to
run only the desired part of the workflow:

```bash
python pod.py --prep      # preprocess input data
python pod.py --compute   # perform the decomposition
python pod.py --plot      # generate figures from results
```

Running a script with no flags executes all steps in sequence. The same options
apply to `spod.py` and `bmsd.py`.

---

## 🧐 Developer Notes

This section summarizes how the repository is organized and the mathematics implemented. It is intended for contributors extending the code.

### Code layout

- `spod.py`, `bmsd.py` and `pod.py` implement Spectral POD, Bispectral Mode Decomposition and standard POD respectively.
- `utils.py` provides the `BaseAnalyzer` class with common routines for loading data, computing FFT blocks via `blocksfft`, and saving results.
- `configs.py` contains global settings such as output directories, plotting defaults and the FFT backend.
- The `fft/` folder houses backend-specific FFT helpers.

### Mathematical overview

**POD** performs a weighted singular value decomposition of the mean-subtracted snapshots. Depending on the dimensions it solves either the temporal or spatial covariance problem and projects the data to obtain modes and time coefficients.

**SPOD** solves an eigenvalue problem for the cross–spectral density matrix. FFT blocks of the signal are computed with Welch's method (`blocksfft`). For each frequency bin `f_i` the weighted matrix
\[M_i = X_i^H W X_i\]
is diagonalized to obtain spatial modes and their energies.

**BSMD** analyzes triadic interactions. For a triad `(p1,p2,p3)` with `f_{p1}+f_{p2}=f_{p3}` it forms matrices `A` and `B` from cached FFT blocks and solves
\[C = A^\dagger W B,\quad C a = \lambda a.\]
The resulting eigenvectors reconstruct two coupled sets of spatial modes.

### Caching

Intermediate FFT blocks (`qhat`) are stored in HDF5 files whose names are generated by `make_result_filename`. Both SPOD and BSMD check for an existing cache before recomputing, and BSMD can reuse SPOD caches. This greatly speeds up iterative analyses and ensures reproducibility.

### Threads and performance

FFT computation uses the backend specified in `configs.py` and automatically leverages the multithreaded BLAS library shipped with NumPy. The helper `get_num_threads()` reports the number of threads (taken from `OMP_NUM_THREADS` when set).

### Extending the code

- New decompositions can be developed by subclassing `BaseAnalyzer`.
- Place cached data and figures in the respective `results_*` and `figs_*` directories.
- Override any global option via a JSON or YAML file passed to `configs.load_config()`.
- For BSMD, adjust the `ALL_TRIADS` list or provide your own set of frequency triplets.

By documenting these design choices, new contributors should be able to navigate the project and implement future updates more easily.

