```
██████╗ ██╗   ██╗███╗   ███╗ ██████╗ ██████╗  █████╗ ██╗
██╔══██╗╚██╗ ██╔╝████╗ ████║██╔═══██╗██╔══██╗██╔══██╗██║
██████╔╝ ╚████╔╝ ██╔████╔██║██║   ██║██║  ██║███████║██║
██╔═══╝   ╚██╔╝  ██║╚██╔╝██║██║   ██║██║  ██║██╔══██║██║
██║        ██║   ██║ ╚═╝ ██║╚██████╔╝██████╔╝██║  ██║███████╗
╚═╝        ╚═╝   ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
```

[![CI](https://github.com/ricardofrantz/pyModal/actions/workflows/ci.yml/badge.svg)](https://github.com/ricardofrantz/pyModal/actions/workflows/ci.yml)
![Python 3.14](https://img.shields.io/badge/Python-3.14-blue)
![macOS](https://img.shields.io/badge/macOS-26-lightgrey)
![Ubuntu](https://img.shields.io/badge/Ubuntu-24.04-orange)

# pyModal — modal decompositions in pure Python

A lightweight, zero-MPI toolkit for extracting coherent structures from spatiotemporal data. Every algorithm fits in a few hundred readable lines—study, tweak, or extend the maths without fighting a framework.

**Methods:** POD | DMD | SPOD | BSMD | ST-POD *(planned)*

---

## Installation

```bash
git clone https://github.com/ricardofrantz/pyModal.git
cd pyModal
pip install -e .
```

This installs `pymodal` as a package with CLI support.

**Optional performance backends:**

```bash
# Intel MKL (2-10x faster FFTs on Intel CPUs)
pip install mkl_fft

# Apple Silicon (Accelerate framework)
pip install pyobjc-framework-Accelerate

# GPU support
pip install cupy torch
```

---

## Quick Start

```python
from pymodal import PODAnalyzer, DMDAnalyzer, SPODAnalyzer

# Load your data
pod = PODAnalyzer(file_path="./data/simulation.mat", n_modes_save=10)
pod.run_analysis()

print(f"Modes shape: {pod.modes.shape}")
print(f"Energy in mode 1: {pod.eigenvalues[0]:.2e}")
```

**Run benchmark examples:**

```bash
python examples/benchmarks_2d.py
```

This generates synthetic flows (Double Gyre, Taylor-Green, Cylinder Wake) and runs all methods.

---

## Python API

### POD — Proper Orthogonal Decomposition

Energy-optimal spatial modes via SVD of mean-subtracted snapshots.

```python
from pymodal import PODAnalyzer

pod = PODAnalyzer(
    file_path="data.mat",
    n_modes_save=10,
    results_dir="./results_pod",
    figures_dir="./figs_pod",
)
pod.run_analysis()

# Access results
pod.modes              # (Nspace, n_modes) — spatial modes
pod.eigenvalues        # (n_modes,) — energy per mode
pod.time_coefficients  # (Ns, n_modes) — temporal evolution
pod.temporal_mean      # (Nspace,) — mean field
```

### DMD — Dynamic Mode Decomposition

Extracts eigenvalues/modes of the best-fit linear operator.

```python
from pymodal import DMDAnalyzer
import numpy as np

dmd = DMDAnalyzer(file_path="data.mat", n_modes_save=10)
dmd.load_and_preprocess()
dmd.perform_dmd()
dmd.save_results()

# Eigenvalues encode dynamics
frequencies = np.angle(dmd.eigenvalues) / (2 * np.pi * dmd.dt)  # Hz
growth_rates = np.log(np.abs(dmd.eigenvalues)) / dmd.dt         # 1/s

# |λ| < 1 → decaying, |λ| > 1 → growing, |λ| = 1 → neutral
```

### SPOD — Spectral POD

Frequency-resolved modes for stationary data. [Towne et al. (2018)](https://arxiv.org/abs/1708.04393)

```python
from pymodal import SPODAnalyzer

spod = SPODAnalyzer(
    file_path="data.mat",
    nfft=256,           # FFT block size
    overlap=0.5,        # 50% overlap
)
spod.run()
spod.perform_spod()

# Results indexed by frequency
spod.freq              # frequency bins (Hz)
spod.eigenvalues[f]    # energy at frequency index f
spod.modes[f]          # spatial modes at frequency f
```

### BSMD — Bispectral Mode Decomposition

Third-order interactions revealing nonlinear energy transfer. [Nekkanti et al. (2025)](https://arxiv.org/abs/2502.15091)

```python
from pymodal import BSMDAnalyzer

# BSMD reuses SPOD's cached FFT blocks
bsmd = BSMDAnalyzer(file_path="data.mat", nfft=256)
bsmd.run()
```

---

## Command Line Interface

```bash
# Run specific method
pymodal pod --data ./data/file.mat --n-modes 20
pymodal spod --data ./data/file.mat --nfft 256 --overlap 0.5
pymodal dmd --data ./data/file.mat

# Run all methods
pymodal all --data ./data/file.mat

# Staged execution
pymodal pod --data ./data/file.mat --compute   # compute only
pymodal pod --data ./data/file.mat --plot      # plot only
```

---

## Data Format

pyModal auto-detects `.mat`, `.h5`, and `.npz` files. Expected structure:

```python
{
    'q': np.ndarray,   # (Ns, Nspace) — snapshots × flattened spatial points
    'dt': float,       # time step between snapshots
    'Nx': int,         # grid points in x
    'Ny': int,         # grid points in y
    'x': np.ndarray,   # x-coordinates (optional)
    'y': np.ndarray,   # y-coordinates (optional)
}
```

**Custom data loader:**

```python
from pymodal import PODAnalyzer

def my_loader(file_path):
    # Load your data however you want
    return {
        'q': my_data,      # (Ns, Nspace)
        'dt': 0.01,
        'Nx': 100, 'Ny': 50,
        'x': x_coords, 'y': y_coords,
    }

pod = PODAnalyzer(file_path="ignored", data_loader=my_loader)
pod.run_analysis()
```

---

## Configuration

Set FFT backend via environment variable:

```bash
PYMODAL_FFT_BACKEND=mkl python script.py
```

Or programmatically:

```python
from pymodal.core.config import load_config

# Load from YAML/JSON
load_config("my_settings.yaml")
```

**Available settings** (`src/pymodal/core/config.py`):

| Setting | Default | Options |
|---------|---------|---------|
| `FFT_BACKEND` | `scipy` | `scipy`, `numpy`, `mkl`, `accelerate`, `cupy`, `torch` |
| `FIG_DPI` | `500` | Any integer |
| `WINDOW_TYPE` | `hamming` | `hann`, `hamming`, `blackman`, etc. |

---

## Methods Comparison

| Method | Input Assumption | Output | Best For |
|--------|------------------|--------|----------|
| **POD** | Any | Energy-ranked modes | Dimensionality reduction, dominant structures |
| **DMD** | Linear dynamics | Modes + frequencies + growth rates | Stability analysis, forecasting |
| **SPOD** | Stationary | Frequency-resolved modes | Turbulence, periodic flows |
| **BSMD** | Nonlinear coupling | Triadic interactions | Energy transfer, nonlinear dynamics |

---

## Project Structure

```
pyModal/
├── src/pymodal/              # Installable package
│   ├── __init__.py           # Exports: PODAnalyzer, DMDAnalyzer, ...
│   ├── pod.py                # Proper Orthogonal Decomposition
│   ├── dmd.py                # Dynamic Mode Decomposition
│   ├── spod.py               # Spectral POD
│   ├── bmsd.py               # Bispectral Mode Decomposition
│   ├── cli.py                # Command-line interface
│   ├── core/
│   │   ├── base.py           # BaseAnalyzer class
│   │   ├── config.py         # Global settings
│   │   ├── io.py             # Data loaders (.mat, .h5, .npz)
│   │   └── parallel.py       # Threading utilities
│   └── fft/
│       ├── fft_backends.py   # Backend selection (scipy, mkl, cupy, ...)
│       └── spectral_utils.py # Welch, periodogram, etc.
├── examples/
│   └── benchmarks_2d.py      # Double Gyre, Taylor-Green, Cylinder Wake
├── tests/                    # 24 unit tests
├── DOC.md                    # Mathematical details & dev guide
├── pyproject.toml            # Package metadata
└── README.md
```

---

## Testing

```bash
pytest                    # Run all tests
pytest -v                 # Verbose
pytest tests/test_pod.py  # Single file
```

---

## References

| Method | Paper |
|--------|-------|
| SPOD | Towne, Schmidt & Colonius (2018) — [arXiv:1708.04393](https://arxiv.org/abs/1708.04393) |
| BSMD | Nekkanti, Pickering, Schmidt & Colonius (2025) — [arXiv:2502.15091](https://arxiv.org/abs/2502.15091) |
| ST-POD | Yeung & Schmidt (2025) — [arXiv:2502.09746](https://arxiv.org/abs/2502.09746) *(planned)* |

---

## License

MIT — see [LICENSE](LICENSE)

## Disclaimer

This software is provided "as is", without warranty of any kind. To the extent
permitted by law, the authors and contributors are not liable for any damage, loss
or claim arising from its use or misuse. You are responsible for how you use it and
for following the laws and rules that apply to you. The full terms are in
[LICENSE](LICENSE).

This is research software. It has not been validated for engineering design,
certification or safety-critical use. Check its results independently before you
rely on them.
