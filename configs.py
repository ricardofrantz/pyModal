"""
Central configuration for modal decomposition analysis.

NOTE: ALL imports are available here and this is imported in utils.py
so we only need to import utils in other files.
"""

import json
import os


os.environ["OS_ACTIVITY_MODE"] = "disable"  # suppress macOS IMKClient logs
"""
Configuration and shared imports for modal decomposition tools.
"""

# Default directories
RESULTS_DIR = "./preprocess"  # Legacy generic results
FIGURES_DIR = "./figs"  # Legacy generic figures
CACHE_DIR = "./cache"

# Analyzer-specific directories
RESULTS_DIR_SPOD = "./results_spod"
RESULTS_DIR_POD = "./results_pod"
RESULTS_DIR_BSMD = "./results_bsmd"
RESULTS_DIR_DMD = "./results_dmd"

FIGURES_DIR_SPOD = "./figs_spod"
FIGURES_DIR_POD = "./figs_pod"
FIGURES_DIR_BSMD = "./figs_bsmd"
FIGURES_DIR_DMD = "./figs_dmd"

# Optional preprocessing directories
PREPROCESS_DIR_SPOD = "./preprocess_spod"
PREPROCESS_DIR_POD = "./preprocess_pod"
PREPROCESS_DIR_BSMD = "./preprocess_bsmd"
PREPROCESS_DIR_DMD = "./preprocess_dmd"

# Figure saving options
FIG_DPI = 500
FIG_FORMAT = "png"  # or "pdf"

# FFT backend selection. Options include:
#  - 'scipy', 'numpy'
#  - 'tensorflow', 'torch'
#  - 'mkl' for Intel MKL via :mod:`mkl_fft`
#  - 'accelerate' for macOS vDSP/Accelerate
#  - 'cv2' (OpenCV)
# The name must match the keys defined in :mod:`fft.fft_backends`.
FFT_BACKEND = "scipy"

# Matplotlib/LaTeX options
USE_LATEX = False  # Set True to enable LaTeX rendering
FONT_FAMILY = "serif"
FONT_SIZE = 12
CMAP_SEQ = "viridis"  # Sequential colormap for general use
CMAP_DIV = "RdBu_r"  # Diverging colormap for signed data

# Default window type for FFT
WINDOW_TYPE = "hamming"
WINDOW_NORM = "power"

# Other global options can be added here as needed


def load_config(config_path):
    """Load a JSON or YAML configuration file and override defaults.

    Parameters
    ----------
    config_path : str
        Path to the configuration file. Supported formats are JSON
        and YAML (requires ``PyYAML``).
    """

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found")

    _, ext = os.path.splitext(config_path)
    ext = ext.lower()

    with open(config_path, "r") as f:
        if ext in {".yml", ".yaml"}:
            try:
                import yaml
            except Exception as exc:  # pragma: no cover - import error path
                raise ImportError("PyYAML must be installed to read YAML configuration files") from exc
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must define a dictionary")

    # Update any matching globals using upper-case keys
    for key, value in config.items():
        key_upper = key.upper()
        if key_upper in globals():
            globals()[key_upper] = value
