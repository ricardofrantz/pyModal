"""
Central configuration for modal decomposition analysis.

NOTE: ALL imports are available here and this is imported in utils.py
so we only need to import utils in other files.
"""

import glob
import json
import os
import re
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from scipy.linalg import eig
from scipy.signal import find_peaks
from tqdm import tqdm

os.environ["OS_ACTIVITY_MODE"] = "disable"  # suppress macOS IMKClient logs
"""
Configuration and shared imports for modal decomposition tools.
"""

# Default directories
RESULTS_DIR = "./preprocess"
FIGURES_DIR = "./figs"
CACHE_DIR = "./cache"

# Figure saving options
FIG_DPI = 300
FIG_FORMAT = "png"  # or "pdf"

# FFT backend: "scipy", "numpy", "tensorflow", "torch", "cv2" (OpenCV)
# Should match the naming used in fft_benchmark.py
FFT_BACKEND = "scipy"  # Default, options: "scipy", "numpy", "tensorflow", "torch", "cv2"

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
