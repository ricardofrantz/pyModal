#!/usr/bin/env python3
"""
Common utilities for modal decomposition methods.

All imports are centralized here to keep the code clean and consistent.
"""

import os
import numpy as np
import h5py
import time

from configs import *
from data_interface import auto_detect_weight_type as di_auto_detect_weight_type
from data_interface import load_data as di_load_data
from data_interface import load_jetles_data as di_load_jetles_data
from data_interface import load_mat_data as di_load_mat_data
from fft.fft_backends import get_fft_func
try:
    from parallel_utils import (
        PARALLEL_AVAILABLE,
        blocksfft_optimized,
        calculate_polar_weights_optimized,
        spod_single_frequency_optimized,
    )
except Exception:
    PARALLEL_AVAILABLE = False


def get_num_threads():
    """Return thread count from ``OMP_NUM_THREADS`` or ``os.cpu_count()``."""
    env = os.environ.get("OMP_NUM_THREADS")
    try:
        val = int(env) if env is not None else None
    except (TypeError, ValueError):
        val = None
    if val is not None and val > 0:
        return val
    cpu = os.cpu_count() or 1
    return cpu


def parallel_map(func, iterable, threads=None):
    """Map function over iterable using threads."""
    threads = threads or get_num_threads()
    if threads <= 1:
        return [func(x) for x in iterable]
    results = []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(func, x) for x in iterable]
        for f in futures:
            results.append(f.result())
    return results


def make_result_filename(root, nfft, overlap, Ns, analysis):
    """
    Generate a harmonized result filename for analysis outputs.
    Args:
        root (str): Base name of the dataset (no extension)
        nfft (int): FFT block size
        overlap (float): Overlap fraction (0-1)
        Ns (int): Number of snapshots
        analysis (str): Analysis type (e.g., 'spod', 'bsmd')
    Returns:
        str: Result filename (always .hdf5)
    """
    return f"{root}_Nfft{nfft}_ovlap{overlap}_{Ns}snapshots_{analysis}.hdf5"


def print_summary(analysis: str, results_dir: str, figures_dir: str) -> None:
    """Print a short summary of where results and figures were saved."""
    print(f"✅ {analysis} analysis finished!")
    print(f"📁 Results: {results_dir}")
    print(f"📊 Figures: {figures_dir}")


def compute_aspect_ratio(x_coords, y_coords):
    """Return ``dy/dx`` if coordinates are 1D vectors, else ``'auto'``."""
    if hasattr(x_coords, "ndim") and hasattr(y_coords, "ndim"):
        if x_coords.ndim == 1 and y_coords.ndim == 1:
            dx = float(x_coords.max() - x_coords.min())
            dy = float(y_coords.max() - y_coords.min())
            if dx > 0 and dy > 0:
                return dy / dx
    return "auto"


from typing import Union

def get_aspect_ratio(data: dict) -> Union[float, str]:
    """Return aspect ratio for ``data`` using available coordinates."""
    x = data.get("x", [])
    y = data.get("y", [])
    return compute_aspect_ratio(x, y)


def get_fig_aspect_ratio(data: dict, clamp_low: float = 0.5, clamp_high: float = 2.0) -> float:
    """Return ``Nx/Ny`` ratio clamped for figure sizing."""
    nx = int(data.get("Nx", 1))
    ny = int(data.get("Ny", 1))
    if ny <= 0:
        aspect = 1.0
    else:
        aspect = nx / ny
    return max(clamp_low, min(aspect, clamp_high))


def load_jetles_data(file_path):
    return di_load_jetles_data(file_path)


def load_mat_data(file_path):
    return di_load_mat_data(file_path)


def load_data(file_path):
    return di_load_data(file_path)


def generate_dummy_data_like_jetles(
    output_path: str,
    Ns: int = 100,
    Nx: int = 30,
    Ny: int = 20,
    dt: float = 0.01,
    f1: float = 5.0,
    f2: float = 2.0,
    noise_level: float = 0.05,
    save_mat: bool = False,
) -> str:
    """Create a small JetLES-like dataset with simple coherent content.

    This utility generates a synthetic pressure field composed of a few
    low-frequency modes rather than purely random noise.  It is intended for
    quick demonstrations when no real dataset is available.

    Parameters
    ----------
    output_path : str
        Path to the file to create.
    Ns : int, optional
        Number of snapshots (time samples).
    Nx : int, optional
        Number of points in the ``x`` direction.
    Ny : int, optional
        Number of points in the radial ``r`` direction.
    dt : float, optional
        Time step between snapshots.
    f1, f2 : float, optional
        Dominant temporal frequencies of the two synthetic modes.
    noise_level : float, optional
        Amplitude of added Gaussian noise relative to the signal.
    save_mat : bool, optional
        If ``True`` the file is created with ``.mat`` extension, otherwise an
        HDF5 ``.h5`` file is created.  The function does not require SciPy and
        always uses ``h5py`` for writing.
    Returns
    -------
    str
        Path to the generated dummy file.
    """

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Coordinates stored as 2-D arrays as in the real dataset
    x = np.linspace(0.0, 1.0, Nx)[:, None]
    r = np.linspace(0.0, 1.0, Ny)[None, :]

    # Temporal vector
    t = np.arange(Ns) * dt

    # Simple spatial modes
    mode1 = np.sin(np.pi * x) * np.cos(np.pi * r)
    mode2 = np.cos(0.5 * np.pi * x) * np.sin(2.0 * np.pi * r)

    # Construct coherent pressure field (shape: Nx, Ny, Ns)
    signal = np.sin(2 * np.pi * f1 * t)[:, None, None] * mode1[None, :, :] + 0.5 * np.sin(2 * np.pi * f2 * t)[:, None, None] * mode2[None, :, :]

    noise = noise_level * np.random.randn(Ns, Nx, Ny)
    p = np.transpose(signal + noise, (1, 2, 0))  # (Nx, Ny, Ns)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("p", data=p)
        f.create_dataset("x", data=x)
        f.create_dataset("r", data=r)
        f.create_dataset("dt", data=np.array([[dt]]))

    # Optionally save a ``.mat`` file for compatibility with some loaders
    if save_mat and not output_path.endswith(".mat"):
        mat_path = os.path.splitext(output_path)[0] + ".mat"
        with h5py.File(mat_path, "w") as f:
            f.create_dataset("p", data=p)
            f.create_dataset("x", data=x)
            f.create_dataset("r", data=r)
            f.create_dataset("dt", data=np.array([[dt]]))

    return output_path


def calculate_polar_weights(x, y, use_parallel=True):
    """Calculate integration weights for a 2D cylindrical grid (x, r)."""
    if use_parallel and PARALLEL_AVAILABLE:
        return calculate_polar_weights_optimized(x, y)
    Nx, Ny = x.shape[0], y.shape[0]

    # Calculate y-direction (r-direction) integration weights (Wy)
    Wy = np.zeros((Ny, 1))

    # First point (centerline)
    if Ny > 1:
        y_mid_right = (y[0] + y[1]) / 2
        Wy[0] = np.pi * y_mid_right**2
    else:
        Wy[0] = np.pi * y[0] ** 2

    # Middle points
    for i in range(1, Ny - 1):
        y_mid_left = (y[i - 1] + y[i]) / 2
        y_mid_right = (y[i] + y[i + 1]) / 2
        Wy[i] = np.pi * (y_mid_right**2 - y_mid_left**2)

    # Last point
    if Ny > 1:
        y_mid_left = (y[-2] + y[-1]) / 2
        Wy[Ny - 1] = np.pi * (y[-1] ** 2 - y_mid_left**2)

    # Calculate x-direction integration weights (Wx)
    Wx = np.zeros((Nx, 1))

    # First point
    if Nx > 1:
        Wx[0] = (x[1] - x[0]) / 2
    else:
        Wx[0] = 1.0

    # Middle points
    for i in range(1, Nx - 1):
        Wx[i] = (x[i + 1] - x[i - 1]) / 2

    # Last point
    if Nx > 1:
        Wx[Nx - 1] = (x[Nx - 1] - x[Nx - 2]) / 2

    # Combine weights
    W = np.reshape(Wx @ np.transpose(Wy), (Nx * Ny, 1))

    return W


def calculate_uniform_weights(x, y):
    """Return uniform weights for a 2D grid (Cartesian)."""
    Nx, Ny = x.shape[0], y.shape[0]
    return np.ones((Nx * Ny, 1))


def sine_window(n):
    """Return a sine window of length n."""
    return np.sin(np.pi * (np.arange(n) + 0.5) / n)


def blocksfft(
    q,
    nfft,
    nblocks,
    novlap,
    blockwise_mean=False,
    normvar=False,
    window_norm="power",
    window_type="hamming",
    n_threads=None,
    use_parallel=True,
):
    """
    Compute blocked FFT using Welch's method for CSD estimation.

    If ``use_parallel`` is True and optimized routines are available,
    ``blocksfft_optimized`` from :mod:`parallel_utils` is used.

    Parameters:
    q (np.ndarray): Input data [time, space]
    nfft (int): Number of FFT points
    nblocks (int): Number of blocks
    novlap (int): Number of overlapping points between blocks
    blockwise_mean (bool): Subtract blockwise mean if True
    normvar (bool): Normalize variance if True
    window_norm (str): Window normalization type ('amplitude' or 'power')
    window_type (str): Window type. Use 'sine' for the custom sine window or any
        name recognized by ``scipy.signal.get_window`` (e.g., 'hamming', 'hann',
        'blackman', etc.)

    Returns:
    q_hat (np.ndarray): FFT coefficients [freq, space, block]

    ---
    IMPORTANT:
    - This function assumes the FFT backend (numpy, scipy, pyfftw, etc.) does NOT normalize the FFT by default (which is true for standard backends).
    - If you use a backend or option that applies normalization (e.g., norm='ortho'), REMOVE the division by nfft below to avoid double normalization.
    - For correct SPOD scaling, ensure that dst (frequency resolution) is set as fs / nfft, where fs is the sampling frequency.
    ---
    """
    if use_parallel and PARALLEL_AVAILABLE:
        return blocksfft_optimized(
            q,
            nfft,
            nblocks,
            novlap,
            blockwise_mean=blockwise_mean,
            normvar=normvar,
            window_norm=window_norm,
            window_type=window_type,
        )

    # Select window function
    if window_type == "sine":
        window = sine_window(nfft)
    else:
        window = get_window(window_type, nfft)

    # Normalize window
    if window_norm == "amplitude":
        cw = 1.0 / window.mean()
    else:  # 'power' normalization (default)
        cw = 1.0 / np.sqrt(np.mean(window**2))

    nmesh = q.shape[1]  # Number of spatial points (Nx * Ny)
    n_freq_out = nfft // 2 + 1  # Number of frequency bins for one-sided spectrum
    q_hat = np.zeros((n_freq_out, nmesh, nblocks), dtype=complex)
    q_mean = np.mean(q, axis=0)
    window_broadcast = window[:, np.newaxis]

    # ``n_threads`` is accepted for backward compatibility but FFT blocks are
    # processed sequentially to avoid oversubscribing underlying math libraries.
    for iblk in range(nblocks):
        ts = min(iblk * (nfft - novlap), q.shape[0] - nfft)
        tf = np.arange(ts, ts + nfft)
        block = q[tf, :]

        # Subtract mean
        if blockwise_mean:
            block_mean = np.mean(block, axis=0)
        else:
            block_mean = q_mean
        block_centered = block - block_mean

        # Normalize variance if requested
        if normvar:
            block_var = np.var(block_centered, axis=0, ddof=1)
            block_var[block_var < 4 * np.finfo(float).eps] = 1.0
            block_centered = block_centered / block_var

        # Apply window and FFT
        fft_func = get_fft_func()
        full_fft_result = fft_func(block_centered * window_broadcast, axis=0)

        # Store one-sided spectrum
        q_hat[:, :, iblk] = (cw / nfft) * full_fft_result[:n_freq_out, :]

    return q_hat


def auto_detect_weight_type(file_path):
    # Always return 'uniform' for dNamiX consolidated .npz files
    if file_path.lower().endswith('.npz'):
        return 'uniform'
    return di_auto_detect_weight_type(file_path)



def spod_function(qhat, nblocks, dst, w, return_psi=False, use_parallel=True):
    """
    Compute SPOD modes and eigenvalues for a single frequency.
    Args:
        qhat (np.ndarray): FFT coefficients for this frequency [space, block].
        nblocks (int): Number of blocks.
        dst (float): Frequency resolution (delta f).
        w (np.ndarray): Spatial integration weights [space, 1].
        return_psi (bool): If True, also return psi (time coefficients).
    Returns:
        tuple: (phi, lambda_tilde[, psi])
            phi (np.ndarray): Spatial SPOD modes for this frequency [space, mode].
            lambda_tilde (np.ndarray): SPOD eigenvalues (energy) for this frequency [mode].
            psi (np.ndarray, optional): Time coefficients for this frequency [block, mode].
    """
    if use_parallel and PARALLEL_AVAILABLE:
        return spod_single_frequency_optimized(
            qhat,
            w,
            nblocks,
            dst,
            return_psi=return_psi,
        )

    # Normalize FFT coefficients to get fluctuation matrix X_f for this frequency f.
    x = qhat / np.sqrt(nblocks * dst)
    # Compute the weighted cross-spectral density (CSD) matrix M_f.
    xprime_w = np.transpose(np.conj(x)) * np.transpose(w)  # X_f^H * W
    m = xprime_w @ x  # (X_f^H * W) * X_f = M_f
    del xprime_w
    # Solve the eigenvalue problem: M_f * Psi_f = Psi_f * Lambda_f
    lambda_tilde, psi = np.linalg.eigh(m)
    # Sort eigenvalues and eigenvectors in descending order
    idx = lambda_tilde.argsort()[::-1]
    lambda_tilde = lambda_tilde[idx]
    psi = psi[:, idx]
    # Compute spatial SPOD modes (Phi_f) of the direct problem.
    inv_sqrt_lambda = np.zeros_like(lambda_tilde)
    mask = lambda_tilde > 1e-12
    inv_sqrt_lambda[mask] = 1.0 / np.sqrt(lambda_tilde[mask])
    phi = x @ psi @ np.diag(inv_sqrt_lambda)
    if return_psi:
        return phi, np.abs(lambda_tilde), psi
    return phi, np.abs(lambda_tilde)


class BaseAnalyzer:
    """Base class for modal decomposition analyzers."""

    def __init__(
        self,
        file_path,
        nfft=128,
        overlap=0.5,
        results_dir="./preprocess",
        figures_dir="./figs",
        data_loader=None,
        spatial_weight_type="auto",
        n_threads=None,
        use_parallel=True,
    ):
        """Initialize the analyzer.

        Args:
            file_path (str): Path to data file.
            nfft (int): Number of snapshots per FFT block.
            overlap (float): Overlap fraction between blocks.
            results_dir (str): Directory to save results.
            figures_dir (str): Directory to save figures.
            data_loader (callable): Function to load data.
            spatial_weight_type (str): Type of spatial weighting.
        """
        self.file_path = file_path
        self.nfft = nfft
        self.overlap = overlap
        self.results_dir = results_dir
        self.figures_dir = figures_dir

        # Set default data loader based on file type
        self.data_loader = data_loader or load_data
        self.n_threads = n_threads if n_threads is not None else get_num_threads()
        self.use_parallel = use_parallel

        # Set default weight type
        if spatial_weight_type == "auto":
            self.spatial_weight_type = auto_detect_weight_type(file_path)
        else:
            self.spatial_weight_type = spatial_weight_type

        # Calculated later
        self.novlap = int(overlap * nfft)
        self.data = {}
        self.W = np.array([])
        self.nblocks = 0
        self.fs = 0.0
        self.qhat = np.array([])

        # Extract root name for output files
        base = os.path.basename(file_path)
        self.data_root = os.path.splitext(base)[0]

        # Ensure output directories exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def load_and_preprocess(self):
        """Load data and calculate weights."""
        # Load data from file
        self.data = self.data_loader(self.file_path)

        # Calculate spatial weights
        if self.spatial_weight_type == "polar":
            self.W = calculate_polar_weights(self.data["x"], self.data["y"], use_parallel=self.use_parallel)
            print("Using polar (cylindrical) spatial weights.")
        else:
            self.W = calculate_uniform_weights(self.data["x"], self.data["y"])
            print("Using uniform spatial weights (rectangular grid).")

        # Calculate derived parameters
        self.nblocks = int(np.ceil((self.data["Ns"] - self.novlap) / (self.nfft - self.novlap)))
        self.fs = 1 / self.data["dt"]

        print(f"Data loaded: {self.data['Ns']} snapshots, {self.data['Nx']}×{self.data['Ny']} spatial points")
        if self.nfft > 1:
            print(f"FFT parameters: {self.nfft} points, {self.overlap * 100}% overlap, {self.nblocks} blocks [backend: {FFT_BACKEND}]")

    def compute_fft_blocks(self):
        """Compute blocked FFT using Welch's method."""
        if "q" not in self.data:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")

        print(f"Computing FFT with {self.nblocks} blocks using {self.n_threads} threads on {FFT_BACKEND} backend...")
        self.qhat = blocksfft(
            self.data["q"],
            self.nfft,
            self.nblocks,
            self.novlap,
            blockwise_mean=getattr(self, "blockwise_mean", False),
            normvar=getattr(self, "normvar", False),
            window_norm=getattr(self, "window_norm", "power"),
            window_type=getattr(self, "window_type", "hamming"),
            n_threads=self.n_threads,
            use_parallel=self.use_parallel,
        )
        print("FFT computation complete.")

    def save_results(self, filename=None, analysis_type="spod"):
        """Save results to HDF5 file with harmonized filename and format.
        Args:
            filename (str, optional): Custom filename. If not provided, uses harmonized scheme.
            analysis_type (str): Analysis type for filename (e.g., 'spod', 'bsmd').
        """
        if not filename:
            filename = make_result_filename(self.data_root, self.nfft, self.overlap, self.data.get("Ns", 0), analysis_type)
        save_path = os.path.join(self.results_dir, filename)
        print(f"Saving results to {save_path}")
        # This is a placeholder - subclasses should implement specific saving logic
        with h5py.File(save_path, "w") as f:
            f.attrs["nfft"] = self.nfft
            f.attrs["overlap"] = self.overlap
            f.attrs["nblocks"] = self.nblocks
            f.attrs["fs"] = self.fs
            # Save coordinates
            f.create_dataset("x", data=self.data["x"], compression="gzip")
            f.create_dataset("y", data=self.data["y"], compression="gzip")
            # Save weights
            f.create_dataset("W", data=self.W, compression="gzip")

    def run(self, compute_fft=True):
        """Run the full analysis pipeline."""
        start_time = time.time()

        # Load data and calculate weights
        self.load_and_preprocess()

        # Compute FFT blocks if requested
        if compute_fft:
            self.compute_fft_blocks()

        end_time = time.time()
        print(f"Completed in {end_time - start_time:.2f} seconds.")

        return self

    def _get_metadata(self):
        """Return a dictionary of common metadata for saving results."""
        meta = {
            "analysis_type": getattr(self, "analysis_type", ""),
            "data_file": self.file_path,
            "nfft": self.nfft,
            "overlap": self.overlap,
            "nblocks": self.nblocks,
            "fs": self.fs,
            "dt": self.data.get("dt", 0),
            "Ns": self.data.get("Ns", 0),
            "Nx": self.data.get("Nx", 0),
            "Ny": self.data.get("Ny", 0),
            "spatial_weight_type": self.spatial_weight_type,
        }
        for attr in ["window_type", "window_norm", "blockwise_mean", "normvar"]:
            if hasattr(self, attr):
                meta[attr] = getattr(self, attr)
        return meta
