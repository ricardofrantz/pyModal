#!/usr/bin/env python3
"""
Simple Parallel Utilities for Modal Decomposition Analysis

This module provides optimized implementations using vectorized NumPy and
high-performance BLAS routines. OpenMP and Numba are no longer required; the
functions run on any standard Python installation.

Author: Modal Decomposition Team
"""

import multiprocessing

import numpy as np
from threadpoolctl import threadpool_info

# OpenMP support was removed. All routines rely on NumPy vectorization and the
# underlying BLAS implementation.
OPENMP_AVAILABLE = False
PARALLEL_AVAILABLE = True


def calculate_polar_weights_optimized(x, y):
    """
    Calculate integration weights for 2D cylindrical grid.

    This function uses a fully vectorized NumPy implementation that works on any
    platform without special dependencies.

    Parameters:
    -----------
    x : np.ndarray
        Axial coordinates
    y : np.ndarray
        Radial coordinates

    Returns:
    --------
    np.ndarray
        Integration weights, shape (Nx * Ny, 1)
    """
    return _calculate_weights_numpy(x, y)


def _calculate_weights_numpy(x, y):
    """Vectorized NumPy implementation of polar weights."""
    Nx, Ny = len(x), len(y)

    # Calculate y-direction (r-direction) integration weights (Wy) - vectorized
    Wy = np.zeros(Ny)

    if Ny > 1:
        # First point (centerline)
        y_mid_right = (y[0] + y[1]) / 2
        Wy[0] = np.pi * y_mid_right**2

        # Middle points - vectorized
        if Ny > 2:
            y_mid_left = (y[:-2] + y[1:-1]) / 2
            y_mid_right = (y[1:-1] + y[2:]) / 2
            Wy[1:-1] = np.pi * (y_mid_right**2 - y_mid_left**2)

        # Last point
        y_mid_left = (y[-2] + y[-1]) / 2
        Wy[-1] = np.pi * (y[-1] ** 2 - y_mid_left**2)
    else:
        Wy[0] = np.pi * y[0] ** 2

    # Calculate x-direction integration weights (Wx) - vectorized
    Wx = np.zeros(Nx)

    if Nx > 1:
        # First point
        Wx[0] = (x[1] - x[0]) / 2

        # Middle points - vectorized
        if Nx > 2:
            Wx[1:-1] = (x[2:] - x[:-2]) / 2

        # Last point
        Wx[-1] = (x[-1] - x[-2]) / 2
    else:
        Wx[0] = 1.0

    # Combine weights using outer product (much faster than loops)
    W = np.outer(Wx, Wy).flatten()

    return W.reshape(-1, 1)


# Placeholder function maintained for backward compatibility. It simply calls
# the NumPy implementation as OpenMP acceleration has been removed.
def _calculate_weights_openmp(x, y):
    return _calculate_weights_numpy(x, y)


def blocksfft_optimized(q, nfft, nblocks, novlap, blockwise_mean=False, normvar=False, window_norm="power", window_type="hamming"):
    """
    Optimized blocked FFT computation.

    This function uses the best available linear algebra backend (BLAS/LAPACK)
    and optimized memory access patterns for better performance.

    Parameters:
    -----------
    q : np.ndarray
        Input data [time, space]
    nfft : int
        Number of FFT points
    nblocks : int
        Number of blocks
    novlap : int
        Number of overlapping points between blocks
    blockwise_mean : bool
        Subtract blockwise mean if True
    normvar : bool
        Normalize variance if True
    window_norm : str
        Window normalization type ('amplitude' or 'power')
    window_type : str
        Window type ('hamming' or 'sine')

    Returns:
    --------
    np.ndarray
        FFT coefficients [freq, space, block]
    """
    # Import FFT backend
    from fft.fft_backends import get_fft_func

    # Select window function
    if window_type == "sine":
        window = np.sin(np.pi * (np.arange(nfft) + 0.5) / nfft)
    else:
        window = np.hamming(nfft)

    # Normalize window
    if window_norm == "amplitude":
        cw = 1.0 / window.mean()
    else:  # 'power' normalization (default)
        cw = 1.0 / np.sqrt(np.mean(window**2))

    nmesh = q.shape[1]  # Number of spatial points (Nx * Ny)
    n_freq_out = nfft // 2 + 1  # Number of frequency bins for one-sided spectrum
    q_hat = np.zeros((n_freq_out, nmesh, nblocks), dtype=complex)
    q_mean = np.mean(q, axis=0)  # Temporal mean (long-time mean)
    window_broadcast = window[:, np.newaxis]  # Reshape window for broadcasting

    # Process each block with optimized memory access
    fft_func = get_fft_func()

    for iblk in range(nblocks):
        ts = min(iblk * (nfft - novlap), q.shape[0] - nfft)  # Start index
        tf = np.arange(ts, ts + nfft)  # Time indices for the block
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
            block_var[block_var < 4 * np.finfo(float).eps] = 1.0  # Avoid division by zero
            block_centered = block_centered / block_var

        # Apply window and FFT
        windowed_block = block_centered * window_broadcast

        # Compute full FFT (uses optimized BLAS/LAPACK routines)
        full_fft_result = fft_func(windowed_block, axis=0)

        # Store only the one-sided spectrum (first n_freq_out points)
        q_hat[:, :, iblk] = (cw / nfft) * full_fft_result[:n_freq_out, :]

    return q_hat


def spod_single_frequency_optimized(qhat, w, nblocks, dst, num_modes=None, return_psi=False):
    """
    Optimized single frequency SPOD computation.

    Uses optimized BLAS/LAPACK routines for matrix operations.

    Parameters:
    -----------
    qhat : np.ndarray
        FFT coefficients for this frequency [space, block]
    w : np.ndarray
        Spatial integration weights [space, 1]
    nblocks : int
        Number of blocks
    dst : float
        Frequency resolution (delta f)
    num_modes : int, optional
        Number of modes to keep (default: all)

    Returns:
    --------
    tuple
        (phi, lambda_tilde)
        phi : Spatial SPOD modes [space, mode]
        lambda_tilde : SPOD eigenvalues [mode]
    """
    # Normalize FFT coefficients
    x = qhat / np.sqrt(nblocks * dst)

    # Compute the weighted cross-spectral density (CSD) matrix M_f
    # This uses optimized BLAS routines
    xprime_w = np.conj(x).T * w.T  # X_f^H * W
    m = xprime_w @ x  # (X_f^H * W) * X_f = M_f

    # Solve eigenvalue problem (uses optimized LAPACK)
    lambda_tilde, psi = np.linalg.eigh(m)

    # Sort eigenvalues and eigenvectors in descending order
    idx = lambda_tilde.argsort()[::-1]
    lambda_tilde = lambda_tilde[idx]
    psi = psi[:, idx]

    # Limit number of modes if specified
    if num_modes is not None:
        num_modes = min(num_modes, len(lambda_tilde))
        lambda_tilde = lambda_tilde[:num_modes]
        psi = psi[:, :num_modes]

    # Compute spatial SPOD modes
    inv_sqrt_lambda = np.zeros_like(lambda_tilde)
    mask = lambda_tilde > 1e-12
    inv_sqrt_lambda[mask] = 1.0 / np.sqrt(lambda_tilde[mask])

    # This uses optimized BLAS matrix multiplication
    phi = x @ (psi * inv_sqrt_lambda[np.newaxis, :])

    if return_psi:
        return phi, np.abs(lambda_tilde), psi
    return phi, np.abs(lambda_tilde)


def pod_computation_optimized(data_matrix, use_method="svd"):
    """
    Optimized POD computation using high-performance linear algebra.

    Parameters:
    -----------
    data_matrix : np.ndarray
        Data matrix [space, time]
    use_method : str
        Method to use ('svd' or 'covariance')

    Returns:
    --------
    tuple
        (phi, sigma, temporal_coeffs)
        phi : Spatial POD modes [space, mode]
        sigma : Singular values [mode]
        temporal_coeffs : Temporal coefficients [time, mode]
    """
    print("🚀 Using optimized POD computation...")

    # Center the data
    data_mean = np.mean(data_matrix, axis=1, keepdims=True)
    data_centered = data_matrix - data_mean

    if use_method == "svd":
        # Direct SVD - automatically uses optimized BLAS/LAPACK
        phi, sigma, vt = np.linalg.svd(data_centered, full_matrices=False)
        temporal_coeffs = vt.T * sigma
    else:
        # Choose method based on matrix shape for optimal performance
        if data_centered.shape[1] > data_centered.shape[0]:
            # More time steps than spatial points - use spatial covariance
            cov_matrix = (data_centered @ data_centered.T) / (data_centered.shape[1] - 1)
            eigenvals, phi = np.linalg.eigh(cov_matrix)

            # Sort in descending order
            idx = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[idx]
            phi = phi[:, idx]

            # Compute temporal coefficients
            sigma = np.sqrt(np.maximum(eigenvals, 0))
            temporal_coeffs = phi.T @ data_centered
        else:
            # More spatial points than time steps - use temporal covariance
            cov_matrix = (data_centered.T @ data_centered) / (data_centered.shape[1] - 1)
            eigenvals, temporal_modes = np.linalg.eigh(cov_matrix)

            # Sort in descending order
            idx = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[idx]
            temporal_modes = temporal_modes[:, idx]

            # Compute spatial modes
            sigma = np.sqrt(np.maximum(eigenvals, 0))
            phi = data_centered @ temporal_modes
            # Normalize spatial modes
            for i in range(phi.shape[1]):
                if sigma[i] > 1e-12:
                    phi[:, i] /= sigma[i]

            temporal_coeffs = temporal_modes * sigma

    return phi, sigma, temporal_coeffs


def get_optimization_info():
    """Return information about available optimizations."""
    info = {"parallel_available": PARALLEL_AVAILABLE, "cpu_count": multiprocessing.cpu_count(), "numpy_blas": "Unknown"}

    # Try to detect BLAS implementation
    try:
        from contextlib import redirect_stdout
        from io import StringIO

        import numpy as np

        # Capture config output instead of printing it
        with redirect_stdout(StringIO()) as config_output:
            np.__config__.show()
        config_info = config_output.getvalue()

        if "mkl" in str(config_info).lower():
            info["numpy_blas"] = "Intel MKL"
        elif "openblas" in str(config_info).lower():
            info["numpy_blas"] = "OpenBLAS"
        elif "atlas" in str(config_info).lower():
            info["numpy_blas"] = "ATLAS"
        else:
            info["numpy_blas"] = "Standard"
    except:
        info["numpy_blas"] = "Unknown"

    return info


def get_threadpool_summary():
    """Return a short description of active thread pools."""
    try:
        pools = threadpool_info()
        return ", ".join(f"{p.get('prefix', '')}{p.get('internal_api')}={p.get('num_threads')}" for p in pools) or "none"
    except Exception:
        return "unavailable"


def print_optimization_status():
    """Print current optimization status."""
    info = get_optimization_info()

    print("🔧 Optimization Status:")
    print(f"   Parallel Available: {info['parallel_available']}")
    print(f"   CPU Cores: {info['cpu_count']}")
    print(f"   NumPy BLAS: {info['numpy_blas']}")
    print(f"   Thread pools: {get_threadpool_summary()}")

    if info["parallel_available"]:
        print("   ⚡ High performance mode (vectorized)")
    else:
        print("   📊 Standard performance mode")


if __name__ == "__main__":
    print_optimization_status()
