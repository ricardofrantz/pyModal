#!/usr/bin/env python3
"""
Parallel Utilities for Modal Decomposition Analysis

This module provides OpenMP-accelerated versions of computationally intensive
functions for FFT processing, matrix operations, and modal decomposition algorithms.

Author: Modal Decomposition Team
"""

import numpy as np
import os
import multiprocessing
from numba import jit, prange, config
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Optional, Tuple, Union

# Configure Numba for OpenMP
config.THREADING_LAYER = 'omp'

# Set OpenMP environment variables if not already set
def setup_openmp_environment():
    """Setup optimal OpenMP environment variables."""
    n_cores = multiprocessing.cpu_count()
    
    # Set number of threads if not already set
    if 'OMP_NUM_THREADS' not in os.environ:
        os.environ['OMP_NUM_THREADS'] = str(n_cores)
    
    # Set OpenMP scheduling
    if 'OMP_SCHEDULE' not in os.environ:
        os.environ['OMP_SCHEDULE'] = 'dynamic'
    
    # Set nested parallelism
    if 'OMP_NESTED' not in os.environ:
        os.environ['OMP_NESTED'] = 'true'
    
    print(f"🚀 OpenMP configured: {os.environ.get('OMP_NUM_THREADS')} threads")

# Initialize OpenMP
setup_openmp_environment()


@jit(nopython=True, parallel=True, cache=True)
def parallel_calculate_polar_weights(x, y):
    """
    Calculate integration weights for 2D cylindrical grid using OpenMP parallelization.
    
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
    Nx, Ny = len(x), len(y)
    
    # Calculate y-direction (r-direction) integration weights (Wy)
    Wy = np.zeros(Ny)
    
    # First point (centerline)
    if Ny > 1:
        y_mid_right = (y[0] + y[1]) / 2
        Wy[0] = np.pi * y_mid_right**2
    else:
        Wy[0] = np.pi * y[0] ** 2
    
    # Middle points - parallelized loop
    for i in prange(1, Ny - 1):
        y_mid_left = (y[i - 1] + y[i]) / 2
        y_mid_right = (y[i] + y[i + 1]) / 2
        Wy[i] = np.pi * (y_mid_right**2 - y_mid_left**2)
    
    # Last point
    if Ny > 1:
        y_mid_left = (y[-2] + y[-1]) / 2
        Wy[Ny - 1] = np.pi * (y[-1] ** 2 - y_mid_left**2)
    
    # Calculate x-direction integration weights (Wx)
    Wx = np.zeros(Nx)
    
    # First point
    if Nx > 1:
        Wx[0] = (x[1] - x[0]) / 2
    else:
        Wx[0] = 1.0
    
    # Middle points - parallelized loop
    for i in prange(1, Nx - 1):
        Wx[i] = (x[i + 1] - x[i - 1]) / 2
    
    # Last point
    if Nx > 1:
        Wx[Nx - 1] = (x[Nx - 1] - x[Nx - 2]) / 2
    
    # Combine weights
    W = np.zeros(Nx * Ny)
    for i in prange(Nx):
        for j in prange(Ny):
            W[i * Ny + j] = Wx[i] * Wy[j]
    
    return W.reshape(-1, 1)


@jit(nopython=True, parallel=True, cache=True)
def parallel_window_application(block_centered, window_broadcast):
    """
    Apply window function to block data using parallel processing.
    
    Parameters:
    -----------
    block_centered : np.ndarray
        Centered block data
    window_broadcast : np.ndarray
        Window function broadcasted to block shape
        
    Returns:
    --------
    np.ndarray
        Windowed block data
    """
    result = np.empty_like(block_centered)
    n_time, n_space = block_centered.shape
    
    for i in prange(n_time):
        for j in prange(n_space):
            result[i, j] = block_centered[i, j] * window_broadcast[i, 0]
    
    return result


def parallel_blocksfft(q, nfft, nblocks, novlap, blockwise_mean=False, normvar=False, 
                      window_norm="power", window_type="hamming", n_workers=None):
    """
    Compute blocked FFT using Welch's method with OpenMP parallelization.
    
    This function parallelizes the block processing loop, which is the most
    computationally intensive part of the FFT computation.
    
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
    n_workers : int, optional
        Number of worker threads (default: number of CPU cores)
        
    Returns:
    --------
    np.ndarray
        FFT coefficients [freq, space, block]
    """
    from fft.fft_backends import get_fft_func
    
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    
    # Select window function
    if window_type == "sine":
        from utils import sine_window
        window = sine_window(nfft)
    else:
        window = np.hamming(nfft)
    
    # Normalize window
    if window_norm == "amplitude":
        cw = 1.0 / window.mean()
    else:  # 'power' normalization (default)
        cw = 1.0 / np.sqrt(np.mean(window**2))
    
    nmesh = q.shape[1]  # Number of spatial points
    n_freq_out = nfft // 2 + 1  # Number of frequency bins for one-sided spectrum
    q_hat = np.zeros((n_freq_out, nmesh, nblocks), dtype=complex)
    q_mean = np.mean(q, axis=0)  # Temporal mean
    window_broadcast = window[:, np.newaxis]  # Reshape window for broadcasting
    
    def process_block(iblk):
        """Process a single FFT block."""
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
            block_var[block_var < 4 * np.finfo(float).eps] = 1.0
            block_centered = block_centered / block_var
        
        # Apply window - use parallel function for large blocks
        if block_centered.size > 10000:
            windowed_block = parallel_window_application(block_centered, window_broadcast)
        else:
            windowed_block = block_centered * window_broadcast
        
        # Compute FFT
        fft_func = get_fft_func()
        full_fft_result = fft_func(windowed_block, axis=0)
        
        # Store only the one-sided spectrum
        return iblk, (cw / nfft) * full_fft_result[:n_freq_out, :]
    
    # Process blocks in parallel
    print(f"🔄 Processing {nblocks} FFT blocks with {n_workers} threads...")
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all block processing tasks
        future_to_block = {executor.submit(process_block, iblk): iblk for iblk in range(nblocks)}
        
        # Collect results
        for future in as_completed(future_to_block):
            iblk, result = future.result()
            q_hat[:, :, iblk] = result
    
    print(f"✅ FFT block processing completed")
    return q_hat


@jit(nopython=True, parallel=True, cache=True)
def parallel_spod_computation_core(qhat_freq, nblocks, dst, w_flat):
    """
    Core SPOD computation for a single frequency with OpenMP parallelization.
    
    Parameters:
    -----------
    qhat_freq : np.ndarray
        FFT coefficients for this frequency [space, block]
    nblocks : int
        Number of blocks
    dst : float
        Frequency resolution
    w_flat : np.ndarray
        Flattened spatial weights
        
    Returns:
    --------
    tuple
        (spatial_modes, eigenvalues, time_coefficients)
    """
    # Normalize FFT coefficients
    x = qhat_freq / np.sqrt(nblocks * dst)
    
    # Compute weighted cross-spectral density matrix in parallel
    nspace, nblocks_actual = x.shape
    m = np.zeros((nblocks_actual, nblocks_actual), dtype=np.complex128)
    
    # Parallel computation of CSD matrix
    for b1 in prange(nblocks_actual):
        for b2 in prange(nblocks_actual):
            temp_sum = 0.0 + 0.0j
            for j in prange(nspace):
                temp_sum += np.conj(x[j, b1]) * w_flat[j] * x[j, b2]
            m[b1, b2] = temp_sum
    
    return m


def parallel_spod_frequency_loop(qhat, nblocks, dst, W, n_workers=None, return_psi=False):
    """
    Parallel SPOD computation across frequency bins.
    
    This is the main computational bottleneck in SPOD analysis.
    Each frequency can be processed independently.
    
    Parameters:
    -----------
    qhat : np.ndarray
        FFT coefficients [freq, space, block]
    nblocks : int
        Number of blocks
    dst : float
        Frequency resolution
    W : np.ndarray
        Spatial weights
    n_workers : int, optional
        Number of worker threads
    return_psi : bool
        Whether to return time coefficients
        
    Returns:
    --------
    tuple
        (spatial_modes, eigenvalues, time_coefficients)
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    
    n_freq, n_space, _ = qhat.shape
    w_flat = W.flatten()
    
    def process_frequency(freq_idx):
        """Process SPOD for a single frequency."""
        qhat_freq = qhat[freq_idx, :, :]
        
        # Use parallel core computation for large problems
        if n_space * nblocks > 10000:
            m = parallel_spod_computation_core(qhat_freq, nblocks, dst, w_flat)
        else:
            # Standard computation for smaller problems
            x = qhat_freq / np.sqrt(nblocks * dst)
            xprime_w = np.transpose(np.conj(x)) * np.transpose(W)
            m = xprime_w @ x
        
        # Eigenvalue decomposition
        lambda_tilde, psi = np.linalg.eigh(m)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = lambda_tilde.argsort()[::-1]
        lambda_tilde = lambda_tilde[idx]
        psi = psi[:, idx]
        
        # Compute spatial modes
        x = qhat_freq / np.sqrt(nblocks * dst)
        inv_sqrt_lambda = np.zeros_like(lambda_tilde)
        mask = lambda_tilde > 1e-12
        inv_sqrt_lambda[mask] = 1.0 / np.sqrt(lambda_tilde[mask])
        phi = x @ psi @ np.diag(inv_sqrt_lambda)
        
        if return_psi:
            return freq_idx, phi, np.abs(lambda_tilde), psi
        else:
            return freq_idx, phi, np.abs(lambda_tilde)
    
    # Process frequencies in parallel
    print(f"🔄 Computing SPOD for {n_freq} frequencies with {n_workers} threads...")
    
    phi_all = np.zeros((n_freq, n_space, nblocks), dtype=complex)
    lambda_all = np.zeros((n_freq, nblocks))
    psi_all = np.zeros((n_freq, nblocks, nblocks), dtype=complex) if return_psi else None
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all frequency processing tasks
        future_to_freq = {executor.submit(process_frequency, freq_idx): freq_idx 
                         for freq_idx in range(n_freq)}
        
        # Collect results
        for future in as_completed(future_to_freq):
            if return_psi:
                freq_idx, phi, lambda_vals, psi = future.result()
                phi_all[freq_idx] = phi
                lambda_all[freq_idx] = lambda_vals
                psi_all[freq_idx] = psi
            else:
                freq_idx, phi, lambda_vals = future.result()
                phi_all[freq_idx] = phi
                lambda_all[freq_idx] = lambda_vals
    
    print(f"✅ SPOD frequency loop completed")
    
    if return_psi:
        return phi_all, lambda_all, psi_all
    else:
        return phi_all, lambda_all


@jit(nopython=True, parallel=True, cache=True)
def parallel_matrix_multiply(A, B):
    """
    Parallel matrix multiplication using OpenMP.
    
    Parameters:
    -----------
    A : np.ndarray
        Left matrix
    B : np.ndarray
        Right matrix
        
    Returns:
    --------
    np.ndarray
        Matrix product A @ B
    """
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, "Matrix dimensions don't match"
    
    C = np.zeros((m, n))
    
    for i in prange(m):
        for j in prange(n):
            temp_sum = 0.0
            for l in range(k):
                temp_sum += A[i, l] * B[l, j]
            C[i, j] = temp_sum
    
    return C


@jit(nopython=True, parallel=True, cache=True)
def parallel_covariance_matrix(data_weighted):
    """
    Compute covariance matrix with OpenMP parallelization.
    
    Parameters:
    -----------
    data_weighted : np.ndarray
        Weighted data matrix [time, space]
        
    Returns:
    --------
    np.ndarray
        Covariance matrix [time, time]
    """
    Ns, _ = data_weighted.shape
    K = np.zeros((Ns, Ns))
    
    # Parallel computation of covariance matrix
    for i in prange(Ns):
        for j in prange(i, Ns):  # Use symmetry
            temp_sum = 0.0
            for k in range(data_weighted.shape[1]):
                temp_sum += data_weighted[i, k] * data_weighted[j, k]
            K[i, j] = temp_sum / Ns
            if i != j:
                K[j, i] = K[i, j]  # Symmetry
    
    return K


def parallel_reconstruction_error(time_coeffs, modes, data_mean_removed, n_modes_check=None, n_workers=None):
    """
    Compute reconstruction error for different numbers of modes in parallel.
    
    Parameters:
    -----------
    time_coeffs : np.ndarray
        Time coefficients
    modes : np.ndarray
        Spatial modes
    data_mean_removed : np.ndarray
        Mean-removed data
    n_modes_check : int, optional
        Number of modes to check
    n_workers : int, optional
        Number of worker threads
        
    Returns:
    --------
    tuple
        (mode_numbers, reconstruction_errors)
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    
    if n_modes_check is None:
        n_modes_check = min(modes.shape[1], 20)
    
    def compute_error_for_k_modes(k):
        """Compute reconstruction error for k modes."""
        reconstructed = time_coeffs[:, :k] @ modes[:, :k].T
        error = np.linalg.norm(data_mean_removed - reconstructed, "fro")
        relative_error = error / np.linalg.norm(data_mean_removed, "fro")
        return k, relative_error
    
    print(f"🔄 Computing reconstruction errors for {n_modes_check} modes with {n_workers} threads...")
    
    mode_numbers = []
    errors = []
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all error computation tasks
        future_to_k = {executor.submit(compute_error_for_k_modes, k): k 
                      for k in range(1, n_modes_check + 1)}
        
        # Collect results
        for future in as_completed(future_to_k):
            k, error = future.result()
            mode_numbers.append(k)
            errors.append(error)
    
    # Sort by mode number
    sorted_pairs = sorted(zip(mode_numbers, errors))
    mode_numbers, errors = zip(*sorted_pairs)
    
    print(f"✅ Reconstruction error computation completed")
    return np.array(mode_numbers), np.array(errors)


def set_openmp_threads(n_threads=None):
    """
    Set the number of OpenMP threads.
    
    Parameters:
    -----------
    n_threads : int, optional
        Number of threads to use. If None, use all available cores.
    """
    if n_threads is None:
        n_threads = multiprocessing.cpu_count()
    
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    
    # Also set for common libraries
    os.environ['MKL_NUM_THREADS'] = str(n_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)
    
    print(f"🚀 OpenMP threads set to: {n_threads}")


def get_parallel_info():
    """Return information about parallel configuration."""
    return {
        'omp_threads': os.environ.get('OMP_NUM_THREADS', 'Not set'),
        'cpu_count': multiprocessing.cpu_count(),
        'numba_threading': config.THREADING_LAYER,
        'mkl_threads': os.environ.get('MKL_NUM_THREADS', 'Not set'),
        'openblas_threads': os.environ.get('OPENBLAS_NUM_THREADS', 'Not set')
    }


@jit(nopython=True, parallel=True, cache=True)
def parallel_cross_spectral_density(qhat, w_flat):
    """
    Compute cross-spectral density matrix using parallel processing.
    
    This is the core computational kernel for SPOD eigenvalue problems.
    
    Parameters:
    -----------
    qhat : np.ndarray
        FFT coefficients for single frequency [space, block]
    w_flat : np.ndarray
        Flattened spatial weights [space]
        
    Returns:
    --------
    np.ndarray
        Cross-spectral density matrix [block, block]
    """
    n_space, n_blocks = qhat.shape
    
    # Compute weighted fluctuation matrix: X_f^H * W
    # This is the most computationally intensive part
    xprime_w = np.zeros((n_blocks, n_space), dtype=numba.complex128)
    
    # Parallel computation of weighted transpose
    for i in prange(n_blocks):
        for j in prange(n_space):
            xprime_w[i, j] = np.conj(qhat[j, i]) * w_flat[j]
    
    # Compute M_f = (X_f^H * W) * X_f  
    m = np.zeros((n_blocks, n_blocks), dtype=numba.complex128)
    for i in prange(n_blocks):
        for j in prange(n_blocks):
            temp = 0.0 + 0.0j
            for k in range(n_space):
                temp += xprime_w[i, k] * qhat[k, j]
            m[i, j] = temp
    
    return m


@jit(nopython=True, parallel=True, cache=True)
def parallel_spatial_modes_computation(qhat, psi, inv_sqrt_lambda):
    """
    Compute spatial SPOD modes using parallel processing.
    
    Parameters:
    -----------
    qhat : np.ndarray
        FFT coefficients [space, block]
    psi : np.ndarray
        Temporal eigenvectors [block, mode]  
    inv_sqrt_lambda : np.ndarray
        Inverse square root of eigenvalues [mode]
        
    Returns:
    --------
    np.ndarray
        Spatial SPOD modes [space, mode]
    """
    n_space, n_blocks = qhat.shape
    n_modes = psi.shape[1]
    
    # Phi_f = X_f * Psi_f * Lambda_f^(-1/2)
    phi = np.zeros((n_space, n_modes), dtype=numba.complex128)
    
    # First compute X_f * Psi_f
    temp = np.zeros((n_space, n_modes), dtype=numba.complex128)
    for i in prange(n_space):
        for j in prange(n_modes):
            val = 0.0 + 0.0j
            for k in range(n_blocks):
                val += qhat[i, k] * psi[k, j]
            temp[i, j] = val
    
    # Then multiply by Lambda_f^(-1/2)
    for i in prange(n_space):
        for j in prange(n_modes):
            phi[i, j] = temp[i, j] * inv_sqrt_lambda[j]
    
    return phi


def parallel_spod_single_frequency(qhat, w, nblocks, dst, num_modes=None):
    """
    Enhanced single frequency SPOD computation with parallel kernels.
    
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
    # Normalize FFT coefficients to get fluctuation matrix X_f for this frequency f.
    x = qhat / np.sqrt(nblocks * dst)
    
    # Use parallel cross-spectral density computation for large problems
    if x.shape[0] > 1000 and x.shape[1] > 8:
        w_flat = w.flatten()
        m = parallel_cross_spectral_density(x, w_flat)
    else:
        # Standard computation for smaller problems
        xprime_w = np.transpose(np.conj(x)) * np.transpose(w)
        m = xprime_w @ x
    
    # Solve the eigenvalue problem: M_f * Psi_f = Psi_f * Lambda_f
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
    
    # Compute spatial SPOD modes (Phi_f) of the direct problem
    inv_sqrt_lambda = np.zeros_like(lambda_tilde)
    mask = lambda_tilde > 1e-12
    inv_sqrt_lambda[mask] = 1.0 / np.sqrt(lambda_tilde[mask])
    
    # Use parallel spatial modes computation for large problems
    if x.shape[0] > 1000 and psi.shape[1] > 4:
        phi = parallel_spatial_modes_computation(x, psi, inv_sqrt_lambda)
    else:
        # Standard computation for smaller problems
        phi = x @ psi @ np.diag(inv_sqrt_lambda)
    
    return phi, np.abs(lambda_tilde)


def parallel_pod_computation(data_matrix, use_method='svd'):
    """
    Parallel POD computation using optimized linear algebra.
    
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
    print("🚀 Using parallel POD computation...")
    
    # Center the data
    data_mean = np.mean(data_matrix, axis=1, keepdims=True)
    data_centered = data_matrix - data_mean
    
    if use_method == 'svd':
        # Direct SVD - automatically uses parallel BLAS
        phi, sigma, vt = np.linalg.svd(data_centered, full_matrices=False)
        temporal_coeffs = vt.T * sigma
    else:
        # Covariance method for wide matrices (more time steps than spatial points)
        if data_centered.shape[1] > data_centered.shape[0]:
            # Spatial covariance: C = (1/N) * X * X^T
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
            # Temporal covariance: C = (1/N) * X^T * X  
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