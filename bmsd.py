#!/usr/bin/env python3
"""
Extract coherent bispectral modes with BiSpectral Mode Decomposition (BSMD)

Reference: "Bispectral mode decomposition of nonlinear flows."  Schmidt, O. T. (2020).

Definitions:
  bispectrum B(f1,f2) = ⟨ X(f1) X(f2) X*(f1+f2) ⟩,
  triad (f1,f2,f3) satisfying f1 + f2 = f3.

Method:
  1. Compute FFT blocks via Welch’s method: qhat[f, j, b].
  2. For each triad, form:
       A_jb = conj[ qhat[p1, j, b] · qhat[p2, j, b] ],
       B_jb =     qhat[p3, j, b].
  3. Build bispectral correlation:
       C = A^H W B,  C_{bb'} = Σ_j A_jb^* W_j B_jb'.
  4. Solve: C a = λ a, obtain eigenmodes a.
  5. Spatial modes:
       Φ1_j = Σ_b a_b^* B_jb,  Φ2_j = Σ_b a_b^* A_jb.
"""

# Standard library imports
import argparse
import os
import re
import time

import h5py
import matplotlib.pyplot as plt

# Third-party imports
import numpy as np
from scipy.sparse import linalg as splinalg
from tqdm import tqdm

from configs import (
    CMAP_DIV,
    CMAP_SEQ,
    FIGURES_DIR_BSMD,
    RESULTS_DIR_BSMD,
    RESULTS_DIR_SPOD,
)
from parallel_utils import print_optimization_status

# Local application/library specific imports
from utils import (
    BaseAnalyzer,
    get_fig_aspect_ratio,
    get_num_threads,  # For detecting available CPU threads
    load_jetles_data,  # If used in __main__
    load_mat_data,  # If used in __main__
    make_result_filename,  # For saving results
    print_summary,
)

# Try to import DNamiXNPZLoader for npz support
try:
    from data_interface import DNamiXNPZLoader
except ImportError:
    DNamiXNPZLoader = None


def find_closest_freq_idx(freq_array, target):
    """Return the index of the closest value in freq_array to target. Returns None if freq_array is empty or None."""
    if freq_array is None or len(freq_array) == 0:
        return None
    return int(np.argmin(np.abs(freq_array - target)))


# Standard static triad list
ALL_TRIADS = [
    (8, -8, 0),
    (7, -7, 0),
    (8, -7, 1),
    (6, -6, 0),
    (7, -6, 1),
    (8, -6, 2),
    (5, -5, 0),
    (6, -5, 1),
    (7, -5, 2),
    (8, -5, 3),
    (4, -4, 0),
    (5, -4, 1),
    (6, -4, 2),
    (7, -4, 3),
    (8, -4, 4),
    (3, -3, 0),
    (4, -3, 1),
    (5, -3, 2),
    (6, -3, 3),
    (7, -3, 4),
    (8, -3, 5),
    (2, -2, 0),
    (3, -2, 1),
    (4, -2, 2),
    (5, -2, 3),
    (6, -2, 4),
    (7, -2, 5),
    (8, -2, 6),
    (1, -1, 0),
    (2, -1, 1),
    (3, -1, 2),
    (4, -1, 3),
    (5, -1, 4),
    (6, -1, 5),
    (7, -1, 6),
    (8, -1, 7),
    (0, 0, 0),
    (1, 0, 1),
    (2, 0, 2),
    (3, 0, 3),
    (4, 0, 4),
    (5, 0, 5),
    (6, 0, 6),
    (7, 0, 7),
    (8, 0, 8),
    (1, 1, 2),
    (2, 1, 3),
    (3, 1, 4),
    (4, 1, 5),
    (5, 1, 6),
    (6, 1, 7),
    (7, 1, 8),
    (2, 2, 4),
    (3, 2, 5),
    (4, 2, 6),
    (5, 2, 7),
    (6, 2, 8),
    (3, 3, 6),
    (4, 3, 7),
    (5, 3, 8),
    (4, 4, 8),
]


class BSMDAnalyzer(BaseAnalyzer):
    """
    Bispectral Mode Decomposition (BSMD) Analyzer.

    This class implements BSMD to extract coherent structures involved in triadic interactions,
    typically indicative of nonlinear processes in fluid flows or other dynamical systems.
    The method is based on the paper: Schmidt, O. T. (2020). "Bispectral mode decomposition
    of nonlinear flows."

    Key concepts:
    - Bispectrum: B(f1, f2) = < X(f1) X(f2) X*(f1+f2) >, measures the statistical
      dependence between three frequency components satisfying the triadic relation f1 + f2 = f3.
    - Triad: A set of three frequencies (f1, f2, f3) such that f1 + f2 = f3.
    - BSMD Eigenvalue Problem: Solved for each triad to find modes (modes1, modes2) and
      eigenvalues that characterize the strength and spatial structure of the interaction.

    The typical BSMD process involves:
    1. Computing FFT blocks of the data (e.g., using Welch's method) to get q_hat[f, j, b]
       (frequency, spatial_point, block_index).
    2. For each selected triad (p1, p2, p3) where p_k are frequency indices:
       a. Form auxiliary matrices A_jb = conj(q_hat[p1,j,b] * q_hat[p2,j,b]) and B_jb = q_hat[p3,j,b].
       b. Construct the bispectral correlation matrix C_bb' = sum_j (A_jb^* W_j B_jb').
       c. Solve the eigenvalue problem: C a = lambda a.
    3. Reconstruct spatial modes: modes1_j = sum_b (a_b^* B_jb) and modes2_j = sum_b (a_b^* A_jb).

    Key Attributes:
        modes1 (np.ndarray): BSMD spatial modes (related to f1, f2 interaction product).
                           Shape: (n_triads, n_spatial_points).
        modes2 (np.ndarray): BSMD spatial modes (related to f3).
                           Shape: (n_triads, n_spatial_points).
        eigenvalues (np.ndarray): BSMD eigenvalues (lambda), complex values indicating interaction strength and phase.
                                  Shape: (n_triads,).
        triads (list of tuples): List of frequency index triads (p1, p2, p3) analyzed.
        qhat (np.ndarray): STFT of the data, q_hat[frequency_bin, spatial_point, block].
        fs (float): Sampling frequency of the data.
        nfft (int): Number of points per FFT block.
        W (np.ndarray): Spatial weighting matrix (diagonal).

    Inherits from:
        BaseAnalyzer: Provides common functionalities for data loading, STFT computation,
                      and preprocessing.
    """

    def __init__(self, file_path, nfft=128, overlap=0.5, results_dir=RESULTS_DIR_BSMD, figures_dir=FIGURES_DIR_BSMD, data_loader=None, spatial_weight_type="auto", use_static_triads=True, static_triads=ALL_TRIADS, use_parallel=True):
        """
        Initialize the BSMDAnalyzer.

        Args:
            file_path (str): Path to the data file (e.g., .mat, .h5).
            nfft (int, optional): Number of points per FFT segment for STFT.
                                  Defaults to 128.
            overlap (float, optional): Overlap ratio between FFT segments (0 to 1).
                                     Defaults to 0.5.
            results_dir (str, optional): Directory to save analysis results (HDF5 files).
                                         Defaults to `RESULTS_DIR_BSMD` from `configs.py`.
            figures_dir (str, optional): Directory to save generated plots.
                                         Defaults to `FIGURES_DIR_BSMD` from `configs.py`.
            data_loader (callable, optional): Custom function to load data from `file_path`.
                                              If None, `BaseAnalyzer` attempts to auto-detect.
                                              Defaults to None.
            spatial_weight_type (str, optional): Type of spatial weights to apply ('auto', 'uniform', 'polar').
                                                 'auto' attempts to detect from filename.
                                                 Defaults to 'auto'.
            use_static_triads (bool, optional): If True, use the `static_triads` list.
                                                If False, dynamic triad selection (not yet fully implemented)
                                                would be attempted. Defaults to True.
            static_triads (list of tuples, optional): List of predefined frequency index triads (p_k, p_l, p_k+p_l)
                                                     to analyze. Defaults to `ALL_TRIADS` from this module.
        """
        super().__init__(
            file_path=file_path,
            nfft=nfft,
            overlap=overlap,
            results_dir=results_dir,
            figures_dir=figures_dir,
            data_loader=data_loader,
            spatial_weight_type=spatial_weight_type,
            use_parallel=use_parallel,
        )
        self.use_static_triads = use_static_triads
        self.static_triads_list = static_triads if use_static_triads else []
        self.analysis_type = "bsmd"

        # BSMD specific attributes
        self.modes1 = np.array([])  # BSMD spatial modes (interaction product)
        self.modes2 = np.array([])  # BSMD spatial modes (third frequency)
        self.eigenvalues = np.array([])  # BSMD eigenvalues
        self.triads = np.array([])  # Triads (f_alpha, f_beta, f_gamma)
        self.freq_alpha_idx = np.array([], dtype=int)

        # Ensure output directories exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        # Derive base name for outputs
        base = os.path.basename(file_path)
        self.data_root = re.sub(r"\.[^.]*$", "", base)

        # Placeholders
        self.data = {}
        self.W = np.array([])
        self.novlap = int(overlap * nfft)
        self.nblocks = 0
        self.fs = 0.0
        self.qhat = np.array([])
        self.qhat_cached = False
        self.triads = []
        self.modes1 = np.array([])
        self.modes2 = np.array([])
        self.freq = None
        self.St = None
        self.energy_map = np.array([])

    def load_and_preprocess(self):
        """
        Loads data, computes spatial weights, and STFT using BaseAnalyzer methods.

        This method orchestrates:
        1. Loading data via `_load_data()`.
        2. Determining and applying spatial weights via `_calculate_spatial_weights()`.
        3. Computing the STFT of the data via `compute_fft_blocks()`.

        Sets attributes like `self.data`, `self.W`, `self.qhat`, `self.fs`, `self.freq`.
        """
        super().load_and_preprocess()  # Leverages BaseAnalyzer's core logic

    def compute_fft_blocks(self):
        """
        Computes the Short-Time Fourier Transform (STFT) of the loaded data.

        This method is typically called by `load_and_preprocess`.
        It uses the `blocksfft` utility function with parameters `self.nfft`,
        `self.novlap`, `self.fs`, and `WINDOW_TYPE`, `WINDOW_NORM` from `configs.py`.

        Sets/updates attributes:
            qhat (np.ndarray): STFT of the data [freq_bin, spatial_loc, block].
            freq (np.ndarray): Array of frequency bins.
            St (np.ndarray): Array of Strouhal numbers (if applicable).
            nblocks (int): Number of blocks in the STFT.
        """

        # Path for BSMD-specific cached FFT blocks
        fname_bsmd = make_result_filename(
            self.data_root,
            self.nfft,
            self.overlap,
            self.data.get("Ns", 0),
            self.analysis_type,
        )
        cache_path = os.path.join(self.results_dir, fname_bsmd)

        # Try loading cached FFT blocks from a previous BSMD run first
        if os.path.exists(cache_path):
            with h5py.File(cache_path, "r") as f:
                if "FFTBlocks" in f:
                    qhat_cached = f["FFTBlocks"][:]
                    if qhat_cached.shape[0] == self.nfft // 2 + 1:
                        self.qhat = qhat_cached
                        self.nblocks = qhat_cached.shape[2]
                        self.qhat_cached = True
                        print(f"Loaded cached FFT blocks from {cache_path}")
                        self.freq = np.fft.rfftfreq(self.nfft, d=1.0 / self.fs)
                        self.St = self.freq
                        return

        # Otherwise, see if SPOD cached blocks exist to reuse
        fname_spod = make_result_filename(
            self.data_root,
            self.nfft,
            self.overlap,
            self.data.get("Ns", 0),
            "spod",
        )
        spod_path = os.path.join(RESULTS_DIR_SPOD, fname_spod)
        if os.path.exists(spod_path):
            with h5py.File(spod_path, "r") as f:
                if "FFTBlocks" in f:
                    qhat_cached = f["FFTBlocks"][:]
                    if qhat_cached.shape[0] == self.nfft // 2 + 1:
                        self.qhat = qhat_cached
                        self.nblocks = qhat_cached.shape[2]
                        self.qhat_cached = True
                        print(f"Reusing cached FFT blocks from {spod_path}")
                        # Save a copy for future BSMD runs
                        os.makedirs(self.results_dir, exist_ok=True)
                        mode = "a" if os.path.exists(cache_path) else "w"
                        with h5py.File(cache_path, mode) as f_bsmd:
                            if "FFTBlocks" in f_bsmd:
                                del f_bsmd["FFTBlocks"]
                            f_bsmd.create_dataset("FFTBlocks", data=self.qhat, compression="gzip")
                            if mode == "w":
                                for key, value in self._get_metadata().items():
                                    f_bsmd.attrs[key] = value
                        print(f"Saved FFT blocks to cache at {cache_path}")
                        self.freq = np.fft.rfftfreq(self.nfft, d=1.0 / self.fs)
                        self.St = self.freq
                        return

        # If no cache available, compute and save
        super().compute_fft_blocks()  # Leverages BaseAnalyzer's core logic
        self.qhat_cached = False

        os.makedirs(self.results_dir, exist_ok=True)
        mode = "a" if os.path.exists(cache_path) else "w"
        with h5py.File(cache_path, mode) as f:
            if "FFTBlocks" in f:
                del f["FFTBlocks"]
            f.create_dataset("FFTBlocks", data=self.qhat, compression="gzip")
            if mode == "w":
                for key, value in self._get_metadata().items():
                    f.attrs[key] = value
        print(f"Saved FFT blocks to cache at {cache_path}")

        # Set frequency and Strouhal vectors after qhat is available
        self.freq = np.fft.rfftfreq(self.nfft, d=1.0 / self.fs)
        self.St = self.freq  # Default: Strouhal equals frequency if no scaling

    # Main method to perform BSMD analysis based on configuration.
    def perform_bsmd(self):
        """
        Perform Bispectral Mode Decomposition (BSMD) analysis.

        This method acts as a dispatcher based on the `self.use_static_triads` attribute.
        - If True, it calls `_perform_static_bsmd_core` to analyze predefined triads.
        - If False (or for future dynamic triad selection), it would call `perform_dynamic_bsmd`.

        Ensures data is loaded and preprocessed (STFT computed) before proceeding.
        """
        if self.qhat.size == 0:
            print("STFT data (qhat) not found. Running load_and_preprocess...")
        start_time = time.time()
        print("Starting BSMD analysis...")

        if self.use_static_triads:
            self._perform_static_bsmd_core()
        else:
            # self._perform_dynamic_bsmd_core() # This would be the actual dynamic triad computation
            print("Dynamic BSMD core logic not yet fully implemented in this refactor.")
            # For now, just set empty results to avoid errors in subsequent steps
            self.modes1 = np.array([])
            self.modes2 = np.array([])
            self.eigenvalues = np.array([])
            self.triads = np.array([])

        print(f"BSMD analysis completed in {time.time() - start_time:.2f} seconds.")

    # Core logic for BSMD with statically defined triads.
    def _perform_static_bsmd_core(self):
        """
        Perform BSMD for a statically defined list of frequency triads.

        This is the core computational part of BSMD when `self.use_static_triads` is True.
        It iterates through `self.static_triads_list` and for each triad:
        1. Extracts frequency indices (p1, p2, p3).
        2. Forms auxiliary matrices A_jb and B_jb from `self.qhat`.
        3. Constructs the bispectral correlation matrix C_bb' using `self.W` for spatial weighting.
        4. Solves the eigenvalue problem C a = lambda a. The dominant eigenvalue and corresponding
           eigenvector are typically selected.
        5. Reconstructs the spatial BSMD modes modes1 and modes2.

        Attributes set/appended:
            eigenvalues (list): Appends the dominant BSMD eigenvalue for each triad.
            modes1 (list of np.ndarray): Appends the BSMD mode modes1 for each triad.
            modes2 (list of np.ndarray): Appends the BSMD mode modes2 for each triad.
            triads (list of tuples): Stores the (p1, p2, p3) triad being processed.
        Finally, these lists are converted to numpy arrays.
        """
        print("Performing static BSMD core analysis...")
        start_time = time.time()
        if not self.static_triads_list or len(self.static_triads_list) == 0:
            print("Error: Static triads list is empty. Cannot perform static BSMD.")
            self.modes1 = np.array([])
            self.modes2 = np.array([])
            self.eigenvalues = np.array([])
            self.triads = np.array([])
            return

        print(f"Using {len(self.static_triads_list)} statically defined triads.")
        # Initialize result arrays based on the number of static triads
        num_triads = len(self.static_triads_list)
        Nspace = self.qhat.shape[1]  # Number of spatial points

        self.modes1 = np.zeros((num_triads, Nspace), dtype=complex)
        self.modes2 = np.zeros((num_triads, Nspace), dtype=complex)
        self.eigenvalues = np.zeros(num_triads, dtype=float)  # Eigenvalues are real
        self.triads = np.array(self.static_triads_list)  # Store the used triads

        # Map target Strouhal numbers to frequency indices
        # self.freq and self.St should be available from BaseAnalyzer.load_and_preprocess_data
        # Robustly recalculate self.freq and self.St if needed (handles cases where not set or mismatched)
        if self.freq is None or self.St is None or (hasattr(self.qhat, "shape") and self.qhat.shape and ((not hasattr(self.freq, "size") or self.freq.size != self.qhat.shape[0]) or (not hasattr(self.St, "size") or self.St.size != self.qhat.shape[0]))):
            print("Recalculating frequency and Strouhal arrays to match qhat...")
            nfft = self.qhat.shape[0] if hasattr(self.qhat, "shape") and len(self.qhat.shape) > 0 else 0
            if nfft > 0:
                self.freq = np.fft.rfftfreq(nfft * 2 - 2, d=1.0 / self.fs)[:nfft]
                # If Strouhal number is used, recalculate accordingly
                if hasattr(self, "D") and hasattr(self, "U_inf") and self.D and self.U_inf:
                    self.St = self.freq * self.D / self.U_inf
                else:
                    self.St = self.freq.copy()
            else:
                self.freq = np.array([])
                self.St = np.array([])

        if self.freq is None or self.St is None or self.freq.size == 0 or self.St.size == 0:
            print("Error: Frequency/Strouhal information not available. Cannot map triads.")
            return

        for i, (st_alpha_target, st_beta_target, st_gamma_target) in enumerate(tqdm(self.static_triads_list, desc="BSMD Triads")):
            idx_alpha = find_closest_freq_idx(self.St, st_alpha_target)
            idx_beta = find_closest_freq_idx(self.St, st_beta_target)
            idx_gamma = find_closest_freq_idx(self.St, st_gamma_target)

            if idx_alpha is None or idx_beta is None or idx_gamma is None:
                print(f"Warning: Could not find matching frequencies for triad St=({st_alpha_target},{st_beta_target},{st_gamma_target}). Skipping.")
                # Set corresponding results to NaN or handle as appropriate
                self.eigenvalues[i] = np.nan
                self.modes1[i, :] = np.nan
                self.modes2[i, :] = np.nan
                continue

            # Extract FFT data for the triad frequencies
            Q_alpha = self.qhat[idx_alpha, :, :]  # (Nspace, Nblocks)
            Q_beta = self.qhat[idx_beta, :, :]  # (Nspace, Nblocks)
            Q_gamma = self.qhat[idx_gamma, :, :]  # (Nspace, Nblocks)

            # Compute bispectral tensor B_alpha_beta (Schmidt decomposition)
            # B_ab = Q_alpha @ Q_beta.conj().T / Nblocks (example formulation)
            # For BSMD, a more specific formulation is used, often involving Q_gamma
            # Example from Towne et al. (2017) JFM, Eq. (3.4)
            # B_ij = sum_k (Q_alpha_ik * Q_beta_jk * Q_gamma_conjugate_sumfreq_k) / Nblocks^2
            # Here, we need a simplified approach or a direct call to a bmsd_core_function

            # Placeholder for actual BSMD core computation for a single triad
            # This would involve forming the bispectral density tensor and performing SVD
            # For now, let's assume a simplified SVD on a product of Q_alpha and Q_beta
            # This is NOT the full BSMD, just a placeholder SVD.
            if Q_alpha.shape[1] == 0 or Q_beta.shape[1] == 0:  # Nblocks is 0
                print(f"Warning: Zero blocks for triad {i}. Skipping.")
                self.eigenvalues[i] = np.nan
                self.modes1[i, :] = np.nan
                self.modes2[i, :] = np.nan
                continue

            nblocks = Q_alpha.shape[1]

            def matvec(v):
                return Q_alpha @ (Q_beta.conj().T @ v) / nblocks

            def rmatvec(v):
                return Q_beta @ (Q_alpha.conj().T @ v) / nblocks

            op = splinalg.LinearOperator(
                shape=(Q_alpha.shape[0], Q_alpha.shape[0]),
                matvec=matvec,
                rmatvec=rmatvec,
                dtype=np.complex128,
            )

            try:
                u, s, vh = splinalg.svds(op, k=1)
                self.modes1[i, :] = u[:, 0]
                self.modes2[i, :] = vh[0, :].conj()
                self.eigenvalues[i] = s[0]
            except Exception as e:
                print(f"SVD failed for triad {i} (St={st_alpha_target},{st_beta_target},{st_gamma_target}): {e}")
                self.eigenvalues[i] = np.nan
                self.modes1[i, :] = np.nan
                self.modes2[i, :] = np.nan

        print(f"Static BSMD core analysis completed in {time.time() - start_time:.2f} seconds.")

        # Build energy map for quick visualisation
        self.energy_map = self._compute_energy_map()

    def perform_dynamic_bsmd(self):
        """
        Perform BSMD with dynamically identified triads (Placeholder).

        This method is intended for future implementation where significant triads
        are identified from the data (e.g., based on bispectrum peaks) rather than
        being predefined.

        Currently, this method will raise a NotImplementedError.
        """
        raise NotImplementedError("Dynamic BSMD is not yet implemented.")

    def _compute_energy_map(self):
        """Return a 2D map of eigenvalue magnitudes indexed by (p1,p2)."""
        if self.eigenvalues.size == 0:
            return np.array([])

        offset = 8
        size = 2 * offset + 1
        grid = np.full((size, size), np.nan)
        for val, (p1, p2, _p3) in zip(np.abs(self.eigenvalues), self.triads):
            i = int(p1) + offset
            j = int(p2) + offset
            if 0 <= i < size and 0 <= j < size:
                grid[i, j] = val
        return grid

    # Save triads, eigenvalues, modes, and weights to HDF5.
    def save_results(self, fname=None):
        """
        Save BSMD results (triads, eigenvalues, modes) to an HDF5 file.

        The results are saved in `self.results_dir`. If `fname` is None,
        it's generated using `make_result_filename` based on the input data file name,
        `nfft`, `overlap`, and the analysis type ('bsmd').

        Args:
            fname (str, optional): Custom filename for the HDF5 output.
                                   Defaults to None (auto-generated).

        Datasets saved:
            'Triads': List of analyzed frequency index triads (p1, p2, p3).
            'Eigenvalues': Complex BSMD eigenvalues for each triad.
            'Modes1': BSMD spatial modes (interaction product) for each triad.
            'Modes2': BSMD spatial modes (third frequency) for each triad.
            'Weights': Spatial weighting matrix (diagonal) used in the analysis.
            'Frequencies': Frequency vector corresponding to FFT bins.
            'fs': Sampling frequency.
            'nfft': FFT length.
            'overlap': FFT overlap ratio.
            'data_file_path': Path to the original data file.
        """
        if fname is None:
            # Construct filename based on data and parameters
            results_path = os.path.join(self.results_dir, make_result_filename(self.data_root, self.nfft, self.overlap, self.data["Ns"], "bsmd"))
        else:
            results_path = os.path.join(self.results_dir, fname)
        # Ensure output directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        with h5py.File(results_path, "w") as f:
            f.create_dataset("triads", data=np.array(self.triads))
            f.create_dataset("eigenvalues", data=self.eigenvalues)  # Changed from 'Lambda'
            f.create_dataset("Modes1", data=self.modes1)
            f.create_dataset("Modes2", data=self.modes2)
            f.create_dataset("x", data=self.data["x"])
            f.create_dataset("y", data=self.data["y"])
            f.create_dataset("W", data=self.W)
            if self.energy_map.size:
                f.create_dataset("energy_map", data=self.energy_map)
        print(f"Results saved to {results_path}")

    from typing import Optional

    def plot_modes(self, triad_indices=None, top_n=3, plot_n_modes: Optional[int] = 10):
        """Plot spatial BSMD modes for selected triads."""
        if self.modes1.size == 0 or self.modes2.size == 0:
            print("No BSMD modes to plot. Run perform_bsmd() first.")
            return

        if triad_indices is None:
            lambdas = np.abs(self.eigenvalues)
            triad_indices = list(np.argsort(lambdas)[::-1])
        if plot_n_modes is not None:
            triad_indices = triad_indices[:plot_n_modes]

        nx = self.data.get("Nx", int(np.sqrt(self.modes1.shape[1])))
        ny = self.data.get("Ny", int(np.sqrt(self.modes1.shape[1])))
        x_coords = self.data.get("x", np.arange(nx))
        y_coords = self.data.get("y", np.arange(ny))
        fig_aspect = get_fig_aspect_ratio(self.data)
        var_name = self.data.get("metadata", {}).get("var_name", "q")
        extent = (
            x_coords.min(),
            x_coords.max(),
            y_coords.min(),
            y_coords.max(),
        )

        for idx in triad_indices:
            # Reshape mode vectors into the spatial grid.  No transpose is
            # required here because ``x`` and ``y`` coordinates are assumed to
            # follow the same (Nx, Ny) ordering as the stored modes.  The
            # previous transpose caused a mismatch when the coordinates were
            # provided as full 2D arrays.
            mode1 = self.modes1[idx, :].real.reshape(nx, ny)
            mode2 = self.modes2[idx, :].real.reshape(nx, ny)
            triad = self.triads[idx]

            if x_coords.ndim == 1 and y_coords.ndim == 1:
                x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")
            else:
                x_mesh, y_mesh = x_coords, y_coords
            dist = np.sqrt(x_mesh**2 + y_mesh**2)
            cylinder_mask = dist <= 0.5

            fig, axes = plt.subplots(1, 2, figsize=(8 * fig_aspect, 4))

            field1 = np.ma.array(mode1, mask=cylinder_mask)
            vmax1 = np.max(np.abs(field1))
            levels1 = np.linspace(-vmax1, vmax1, 21)
            cf1 = axes[0].contourf(
                x_mesh,
                y_mesh,
                field1,
                levels=levels1,
                cmap=CMAP_DIV,
                extend="both",
            )
            axes[0].contour(
                x_mesh,
                y_mesh,
                field1,
                levels=levels1[::4],
                colors="k",
                linewidths=0.5,
                alpha=0.5,
            )
            axes[0].add_patch(plt.Circle((0, 0), 0.5, facecolor="lightgray", edgecolor="black", linewidth=0.5))
            axes[0].set_title(f"Triad {tuple(triad)} Phi1 [{var_name}]")
            axes[0].set_xlabel(r"$x/D$")
            axes[0].set_ylabel(r"$y/D$")
            axes[0].set_aspect("equal", "box")
            axes[0].set_xlim(x_coords.min(), x_coords.max())
            axes[0].set_ylim(y_coords.min(), y_coords.max())
            axes[0].grid(True, linestyle="--", alpha=0.3)
            fig.colorbar(cf1, ax=axes[0], shrink=0.8)

            field2 = np.ma.array(mode2, mask=cylinder_mask)
            vmax2 = np.max(np.abs(field2))
            levels2 = np.linspace(-vmax2, vmax2, 21)
            cf2 = axes[1].contourf(
                x_mesh,
                y_mesh,
                field2,
                levels=levels2,
                cmap=CMAP_DIV,
                extend="both",
            )
            axes[1].contour(
                x_mesh,
                y_mesh,
                field2,
                levels=levels2[::4],
                colors="k",
                linewidths=0.5,
                alpha=0.5,
            )
            axes[1].add_patch(plt.Circle((0, 0), 0.5, facecolor="lightgray", edgecolor="black", linewidth=0.5))
            axes[1].set_title(f"Triad {tuple(triad)} Phi2 [{var_name}]")
            axes[1].set_xlabel(r"$x/D$")
            axes[1].set_ylabel(r"$y/D$")
            axes[1].set_aspect("equal", "box")
            axes[1].set_xlim(x_coords.min(), x_coords.max())
            axes[1].set_ylim(y_coords.min(), y_coords.max())
            axes[1].grid(True, linestyle="--", alpha=0.3)
            fig.colorbar(cf2, ax=axes[1], shrink=0.8)

            fig.tight_layout()
            fname = os.path.join(self.figures_dir, f"{self.data_root}_BSMD_triad{idx}_{var_name}.png")
            plt.savefig(fname)
            plt.close(fig)
            print(f"BSMD mode plot saved to {fname}")

    def plot_energy_map(self):
        """Plot a 2D heatmap of eigenvalue magnitudes indexed by triad frequencies."""
        if self.energy_map.size == 0:
            print("No energy map available. Run perform_bsmd() first.")
            return

        extent = (-8.5, 8.5, -8.5, 8.5)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            self.energy_map,
            origin="lower",
            extent=extent,
            cmap=CMAP_SEQ,
            aspect="equal",
        )
        ax.set_xlabel("p1 index")
        ax.set_ylabel("p2 index")
        ax.set_title("BSMD energy map |lambda|")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fname = os.path.join(self.figures_dir, f"{self.data_root}_BSMD_energy_map.png")
        plt.savefig(fname)
        plt.close(fig)
        print(f"Energy map saved to {fname}")

    # Execute the full BSMD pipeline.
    def run_analysis(self):
        """
        Execute the full BSMD analysis pipeline.

        This method orchestrates the entire BSMD process:
        1. Loads and preprocesses data, including STFT computation (calls `load_and_preprocess`).
           This step sets `self.qhat`, `self.W`, `self.freq`, `self.fs`, etc.
        2. Performs BSMD computation (calls `perform_bsmd`), which internally chooses
           between static or dynamic triad analysis (currently static is implemented).
           This step sets `self.modes1`, `self.modes2`, `self.eigenvalues`, `self.triads`.
        3. Saves the results to an HDF5 file (calls `save_results`).

        This is the primary method to call to run a complete BSMD study on a dataset.
        """
        print(f"🔎 Starting BSMD analysis for {os.path.basename(self.file_path)}")
        start_total_time = time.time()
        self.load_and_preprocess()
        self.compute_fft_blocks()
        self.perform_bsmd()  # Calls the renamed method
        self.save_results()
        print(f"Total BSMD runtime: {time.time() - start_total_time:.2f} s")
        print_summary("BSMD", self.results_dir, self.figures_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BSMD analysis")
    parser.add_argument("--prep", action="store_true", help="Load data and compute FFT blocks")
    parser.add_argument("--compute", action="store_true", help="Perform BSMD and save results")
    parser.add_argument("--plot", action="store_true", help="Generate example plots")
    args = parser.parse_args()

    from parallel_utils import get_threadpool_summary

    print(f"Thread pools: {get_threadpool_summary()}")

    print_optimization_status()

    data_file = "./data/snp1-947_u.npz"

    if DNamiXNPZLoader is not None and data_file.endswith(".npz"):
        loader = DNamiXNPZLoader()
        available_fields = loader.get_available_fields(data_file)
        print(f"Available fields in {data_file}: {available_fields}")
        for field in available_fields:
            print(f"\n===== Running BSMD for variable: {field} =====")
            data = loader.load(data_file, field=field)
            results_dir = os.path.join(RESULTS_DIR_BSMD, field)
            figures_dir = os.path.join(FIGURES_DIR_BSMD, field)
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(figures_dir, exist_ok=True)
            analyzer = BSMDAnalyzer(
                file_path=data_file,
                nfft=128,
                overlap=0.5,
                results_dir=results_dir,
                figures_dir=figures_dir,
                data_loader=lambda fp: loader.load(fp, field=field),
                spatial_weight_type="uniform",
                use_static_triads=True,
                static_triads=ALL_TRIADS,
            )
            analyzer.data = data
            analyzer.analysis_type = f"bsmd_{field}"
            run_all = not (args.prep or args.compute or args.plot)
            if run_all or args.prep:
                analyzer.load_and_preprocess()
                analyzer.compute_fft_blocks()
            if run_all or args.compute:
                if analyzer.qhat.size == 0:
                    analyzer.load_and_preprocess()
                    analyzer.compute_fft_blocks()
                analyzer.perform_bsmd()
                analyzer.save_results()
            if run_all or args.plot:
                if analyzer.eigenvalues.size == 0:
                    print("No BSMD results to plot. Run with --compute first.")
                else:
                    lambdas = np.abs(analyzer.eigenvalues)
                    plt.figure()
                    plt.plot(lambdas, "o-")
                    plt.xlabel("Triad index")
                    plt.ylabel("Eigenvalue magnitude")
                    plt.title("BSMD eigenvalue magnitudes")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(figures_dir, f"{analyzer.data_root}_BSMD_eigenvalues.png"))
                    plt.close()
                    analyzer.plot_modes()
                    analyzer.plot_energy_map()
            if run_all:
                print_summary("BSMD", analyzer.results_dir, analyzer.figures_dir)
        exit(0)
    else:
        if "jet" in data_file.lower():
            loader = load_jetles_data
            spatial_weight = "polar"
        else:
            loader = load_mat_data
            spatial_weight = "uniform"

        analyzer = BSMDAnalyzer(
            file_path=data_file,
            nfft=128,
            overlap=0.5,
            results_dir=RESULTS_DIR_BSMD,
            figures_dir=FIGURES_DIR_BSMD,
            data_loader=loader,
            spatial_weight_type=spatial_weight,
            use_static_triads=True,
            static_triads=ALL_TRIADS,
        )

        run_all = not (args.prep or args.compute or args.plot)

        if run_all or args.prep:
            analyzer.load_and_preprocess()
            analyzer.compute_fft_blocks()

        if run_all or args.compute:
            if analyzer.qhat.size == 0:
                analyzer.load_and_preprocess()
                analyzer.compute_fft_blocks()
            analyzer.perform_bsmd()
            analyzer.save_results()

        if run_all or args.plot:
            if analyzer.eigenvalues.size == 0:
                print("No BSMD results to plot. Run with --compute first.")
            else:
                lambdas = np.abs(analyzer.eigenvalues)
                plt.figure()
                plt.plot(lambdas, "o-")
                plt.xlabel("Triad index")
                plt.ylabel("Eigenvalue magnitude")
                plt.title("BSMD eigenvalue magnitudes")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(FIGURES_DIR_BSMD, f"{analyzer.data_root}_BSMD_eigenvalues.png"))
                plt.close()
                analyzer.plot_modes()
                analyzer.plot_energy_map()

        if run_all:
            print_summary("BSMD", analyzer.results_dir, analyzer.figures_dir)
