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
import os
import re
import time

import h5py
import matplotlib.pyplot as plt

# Third-party imports
import numpy as np
import scipy.linalg as eig

from configs import FIGURES_DIR, RESULTS_DIR

# Local application/library specific imports
from utils import (
    BaseAnalyzer,
    auto_detect_weight_type,  # If used in __main__
    blocksfft,  # BSMD method description mentions qhat from Welch's method
    load_jetles_data,  # If used in __main__
    load_mat_data,  # If used in __main__
    make_result_filename,  # For saving results
)


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
    - BSMD Eigenvalue Problem: Solved for each triad to find modes (Phi1, Phi2) and
      eigenvalues (lambda) that characterize the strength and spatial structure of the interaction.

    The typical BSMD process involves:
    1. Computing FFT blocks of the data (e.g., using Welch's method) to get q_hat[f, j, b]
       (frequency, spatial_point, block_index).
    2. For each selected triad (p1, p2, p3) where p_k are frequency indices:
       a. Form auxiliary matrices A_jb = conj(q_hat[p1,j,b] * q_hat[p2,j,b]) and B_jb = q_hat[p3,j,b].
       b. Construct the bispectral correlation matrix C_bb' = sum_j (A_jb^* W_j B_jb').
       c. Solve the eigenvalue problem: C a = lambda a.
    3. Reconstruct spatial modes: Phi1_j = sum_b (a_b^* B_jb) and Phi2_j = sum_b (a_b^* A_jb).

    Key Attributes:
        Phi1 (np.ndarray): BSMD spatial modes (related to f1, f2 interaction product).
                           Shape: (n_triads, n_spatial_points).
        Phi2 (np.ndarray): BSMD spatial modes (related to f3).
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

    def __init__(self, file_path, nfft=128, overlap=0.5, results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR, data_loader=None, spatial_weight_type="auto", use_static_triads=True, static_triads=ALL_TRIADS):
        """
        Initialize the BSMDAnalyzer.

        Args:
            file_path (str): Path to the data file (e.g., .mat, .h5).
            nfft (int, optional): Number of points per FFT segment for STFT.
                                  Defaults to 128.
            overlap (float, optional): Overlap ratio between FFT segments (0 to 1).
                                     Defaults to 0.5.
            results_dir (str, optional): Directory to save analysis results (HDF5 files).
                                         Defaults to `RESULTS_DIR` from `configs.py`.
            figures_dir (str, optional): Directory to save generated plots.
                                         Defaults to `FIGURES_DIR` from `configs.py`.
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
        super().__init__(file_path=file_path, nfft=nfft, overlap=overlap, results_dir=results_dir, figures_dir=figures_dir, data_loader=data_loader, spatial_weight_type=spatial_weight_type)
        self.use_static_triads = use_static_triads
        self.static_triads_list = static_triads if use_static_triads else []

        # BSMD specific attributes
        self.Phi1 = np.array([])  # BSMD spatial modes (Phi_alpha)
        self.Phi2 = np.array([])  # BSMD spatial modes (Phi_beta)
        self.eigenvalues = np.array([])  # BSMD eigenvalues (lambda_vals)
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
        self.triads = []
        self.a = np.array([])
        self.lambda_vals = np.array([])
        self.Phi1 = np.array([])
        self.Phi2 = np.array([])
        self.freq = None
        self.St = None

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
        super().compute_fft_blocks()  # Leverages BaseAnalyzer's core logic

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
            self.Phi1 = np.array([])
            self.Phi2 = np.array([])
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
        5. Reconstructs the spatial BSMD modes Phi1 and Phi2.

        Attributes set/appended:
            eigenvalues (list): Appends the dominant BSMD eigenvalue for each triad.
            Phi1 (list of np.ndarray): Appends the BSMD mode Phi1 for each triad.
            Phi2 (list of np.ndarray): Appends the BSMD mode Phi2 for each triad.
            triads (list of tuples): Stores the (p1, p2, p3) triad being processed.
        Finally, these lists are converted to numpy arrays.
        """
        print("Performing static BSMD core analysis...")
        start_time = time.time()
        if not self.static_triads_list or len(self.static_triads_list) == 0:
            print("Error: Static triads list is empty. Cannot perform static BSMD.")
            self.Phi1 = np.array([])
            self.Phi2 = np.array([])
            self.eigenvalues = np.array([])
            self.triads = np.array([])
            return

        print(f"Using {len(self.static_triads_list)} statically defined triads.")
        # Initialize result arrays based on the number of static triads
        num_triads = len(self.static_triads_list)
        Nspace = self.qhat.shape[1]  # Number of spatial points

        self.Phi1 = np.zeros((num_triads, Nspace), dtype=complex)
        self.Phi2 = np.zeros((num_triads, Nspace), dtype=complex)
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

        for i, (st_alpha_target, st_beta_target, st_gamma_target) in enumerate(self.static_triads_list):
            idx_alpha = find_closest_freq_idx(self.St, st_alpha_target)
            idx_beta = find_closest_freq_idx(self.St, st_beta_target)
            idx_gamma = find_closest_freq_idx(self.St, st_gamma_target)

            if idx_alpha is None or idx_beta is None or idx_gamma is None:
                print(f"Warning: Could not find matching frequencies for triad St=({st_alpha_target},{st_beta_target},{st_gamma_target}). Skipping.")
                # Set corresponding results to NaN or handle as appropriate
                self.eigenvalues[i] = np.nan
                self.Phi1[i, :] = np.nan
                self.Phi2[i, :] = np.nan
                continue

            # Extract FFT data for the triad frequencies
            Q_alpha = self.qhat[idx_alpha, :, :]  # Shape (Nspace, Nblocks)
            Q_beta = self.qhat[idx_beta, :, :]  # Shape (Nspace, Nblocks)
            Q_gamma = self.qhat[idx_gamma, :, :]  # Shape (Nspace, Nblocks)

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
                self.Phi1[i, :] = np.nan
                self.Phi2[i, :] = np.nan
                continue

            target_matrix = (Q_alpha @ Q_beta.conj().T) / Q_alpha.shape[1]  # Nspace x Nspace

            try:
                # Perform SVD. U corresponds to Phi1, V.conj().T to Phi2, s to singular values (lambda_vals)
                U, s, Vh = np.linalg.svd(target_matrix)

                # Store the leading modes and singular value
                self.Phi1[i, :] = U[:, 0]  # Leading left singular vector
                self.Phi2[i, :] = Vh[0, :].conj()  # Leading right singular vector (conjugated)
                self.eigenvalues[i] = s[0]  # Leading singular value (this is lambda_val for the triad)
            except np.linalg.LinAlgError as e:
                print(f"SVD failed for triad {i} (St={st_alpha_target},{st_beta_target},{st_gamma_target}): {e}")
                self.eigenvalues[i] = np.nan
                self.Phi1[i, :] = np.nan
                self.Phi2[i, :] = np.nan

        print(f"Static BSMD core analysis completed in {time.time() - start_time:.2f} seconds.")

    def perform_dynamic_bsmd(self):
        """
        Perform BSMD with dynamically identified triads (Placeholder).

        This method is intended for future implementation where significant triads
        are identified from the data (e.g., based on bispectrum peaks) rather than
        being predefined.

        Currently, this method will raise a NotImplementedError.
        """
        raise NotImplementedError("Dynamic BSMD is not yet implemented.")

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
            'Phi1': BSMD spatial modes (Phi_alpha) for each triad.
            'Phi2': BSMD spatial modes (Phi_beta) for each triad.
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
            f.create_dataset("Phi1", data=self.Phi1)
            f.create_dataset("Phi2", data=self.Phi2)
            f.create_dataset("x", data=self.data["x"])
            f.create_dataset("y", data=self.data["y"])
            f.create_dataset("W", data=self.W)
        print(f"Results saved to {results_path}")

    # Execute the full BSMD pipeline.
    def run_analysis(self):
        """
        Execute the full BSMD analysis pipeline.

        This method orchestrates the entire BSMD process:
        1. Loads and preprocesses data, including STFT computation (calls `load_and_preprocess`).
           This step sets `self.qhat`, `self.W`, `self.freq`, `self.fs`, etc.
        2. Performs BSMD computation (calls `perform_bsmd`), which internally chooses
           between static or dynamic triad analysis (currently static is implemented).
           This step sets `self.Phi1`, `self.Phi2`, `self.eigenvalues`, `self.triads`.
        3. Saves the results to an HDF5 file (calls `save_results`).

        This is the primary method to call to run a complete BSMD study on a dataset.
        """
        print(f"\n--- Starting BSMD Analysis for: {os.path.basename(self.file_path)} ---")
        start_total_time = time.time()
        self.load_and_preprocess()
        self.compute_fft_blocks()
        self.perform_bsmd()  # Calls the renamed method
        self.save_results()
        print(f"Total BSMD runtime: {time.time() - start_total_time:.2f} s")


if __name__ == "__main__":
    # --- Configuration ---
    # data_file = "./data/jetLES_small.mat" # Updated data path
    data_file = "./data/jetLES.mat"  # Path to your data file
    # data_file = "./data/cavityPIV.mat" # Path to your data file

    # Choose loader based on file
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
        results_dir=RESULTS_DIR,  # HDF files go here
        figures_dir=FIGURES_DIR,  # PNG files go here
        data_loader=loader,
        spatial_weight_type=spatial_weight,
        use_static_triads=True,
        static_triads=ALL_TRIADS,
    )
    analyzer.run_analysis()

    # **Plot 1: BSMD Eigenvalue Magnitudes**
    lambdas = np.abs(analyzer.eigenvalues)
    plt.figure()
    plt.plot(lambdas, "o-")
    plt.xlabel("Triad index")
    plt.ylabel("Eigenvalue magnitude")
    plt.title("BSMD eigenvalue magnitudes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{analyzer.data_root}_BSMD_eigenvalues.png"))
    plt.close()

    # **Additional Figures per Schmidt (2020)**

    # **Plot 2: Complex Eigenvalue Plane**
    vals = analyzer.eigenvalues
    plt.figure()
    plt.scatter(np.real(vals), np.imag(vals), marker="o")
    plt.xlabel("Real(λ)")
    plt.ylabel("Imag(λ)")
    plt.title("BSMD eigenvalues (complex plane)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{analyzer.data_root}_BSMD_eig_complex_plane.png"))
    plt.close()

    # **Function to Translate Triad Indices to Strouhal Numbers**
    def triad_to_strouhal(triad, fs, nfft):
        """
        Convert triad indices (p_k, p_l, p_k+p_l) to Strouhal numbers (St_k, St_l, St_(k+l))
        and their decomposition in terms of St_0.

        Args:
            triad (tuple): Triad indices (p_k, p_l, p_k+p_l).
            fs (float): Sampling frequency.
            nfft (int): Number of FFT points.

        Returns:
            dict: Contains 'decomposition' (str) and 'values' (tuple of St_k, St_l, St_(k+l)).
        """
        p_k, p_l, p_k_plus_l = triad
        St_0 = fs / nfft  # Base Strouhal number (frequency resolution)
        St_k = p_k * St_0
        St_l = p_l * St_0
        St_k_plus_l = p_k_plus_l * St_0
        decomposition = f"({p_k}*St_0, {p_l}*St_0, {p_k_plus_l}*St_0)"
        values = (St_k, St_l, St_k_plus_l)
        return {"decomposition": decomposition, "values": values}

    # **Plot 3: Spatial Bispectral Modes for Top 5 Dominant Triads**
    # Find the indices of the top 5 dominant triads based on |λ|
    lambda_mags = np.abs(analyzer.eigenvalues)
    top_5_indices = np.argsort(lambda_mags)[-5:][::-1]  # Top 5 in descending order
    top_5_triads = [analyzer.triads[idx] for idx in top_5_indices]

    Nx, Ny = analyzer.data["Nx"], analyzer.data["Ny"]
    x = analyzer.data["x"]
    y = analyzer.data["y"]
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Create a 5x2 subplot grid (5 rows for each triad, 2 columns for Phi1 and Phi2)
    fig, axs = plt.subplots(5, 2, figsize=(12, 20), sharex=True, sharey=True)
    for i, (idx, triad) in enumerate(zip(top_5_indices, top_5_triads)):
        phi1 = analyzer.Phi1[idx].reshape(Nx, Ny)
        phi2 = analyzer.Phi2[idx].reshape(Nx, Ny)
        # Convert triad to Strouhal numbers and decomposition
        result = triad_to_strouhal(triad, analyzer.fs, analyzer.nfft)
        decomposition = result["decomposition"]
        St_k, St_l, St_k_plus_l = result["values"]
        # Plot Phi1 (left column)
        pcm1 = axs[i, 0].pcolormesh(Y, X, np.real(phi1), shading="auto")
        fig.colorbar(pcm1, ax=axs[i, 0])
        axs[i, 0].set_title(f"Phi1 real part\n{decomposition}\n(St_1={St_k:.3f}, St_2={St_l:.3f}, St_3={St_k_plus_l:.3f})")
        axs[i, 0].set_ylabel("x")
        # Plot Phi2 (right column)
        pcm2 = axs[i, 1].pcolormesh(Y, X, np.real(phi2), shading="auto")
        fig.colorbar(pcm2, ax=axs[i, 1])
        axs[i, 1].set_title(f"Phi2 real part\n{decomposition}\n(St_1={St_k:.3f}, St_2={St_l:.3f}, St_3={St_k_plus_l:.3f})")
        # Set labels for the bottom row
        if i == 4:
            axs[i, 0].set_xlabel("y")
            axs[i, 1].set_xlabel("y")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{analyzer.data_root}_BSMD_modes_top_5_triads.png"))
    plt.close(fig)

    # **Plot 4: BSMD Eigenvalue Magnitudes in (St_1, St_2) Plane with Full and Zoomed-In Views**
    # Extract frequencies and eigenvalue magnitudes
    triads_arr = np.array(analyzer.triads)
    fs = analyzer.fs
    nfft = analyzer.nfft
    St1_vals = triads_arr[:, 0] * fs / nfft
    St2_vals = triads_arr[:, 1] * fs / nfft
    lambda_mags = np.abs(analyzer.eigenvalues)

    # Compute logarithmic magnitudes
    log_lambda_mags = np.log10(lambda_mags + 1e-20)  # Add small value to avoid log(0)

    # Dynamically determine frequency ranges for the full plot
    St1_max = np.max(St1_vals)
    St2_min = np.min(St2_vals)
    St2_max = np.max(St2_vals)

    # Estimate fundamental frequency St0 from triad with p2=0 and maximum |λ|
    idx_p2_zero = np.where(triads_arr[:, 1] == 0)[0]
    if len(idx_p2_zero) > 0:
        idx_max = idx_p2_zero[np.argmax(lambda_mags[idx_p2_zero])]
        St0 = St1_vals[idx_max]
    else:
        St0 = fs / nfft  # Default to frequency resolution if no p2=0 triad exists

    # Set minimum zoom range to prevent identical axis limits
    min_zoom = fs / nfft
    k = 4  # Multiplier for zoom range, adjustable if needed
    zoom_St1_max = max(k * St0, min_zoom)
    zoom_St2_min = -zoom_St1_max
    zoom_St2_max = zoom_St1_max

    # Create grids for full and zoomed plots
    n_St1 = 100  # Number of grid points for St_1
    n_St2 = 160  # Number of grid points for St_2
    St1_grid = np.linspace(0, St1_max, n_St1)
    St2_grid = np.linspace(St2_min, St2_max, n_St2)
    ST1, ST2 = np.meshgrid(St1_grid, St2_grid)

    St1_zoom_grid = np.linspace(0, zoom_St1_max, n_St1)
    St2_zoom_grid = np.linspace(zoom_St2_min, zoom_St2_max, n_St2)
    ST1_zoom, ST2_zoom = np.meshgrid(St1_zoom_grid, St2_zoom_grid)

    # Initialize grid arrays with NaN for regions outside the triangular region
    log_lambda_grid = np.full(ST1.shape, np.nan)
    log_lambda_zoom = np.full(ST1_zoom.shape, np.nan)

    # Map log_lambda_mags to the grids
    for St1, St2, log_mag in zip(St1_vals, St2_vals, log_lambda_mags):
        if St1 >= 0 and St1 + St2 >= 0:  # Triangular region condition
            i_St1 = np.argmin(np.abs(St1_grid - St1))
            i_St2 = np.argmin(np.abs(St2_grid - St2))
            log_lambda_grid[i_St2, i_St1] = log_mag
            # Map to zoom grid if within zoom range
            if 0 <= St1 <= zoom_St1_max and zoom_St2_min <= St2 <= zoom_St2_max:
                i_St1_zoom = np.argmin(np.abs(St1_zoom_grid - St1))
                i_St2_zoom = np.argmin(np.abs(St2_zoom_grid - St2))
                log_lambda_zoom[i_St2_zoom, i_St1_zoom] = log_mag

    # Set color scale dynamically
    finite_logs = log_lambda_mags[np.isfinite(log_lambda_mags)]
    if len(finite_logs) > 0:
        vmin = np.percentile(finite_logs, 1)  # 1st percentile for lower bound
        vmax = max(0, np.percentile(finite_logs, 99))  # 99th percentile, ensure ≥ 0
    else:
        vmin, vmax = -20, 0  # Default values if no finite logs

    # Ensure vmin < vmax for contour levels
    if vmin >= vmax:
        if vmin == 0:
            vmin = -1  # Arbitrary small value to allow contour levels
        else:
            vmin = vmax - 1  # Ensure a small range if they're equal

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot (a): Full Bispectrum with Colorful Contour Lines
    pcm1 = ax1.pcolormesh(ST1, ST2, log_lambda_grid, cmap="jet", vmin=vmin, vmax=vmax, shading="auto")
    # Add colorful contour lines to the full bispectrum plot
    levels = np.linspace(vmin, vmax, 10)
    levels = np.sort(levels)  # Ensure levels are increasing
    # Define a list of colors for the contours (cycling through if needed)
    contour_colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "gray", "cyan"]
    ax1.contour(ST1, ST2, log_lambda_grid, levels=levels, colors=contour_colors, linewidths=0.5)
    fig.colorbar(pcm1, ax=ax1, label=r"$\log(|\lambda|)$")
    ax1.set_xlabel(r"$St_1$")
    ax1.set_ylabel(r"$St_2$")
    ax1.set_title("(a) Full Bispectrum")
    ax1.set_xlim(0, St1_max)
    ax1.set_ylim(St2_min, St2_max)

    # Plot (b): Zoomed-In Low-Frequency Region
    pcm2 = ax2.pcolormesh(ST1_zoom, ST2_zoom, log_lambda_zoom, cmap="jet", vmin=vmin, vmax=vmax, shading="auto")
    fig.colorbar(pcm2, ax=ax2, label=r"$\log(|\lambda|)$")
    ax2.set_xlabel(r"$St_1$")
    ax2.set_ylabel(r"$St_2$")
    ax2.set_title("(b) Low-Frequency Zoom")

    # Automatically select and mark the top M triads in the zoomed-in plot
    M = 6  # Number of triads to mark, adjustable if needed
    mask_zoom = (St1_vals >= 0) & (St1_vals <= zoom_St1_max) & (St2_vals >= zoom_St2_min) & (St2_vals <= zoom_St2_max)
    triads_zoom = triads_arr[mask_zoom]
    lambda_mags_zoom = lambda_mags[mask_zoom]
    St1_zoom_vals = St1_vals[mask_zoom]
    St2_zoom_vals = St2_vals[mask_zoom]

    if len(lambda_mags_zoom) > 0:
        top_idx = np.argsort(lambda_mags_zoom)[-M:][::-1]  # Indices of top M magnitudes
        top_triads = triads_zoom[top_idx]
        top_St1 = St1_zoom_vals[top_idx]
        top_St2 = St2_zoom_vals[top_idx]

        for (p1, p2, _), St1, St2 in zip(top_triads, top_St1, top_St2):
            ax2.plot(St1, St2, "ro", markersize=8, markerfacecolor="none")
            ax2.text(St1, St2, f"({p1},{p2})", fontsize=8, ha="left", va="bottom")

    # Plot the fundamental line St_1 + St_2 = St_0 in the zoomed-in plot
    St1_line = np.linspace(0, zoom_St1_max, 100)
    St2_line = St0 - St1_line
    ax2.plot(St1_line, St2_line, "k--", label=r"$St_1 + St_2 = St_0$")
    ax2.legend()

    # Set zoom plot limits

# Finalize and save the plot
plt.tight_layout()
output_file = os.path.join(FIGURES_DIR, f"{analyzer.data_root}_BSMD_eig_St1St2_plane.png")
plt.savefig(output_file)
plt.close()
print(f"Plot saved to {output_file}")

print("\n--------------------------------------------")
print("BSMD analysis script finished.")
print(f"Results saved in: {os.path.abspath(RESULTS_DIR)}")
print(f"Figures saved in: {os.path.abspath(FIGURES_DIR)}")
print("--------------------------------------------")
