#!/usr/bin/env python3
"""
Proper orthogonal decomposition (POD)

Following the implementation of https://github.com/MathEXLab/PySPOD/blob/main/pyspod/pod/standard.py

we want a pure python version using the same style and language as spod.py

Author: R. Frantz

Reference codes:
    - https://github.com/MathEXLab/PySPOD/blob/main/pyspod/pod/standard.py
"""

# Standard library imports
import os
import time

import h5py
import matplotlib.pyplot as plt

# Third-party imports
import numpy as np
import scipy.linalg  # For eigh

from configs import CMAP_DIV, CMAP_SEQ, FIGURES_DIR, RESULTS_DIR

# Local application/library specific imports
from utils import (
    BaseAnalyzer,
    auto_detect_weight_type,
    load_jetles_data,
    load_mat_data,
    make_result_filename,  # For saving results
)


class PODAnalyzer(BaseAnalyzer):
    """Proper Orthogonal Decomposition (POD) analyzer.

    This class implements POD, a technique to decompose a data ensemble into
    a set of optimal orthogonal modes (spatial structures) and corresponding
    time coefficients. The modes are ranked by their energy content, captured
    by the eigenvalues.

    The POD method typically involves:
    1. Forming a data matrix from snapshots of the flow field (or other data).
    2. Subtracting the temporal mean from the data.
    3. Performing a Singular Value Decomposition (SVD) of the (weighted) mean-subtracted data matrix.
       Alternatively, for snapshot POD, an eigenvalue decomposition of the covariance matrix.

    Key Attributes:
        modes (np.ndarray): Spatial POD modes (Phi). Shape: (n_spatial_points, n_modes_save).
        eigenvalues (np.ndarray): POD eigenvalues (lambda), representing energy of modes.
                                  Shape: (n_modes_save,).
        time_coefficients (np.ndarray): Temporal coefficients (A) corresponding to POD modes.
                                        Shape: (n_snapshots, n_modes_save).
        temporal_mean (np.ndarray): Mean snapshot subtracted from the data before POD.
                                    Shape: (n_spatial_points,).
        n_modes_save (int): Number of POD modes to compute, save, and use for plotting.
        data_matrix (np.ndarray): Preprocessed data matrix [time, space].
        W (np.ndarray): Spatial weighting matrix (diagonal).
        fs (float): Sampling frequency of the data (if available, mainly for context).

    Inherits from:
        BaseAnalyzer: Provides common functionalities for data loading and preprocessing.
                      Note: `nfft` and `overlap` from `BaseAnalyzer` are not directly used by POD
                      but are initialized with dummy values.
    """

    def __init__(self, file_path, results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR, data_loader=None, spatial_weight_type="auto", n_modes_save=10):
        """
        Initialize the PODAnalyzer.

        Args:
            file_path (str): Path to the data file (e.g., .mat, .h5).
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
            n_modes_save (int, optional): Number of dominant POD modes to compute, save,
                                          and consider for plotting/reconstruction.
                                          Defaults to 10.
        """
        # Call BaseAnalyzer's __init__.
        # nfft and overlap are not directly used by POD but are part of BaseAnalyzer.
        super().__init__(
            file_path=file_path,
            nfft=1,  # Not used by POD, can be a dummy value
            overlap=0,  # Not used by POD, can be a dummy value
            results_dir=results_dir,
            figures_dir=figures_dir,
            data_loader=data_loader,
            spatial_weight_type=spatial_weight_type,
        )

        self.n_modes_save = n_modes_save
        self.modes = np.array([])  # Spatial modes (Phi)
        self.eigenvalues = np.array([])  # Eigenvalues (lambda)
        self.time_coefficients = np.array([])  # Temporal coefficients (Psi)
        self.temporal_mean = np.array([])  # Temporal mean of the data

        # Update the analysis type for filenames
        self.analysis_type = "pod"

    def perform_pod(self):
        """Perform POD analysis on the loaded and preprocessed data.

        This method computes the POD modes, eigenvalues, and time coefficients.
        The steps involved are:
        1. Ensure data is loaded (expects `self.data['q']` to be [time, space]).
        2. Subtract the temporal mean (`self.temporal_mean`) from the data matrix.
        3. Apply spatial weights (`self.W`) to the mean-subtracted data.
        4. Compute the covariance matrix (snapshot POD approach: C = X^T * W * X).
        5. Solve the eigenvalue problem for the covariance matrix to get eigenvalues
           and eigenvectors (which relate to time coefficients).
        6. Reconstruct spatial modes by projecting the data onto the eigenvectors.
        7. Sort modes and eigenvalues by energy (descending eigenvalues).
        8. Truncate to `self.n_modes_save`.

        Attributes set:
            eigenvalues (np.ndarray): Sorted POD eigenvalues.
            modes (np.ndarray): Sorted spatial POD modes.
            time_coefficients (np.ndarray): Sorted temporal coefficients.
            temporal_mean (np.ndarray): Calculated temporal mean of the data.
        """
        if "q" not in self.data:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")

        print("Performing POD analysis...")
        start_time = time.time()

        # Data is expected as [time, space]
        data_matrix = self.data["q"]  # Shape (Ns, Nspace)
        Ns = self.data["Ns"]
        Nspace = data_matrix.shape[1]

        # 1. Subtract temporal mean
        # For POD, typically the mean over snapshots is removed.
        self.temporal_mean = np.mean(data_matrix, axis=0)  # Mean over time, shape (Nspace,)
        data_mean_removed = data_matrix - self.temporal_mean  # Shape (Ns, Nspace)

        # 2. Apply spatial weights
        # self.W is [Nspace, Nspace] diagonal matrix or [Nspace] vector for weights.
        # If W is a vector, apply element-wise multiplication after transposing data_mean_removed.
        # If W is a diagonal matrix, it's data_mean_removed @ W (assuming W is diag(weights_vector))
        # PySPOD uses d.conj().T @ (d * self._weights), where d is [Nspace, Ntime] and weights are [Nspace, 1]
        # Let's ensure W is a 1D array of weights for element-wise multiplication with spatial dimensions.
        if self.W.ndim == 2 and self.W.shape[0] == self.W.shape[1]:
            weights_vector = np.diag(self.W)
        elif self.W.ndim == 1:
            weights_vector = self.W
        elif self.W.ndim == 2 and self.W.shape[1] == 1:  # Handle (Nspace, 1) column vector
            weights_vector = self.W.flatten()
        else:
            raise ValueError(f"Unexpected shape for spatial weights W: {self.W.shape}")

        # data_mean_removed is (Ns, Nspace). Apply weights to spatial dimension.
        # (data_mean_removed * sqrt(weights_vector)) results in shape (Ns, Nspace)
        data_weighted = data_mean_removed * np.sqrt(weights_vector)  # Element-wise multiplication

        # 3. Form covariance matrix (Temporal covariance K_ij = <u_i(t) u_j(t)>_t )
        # Using method of snapshots: C = X^T * X where X is (Ns, Nspace_weighted)
        # The eigenvalue problem is C * A = Lambda * A, where A are temporal coeffs.
        # Or, use SVD: X = U * Sigma * Vh. Modes are U or Vh depending on X arrangement.
        # If X is (Time, Space), then U gives temporal structures, Vh gives spatial structures.
        # PySPOD style: d is (Nspace, Ntime). Q = d.conj().T @ (d * self._weights_vector)
        # This is (Ntime, Nspace_w) @ (Nspace_w, Ntime) -> (Ntime, Ntime) (temporal kernel)

        # Let's follow snapshot POD: data_weighted is (Ns, Nspace)
        # K_t = data_weighted @ data_weighted.T  (Ns, Ns) -- Temporal kernel
        # K_s = data_weighted.T @ data_weighted  (Nspace, Nspace) -- Spatial kernel
        # For Ns < Nspace, solve temporal kernel. For Nspace < Ns, solve spatial kernel.

        if Ns < Nspace:
            print(f"Number of snapshots ({Ns}) < number of spatial points ({Nspace}). Solving temporal eigenvalue problem.")
            K = (1.0 / Ns) * (data_weighted @ data_weighted.T)  # (Ns, Ns)
            eigenvalues_temp, temporal_coeffs_unscaled = scipy.linalg.eigh(K)
            # Sort eigenvalues and eigenvectors in descending order
            sorted_indices = np.argsort(eigenvalues_temp)[::-1]
            self.eigenvalues = eigenvalues_temp[sorted_indices]
            temporal_coeffs_unscaled = temporal_coeffs_unscaled[:, sorted_indices]
            # Calculate spatial modes: Phi = data_weighted.T @ A_temp * (1/sqrt(Lambda_temp * Ns))
            # data_weighted.T is (Nspace, Ns). temporal_coeffs_unscaled is (Ns, Ns)
            # modes_unnorm = data_weighted.T @ temporal_coeffs_unscaled # (Nspace, Ns)
            # self.modes = modes_unnorm / np.sqrt(self.eigenvalues * Ns) # Normalize modes
            # Alternative normalization for modes to be orthonormal with weights:
            # Phi = X_weighted^T * Psi_weighted / Lambda_weighted
            # Let Psi_temp be the eigenvectors of K_t. Phi = X_weighted^T * Psi_temp
            modes_temp = data_weighted.T @ temporal_coeffs_unscaled  # (Nspace, Ns)
            # Normalize spatial modes: divide by sqrt(eigenvalue * Ns) and by sqrt(weights_vector)
            # Each column of modes_temp corresponds to an eigenvalue.
            # modes_temp[:, k] / sqrt(lambda_k * Ns) gives modes scaled by sqrt(W)
            # To get actual spatial modes, divide by sqrt(W) again
            # For modes to be orthonormal w.r.t W: Phi_i^T W Phi_j = delta_ij
            # Phi = X^T Psi / sqrt(Lambda)
            # Let X_w = X * sqrt(W). K_t = X_w X_w^T. Psi_w from K_t Psi_w = Lambda Psi_w.
            # Phi_w = X_w^T Psi_w / sqrt(Lambda). Here Phi_w = Phi * sqrt(W).
            # So, Phi = (X_w^T Psi_w / sqrt(Lambda)) / sqrt(W)
            # modes_temp = data_weighted.T @ temporal_coeffs_unscaled is (Nspace, Ns) = X_w^T Psi_w
            self.modes = (modes_temp / np.sqrt(self.eigenvalues * Ns)) / np.sqrt(weights_vector[:, np.newaxis])
            self.time_coefficients = temporal_coeffs_unscaled * np.sqrt(self.eigenvalues * Ns)  # Scale temporal coefficients

        else:
            print(f"Number of spatial points ({Nspace}) <= number of snapshots ({Ns}). Solving spatial eigenvalue problem.")
            K = (1.0 / Ns) * (data_weighted.T @ data_weighted)  # (Nspace, Nspace)
            eigenvalues_spatial, spatial_modes_weighted = scipy.linalg.eigh(K)
            # Sort eigenvalues and eigenvectors
            sorted_indices = np.argsort(eigenvalues_spatial)[::-1]
            self.eigenvalues = eigenvalues_spatial[sorted_indices]
            spatial_modes_weighted = spatial_modes_weighted[:, sorted_indices]
            # Spatial modes (weighted) are eigenvectors of K_s.
            # To get unweighted modes: Phi = Phi_w / sqrt(W)
            self.modes = spatial_modes_weighted / np.sqrt(weights_vector[:, np.newaxis])
            # Calculate temporal coefficients: Psi = X @ Phi (project original mean-removed data onto unweighted modes)
            # data_mean_removed is (Ns, Nspace). self.modes is (Nspace, n_modes_save)
            self.time_coefficients = data_mean_removed @ self.modes

        # Ensure modes are real (they should be from eigh on symmetric matrix)
        self.modes = np.real(self.modes)
        self.eigenvalues = np.real(self.eigenvalues)
        self.time_coefficients = np.real(self.time_coefficients)

        # Truncate if n_modes_save is less than available modes
        n_available_modes = self.eigenvalues.shape[0]
        if self.n_modes_save > n_available_modes:
            print(f"Warning: n_modes_save ({self.n_modes_save}) is greater than available modes ({n_available_modes}). Using all available modes.")
            self.n_modes_save = n_available_modes

        self.modes = self.modes[:, : self.n_modes_save]
        self.eigenvalues = self.eigenvalues[: self.n_modes_save]
        self.time_coefficients = self.time_coefficients[:, : self.n_modes_save]

        end_time = time.time()
        print(f"POD analysis completed in {end_time - start_time:.2f} seconds.")
        print(f"Computed {self.modes.shape[1]} POD modes.")

    def save_results(self, filename=None):
        """Save POD modes, eigenvalues, and time coefficients to an HDF5 file.

        The results are saved in `self.results_dir`. If `filename` is None,
        it's generated using `make_result_filename` based on the input data file name
        and the analysis type ('pod').

        Args:
            filename (str, optional): Custom filename for the HDF5 output.
                                      Defaults to None (auto-generated).

        Datasets saved:
            'Eigenvalues': POD eigenvalues.
            'Modes': Spatial POD modes.
            'TimeCoefficients': Temporal POD coefficients.
            'TemporalMean': The temporal mean snapshot subtracted from the data.
            'dt': Time step of the original data (if available).
        """
        if not filename:
            # Use a simplified name for POD as nfft/overlap are not primary params
            filename = f"{self.data_root}_{self.data.get('Ns', 0)}snapshots_{self.analysis_type}.hdf5"

        save_path = os.path.join(self.results_dir, filename)
        print(f"Saving POD results to {save_path}")

        with h5py.File(save_path, "w") as f:
            # Save attributes from BaseAnalyzer's save_results if relevant
            f.attrs["analysis_type"] = self.analysis_type
            f.attrs["n_modes_saved"] = self.n_modes_save
            f.attrs["n_snapshots"] = self.data.get("Ns", 0)
            f.attrs["dt"] = self.data.get("dt", 0)
            f.attrs["Nspace"] = self.modes.shape[0]

            # Save coordinates and weights
            if "x" in self.data:
                f.create_dataset("x", data=self.data["x"], compression="gzip")
            if "y" in self.data:
                f.create_dataset("y", data=self.data["y"], compression="gzip")
            if self.W.size > 0:
                f.create_dataset("W", data=self.W, compression="gzip")
            if self.temporal_mean.size > 0:
                f.create_dataset("temporal_mean", data=self.temporal_mean, compression="gzip")

            # Save POD specific results
            f.create_dataset("modes", data=self.modes, compression="gzip")  # Phi (spatial modes)
            f.create_dataset("eigenvalues", data=self.eigenvalues, compression="gzip")  # Lambda
            f.create_dataset("time_coefficients", data=self.time_coefficients, compression="gzip")  # Psi (temporal coefficients)

        print("POD results saved.")

    def plot_eigenvalues(self):
        """Plot the POD eigenvalue spectrum (energy vs. mode number).

        Shows the decay of energy (eigenvalues) with increasing mode number.
        The plot is saved to `self.figures_dir`.
        """
        if self.eigenvalues.size == 0:
            print("No eigenvalues to plot. Run perform_pod() first.")
            return

        plt.figure(figsize=(8, 5))
        mode_indices = np.arange(1, len(self.eigenvalues) + 1)
        plt.plot(mode_indices, self.eigenvalues / np.sum(self.eigenvalues) * 100, "o-", linewidth=2, markersize=6)
        plt.yscale("log")  # Eigenvalue spectrum is often plotted on a log scale
        plt.xlabel("Mode Number")
        plt.ylabel("Normalized Eigenvalue (Energy Percentage %)")
        plt.title("POD Eigenvalue Spectrum")
        plt.grid(True, which="both", ls="--")
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_pod_eigenvalues.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"POD eigenvalue plot saved to {plot_filename}")

    def plot_modes(self, n_modes_to_plot=4):
        """Plot the spatial POD modes.

        Visualizes the first `n_modes_to_plot` dominant spatial modes.
        Requires spatial coordinates (e.g., `self.data['x']`, `self.data['y']`) to be loaded.
        Assumes 2D spatial data that can be reshaped using `Nx` and `Ny` from `self.data`.
        Plots are saved to `self.figures_dir`.

        Args:
            n_modes_to_plot (int, optional): Number of leading spatial modes to plot.
                                           Defaults to 4.
        """
        if self.modes.size == 0:
            print("No modes to plot. Run perform_pod() first.")
            return

        n_modes_to_plot = min(n_modes_to_plot, self.modes.shape[1], self.n_modes_save)
        if n_modes_to_plot == 0:
            print("No modes available to plot.")
            return

        Nx = self.data.get("Nx", int(np.sqrt(self.modes.shape[0])))
        Ny = self.data.get("Ny", int(np.sqrt(self.modes.shape[0])))
        is_2d_plot = (self.modes.shape[0] == Nx * Ny) and (Nx > 1 and Ny > 1)
        x_coords = self.data.get("x", np.arange(Nx))
        y_coords = self.data.get("y", np.arange(Ny))

        # Determine if plotting 1D or 2D modes
        is_2d_plot = (self.modes.shape[0] == Nx * Ny) and (Nx > 1 and Ny > 1)

        num_cols = min(n_modes_to_plot, 2)  # Max 2 columns
        num_rows = (n_modes_to_plot + num_cols - 1) // num_cols
        fig_width = 6 * num_cols
        fig_height = 5 * num_rows

        plt.figure(figsize=(fig_width, fig_height))

        for i in range(n_modes_to_plot):
            plt.subplot(num_rows, num_cols, i + 1)
            mode_to_plot = self.modes[:, i]
            if is_2d_plot:
                mode_reshaped = mode_to_plot.reshape(Nx, Ny)
                # Determine extent for imshow if x,y are 1D arrays
                extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()] if x_coords.ndim == 1 and y_coords.ndim == 1 else None
                plt.imshow(mode_reshaped.T, aspect="auto", origin="lower", extent=extent, cmap=CMAP_SEQ)
                plt.colorbar(label="Mode Amplitude")
                plt.xlabel("X")
                plt.ylabel("Y")
            else:  # 1D plot
                plt.plot(mode_to_plot)
                plt.xlabel("Spatial Index")
                plt.ylabel("Mode Amplitude")

            plt.title(f"POD Mode {i + 1} (Energy: {self.eigenvalues[i] / np.sum(self.eigenvalues) * 100:.2f}%)")

        plt.tight_layout()
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_pod_modes.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"POD mode plot saved to {plot_filename}")

    def plot_time_coefficients(self, n_coeffs_to_plot=2, n_snapshots_plot=None):
        """Plot the temporal coefficients for selected modes.

        Displays the time evolution of the coefficients for the first `n_coeffs_to_plot` modes.
        If `self.data['t']` (time vector) is available, it's used for the x-axis.
        Otherwise, snapshot index is used.
        Plots are saved to `self.figures_dir`.

        Args:
            n_coeffs_to_plot (int, optional): Number of leading temporal coefficients to plot.
                                            Defaults to 2.
            n_snapshots_plot (int, optional): Number of time snapshots to include in the plot.
                                              If None, all snapshots are used. Defaults to None.
        """
        if self.time_coefficients.size == 0:
            print("No time coefficients to plot. Run perform_pod() first.")
            return

        n_coeffs_to_plot = min(n_coeffs_to_plot, self.time_coefficients.shape[1], self.n_modes_save)
        if n_coeffs_to_plot == 0:
            print("No coefficients available to plot.")
            return

        Ns_total = self.time_coefficients.shape[0]
        if n_snapshots_plot is None or n_snapshots_plot > Ns_total:
            n_snapshots_plot = Ns_total

        time_vector = np.arange(n_snapshots_plot) * self.data.get("dt", 1.0)

        plt.figure(figsize=(10, 3 * n_coeffs_to_plot))
        for i in range(n_coeffs_to_plot):
            plt.subplot(n_coeffs_to_plot, 1, i + 1)
            plt.plot(time_vector, self.time_coefficients[:n_snapshots_plot, i], linewidth=1.5)
            plt.xlabel("Time")
            plt.ylabel(f"Amplitude Mode {i + 1}")
            plt.title(f"Temporal Coefficient for POD Mode {i + 1}")
            plt.grid(True, linestyle=":")
            plt.xlim(time_vector.min(), time_vector.max())

        plt.tight_layout()
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_pod_time_coeffs.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"POD time coefficients plot saved to {plot_filename}")

    def plot_cumulative_energy(self):
        """Plot the cumulative energy captured by POD modes.

        Shows the percentage of total energy captured as more modes are included.
        The plot is saved to `self.figures_dir`.
        """
        if self.eigenvalues.size == 0:
            print("No eigenvalues to plot. Run perform_pod() first.")
            return

        cumulative_energy = np.cumsum(self.eigenvalues) / np.sum(self.eigenvalues) * 100
        mode_indices = np.arange(1, len(self.eigenvalues) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(mode_indices, cumulative_energy, "o-", linewidth=2, markersize=6)
        plt.xlabel("Number of Modes")
        plt.ylabel("Cumulative Energy (%)")
        plt.title("Cumulative Energy of POD Modes")
        plt.grid(True, which="both", ls="--")
        plt.ylim(0, 105)  # Show up to 100% or slightly more
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_pod_cumulative_energy.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"POD cumulative energy plot saved to {plot_filename}")

    def plot_reconstruction_error(self):
        """Plot the data reconstruction error using an increasing number of POD modes.

        Calculates and plots the normalized mean squared error (NMSE) of reconstructing
        the original data using a subset of POD modes. The error is shown as a function
        of the number of modes used for reconstruction.
        The plot is saved to `self.figures_dir`.
        """
        if self.modes.size == 0 or self.time_coefficients.size == 0 or "q" not in self.data:
            print("Data, modes, or time coefficients not available. Run perform_pod() first.")
            return

        data_matrix = self.data["q"]
        data_mean_removed = data_matrix - self.temporal_mean
        norm_data_mean_removed = np.linalg.norm(data_mean_removed, "fro")

        reconstruction_errors = []
        n_modes_check = self.modes.shape[1]  # Number of available/saved modes

        for k in range(1, n_modes_check + 1):
            # Reconstruct data using k modes: Psi_k @ Phi_k.T
            # self.time_coefficients is (Ns, n_modes_save)
            # self.modes is (Nspace, n_modes_save)
            reconstructed_data_k_modes = self.time_coefficients[:, :k] @ self.modes[:, :k].T
            error = np.linalg.norm(data_mean_removed - reconstructed_data_k_modes, "fro") / norm_data_mean_removed
            reconstruction_errors.append(error * 100)  # As percentage

        mode_indices = np.arange(1, n_modes_check + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(mode_indices, reconstruction_errors, "s-", linewidth=2, markersize=6)
        plt.xlabel("Number of Modes Used for Reconstruction")
        plt.ylabel("Reconstruction Error (%)")
        plt.title("Data Reconstruction Error vs. Number of POD Modes")
        plt.grid(True, which="both", ls="--")
        plt.yscale("log")  # Error often drops off exponentially
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_pod_reconstruction_error.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"POD reconstruction error plot saved to {plot_filename}")

    def plot_reconstruction_comparison(self, snapshot_indices_to_plot=None, modes_for_reconstruction=None):
        """Compare original snapshots with their POD reconstructions.

        Visualizes selected original data snapshots alongside their reconstructions
        using a specified number of POD modes. Requires spatial coordinates for plotting.
        Plots are saved to `self.figures_dir`.

        Args:
            snapshot_indices_to_plot (list of int, optional): Indices of snapshots to plot.
                                                              Defaults to [0, Ns//2, Ns-1] if None.
            modes_for_reconstruction (list of int, optional): Number of modes to use for reconstruction
                                                              in each comparison plot.
                                                              Defaults to [1, n_modes_save//2, n_modes_save] if None.
        """
        if self.modes.size == 0 or self.time_coefficients.size == 0 or "q" not in self.data:
            print("Data, modes, or time coefficients not available. Run perform_pod() first.")
            return

        data_matrix = self.data["q"]
        data_mean_removed = data_matrix - self.temporal_mean
        Ns, Nspace = data_mean_removed.shape

        if snapshot_indices_to_plot is None:
            snapshot_indices_to_plot = [0, Ns // 2, Ns - 1]
            # Ensure indices are unique and within bounds, especially for small Ns
            snapshot_indices_to_plot = sorted(list(set(idx for idx in snapshot_indices_to_plot if 0 <= idx < Ns)))
            if not snapshot_indices_to_plot:  # if Ns is too small, pick at least the first one
                snapshot_indices_to_plot = [0]

        if modes_for_reconstruction is None:
            k_max = self.modes.shape[1]
            modes_for_reconstruction = [1, k_max // 2, k_max]
            # Ensure values are unique, positive, and within bounds
            modes_for_reconstruction = sorted(list(set(k for k in modes_for_reconstruction if 0 < k <= k_max)))
            if not modes_for_reconstruction and k_max > 0:  # if k_max is small
                modes_for_reconstruction = [k_max]
            elif k_max == 0:
                print("No modes available for reconstruction comparison.")
                return

        if not snapshot_indices_to_plot or not modes_for_reconstruction:
            print("Not enough snapshots or modes to plot reconstruction comparison.")
            return

        # Determine plot layout details (similar to plot_modes)
        Nx = self.data.get("Nx", int(np.sqrt(Nspace)))
        Ny = self.data.get("Ny", int(np.sqrt(Nspace)))
        is_2d_plot = (Nspace == Nx * Ny) and (Nx > 1 and Ny > 1)
        x_coords = self.data.get("x", np.arange(Nx))
        y_coords = self.data.get("y", np.arange(Ny))

        num_snapshots_to_show = len(snapshot_indices_to_plot)
        num_recons_per_snapshot = len(modes_for_reconstruction)

        # Each row: Original + Reconstructions
        # num_cols = 1 (original) + num_recons_per_snapshot
        fig, axes = plt.subplots(num_snapshots_to_show, 1 + num_recons_per_snapshot, figsize=(5 * (1 + num_recons_per_snapshot), 4 * num_snapshots_to_show), squeeze=False)  # ensure axes is always 2D array

        for i, snap_idx in enumerate(snapshot_indices_to_plot):
            original_snapshot = data_mean_removed[snap_idx, :]

            # Plot original snapshot
            ax = axes[i, 0]
            if is_2d_plot:
                img_data = original_snapshot.reshape(Nx, Ny).T
                extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()] if x_coords.ndim == 1 and y_coords.ndim == 1 else None
                im = ax.imshow(img_data, aspect="auto", origin="lower", extent=extent, cmap=CMAP_DIV)
                fig.colorbar(im, ax=ax, label="Amplitude")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
            else:
                ax.plot(original_snapshot)
                ax.set_xlabel("Spatial Index")
                ax.set_ylabel("Amplitude")
            ax.set_title(f"Original Snapshot {snap_idx} (Mean Rem.)")

            # Plot reconstructions
            for j, k_modes in enumerate(modes_for_reconstruction):
                ax = axes[i, j + 1]
                # Reconstruct using k_modes: Psi[snap_idx, :k_modes] @ Phi[:k_modes, :].T
                # self.time_coefficients is (Ns, n_modes_save)
                # self.modes is (Nspace, n_modes_save)
                reconstructed_snapshot_k = self.time_coefficients[snap_idx, :k_modes] @ self.modes[:, :k_modes].T

                if is_2d_plot:
                    img_data_recon = reconstructed_snapshot_k.reshape(Nx, Ny).T
                    im = ax.imshow(img_data_recon, aspect="auto", origin="lower", extent=extent, cmap=CMAP_DIV)
                    fig.colorbar(im, ax=ax, label="Amplitude")
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                else:
                    ax.plot(reconstructed_snapshot_k)
                    ax.set_xlabel("Spatial Index")
                    ax.set_ylabel("Amplitude")
                ax.set_title(f"Recon. w/ {k_modes} Modes")

        plt.tight_layout()
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_pod_reconstruction_comparison.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"POD reconstruction comparison plot saved to {plot_filename}")

    def check_spatial_mode_orthogonality(self, tolerance=1e-9):
        """Check the orthogonality of spatial modes with respect to weights W.

        Verifies that `Modes.T @ W_diag @ Modes` is close to the identity matrix,
        where `W_diag` is the diagonal matrix of spatial weights.
        Prints a message indicating whether the modes are orthogonal within the given tolerance.

        Args:
            tolerance (float, optional): Tolerance for checking orthogonality.
                                       Defaults to 1e-9.
        """
        if self.modes.size == 0 or self.W.size == 0:
            print("Modes or weights not available. Run perform_pod() first.")
            return False

        print("\nChecking spatial mode orthogonality (Modes.T @ W @ Modes)...")
        Nspace, n_saved_modes = self.modes.shape

        # Ensure W is a diagonal matrix for the check
        if self.W.ndim == 1:
            W_diag_matrix = np.diag(self.W)
        elif self.W.ndim == 2 and self.W.shape[0] == self.W.shape[1] and np.allclose(self.W, np.diag(np.diag(self.W))):
            W_diag_matrix = self.W
        elif self.W.ndim == 2 and self.W.shape[1] == 1:  # (Nspace, 1) column vector
            W_diag_matrix = np.diag(self.W.flatten())
        else:
            print(f"  Warning: Unexpected shape or type for spatial weights W: {self.W.shape}. Cannot perform accurate orthogonality check.")
            return False

        ortho_check_matrix = self.modes.T @ W_diag_matrix @ self.modes
        identity_matrix = np.eye(n_saved_modes)

        # Check diagonals are close to 1
        diag_diff = np.abs(np.diag(ortho_check_matrix) - 1.0)
        max_diag_deviation = np.max(diag_diff)

        # Check off-diagonals are close to 0
        off_diag_mask = ~np.eye(n_saved_modes, dtype=bool)
        max_off_diag_val = np.max(np.abs(ortho_check_matrix[off_diag_mask])) if n_saved_modes > 1 else 0.0

        is_orthogonal = (max_diag_deviation < tolerance) and (max_off_diag_val < tolerance)

        print(f"  Max deviation of diagonal elements from 1: {max_diag_deviation:.2e}")
        print(f"  Max absolute value of off-diagonal elements: {max_off_diag_val:.2e}")
        if is_orthogonal:
            print("  Spatial modes appear to be W-orthogonal.")
        else:
            print("  Warning: Spatial modes may not be perfectly W-orthogonal within tolerance.")

        # Optional: Plot the orthogonality matrix
        plt.figure(figsize=(7, 6))
        plt.imshow(ortho_check_matrix, cmap=CMAP_DIV, vmin=-np.max(np.abs(ortho_check_matrix)), vmax=np.max(np.abs(ortho_check_matrix)))
        plt.colorbar(label="Value")
        plt.title("Spatial Mode Orthogonality Check (Modes.T @ W @ Modes)")
        plt.xlabel("Mode Index")
        plt.ylabel("Mode Index")
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_pod_spatial_ortho_check.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"  Spatial orthogonality check plot saved to {plot_filename}")
        return is_orthogonal

    def check_temporal_coefficient_orthogonality(self, tolerance=1e-9):
        """Check the orthogonality of temporal coefficients.

        Verifies that `(1/Ns) * TimeCoeffs.T @ TimeCoeffs` is close to `diag(Eigenvalues)`,
        where `Ns` is the number of snapshots.
        Prints a message indicating whether the coefficients are orthogonal (scaled by eigenvalues)
        within the given tolerance.

        Args:
            tolerance (float, optional): Tolerance for checking orthogonality.
                                       Defaults to 1e-9.
        """
        if self.time_coefficients.size == 0 or self.eigenvalues.size == 0 or "Ns" not in self.data:
            print("Time coefficients, eigenvalues, or Ns not available. Run perform_pod() first.")
            return False

        print("\nChecking temporal coefficient pseudo-orthogonality ((1/Ns) * Psi.T @ Psi)...")
        Ns = self.data["Ns"]
        n_saved_coeffs = self.time_coefficients.shape[1]

        # Expected matrix based on POD theory for snapshot POD temporal eigenvectors
        # (Psi_temp.T @ Psi_temp) should be Identity if Psi_temp are normalized eigenvectors of K_t.
        # self.time_coefficients = Psi_temp * sqrt(eigenvalues_temp * Ns)
        # So (1/Ns) * self.time_coefficients.T @ self.time_coefficients =
        # (1/Ns) * sqrt(L*Ns) * Psi_temp.T @ Psi_temp * sqrt(L*Ns) = L (diag(eigenvalues))
        ortho_check_matrix = (1.0 / Ns) * (self.time_coefficients.T @ self.time_coefficients)
        expected_diag_matrix = np.diag(self.eigenvalues[:n_saved_coeffs])

        diff_matrix = ortho_check_matrix - expected_diag_matrix

        # Check diagonals
        diag_abs_error = np.abs(np.diag(diff_matrix))
        max_diag_abs_error = np.max(diag_abs_error)

        # Check off-diagonals (should be close to 0 in both ortho_check_matrix and expected_diag_matrix)
        off_diag_mask = ~np.eye(n_saved_coeffs, dtype=bool)
        max_off_diag_val_computed = np.max(np.abs(ortho_check_matrix[off_diag_mask])) if n_saved_coeffs > 1 else 0.0

        # is_orthogonal means diag(ortho_check_matrix) approx diag(expected_matrix) AND off-diag(ortho_check_matrix) approx 0
        is_pseudo_orthogonal = (max_diag_abs_error < tolerance) and (max_off_diag_val_computed < tolerance)

        print(f"  Max absolute error of diagonal elements from eigenvalues: {max_diag_abs_error:.2e}")
        print(f"  Max absolute value of off-diagonal elements in computed matrix: {max_off_diag_val_computed:.2e}")
        if is_pseudo_orthogonal:
            print("  Temporal coefficients appear to satisfy (1/Ns) * Psi.T @ Psi = diag(Lambda).")
        else:
            print("  Warning: Temporal coefficients may not perfectly satisfy (1/Ns) * Psi.T @ Psi = diag(Lambda).")

        # Optional: Plot the computed matrix and the expected diagonal matrix
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Plot 1: (1/Ns) * Psi.T @ Psi
        im1 = axes[0].imshow(ortho_check_matrix, cmap=CMAP_DIV, vmin=-np.max(np.abs(ortho_check_matrix)), vmax=np.max(np.abs(ortho_check_matrix)))
        fig.colorbar(im1, ax=axes[0], label="Value")
        axes[0].set_title("(1/Ns) * Psi.T @ Psi (Computed)")
        axes[0].set_xlabel("Mode Index")
        axes[0].set_ylabel("Mode Index")
        # Plot 2: diag(Eigenvalues)
        im2 = axes[1].imshow(expected_diag_matrix, cmap=CMAP_DIV, vmin=-np.max(np.abs(expected_diag_matrix)), vmax=np.max(np.abs(expected_diag_matrix)))
        fig.colorbar(im2, ax=axes[1], label="Value")
        axes[1].set_title("diag(Eigenvalues) (Expected)")
        axes[1].set_xlabel("Mode Index")
        axes[1].set_ylabel("Mode Index")

        plt.tight_layout()
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_pod_temporal_ortho_check.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"  Temporal orthogonality check plot saved to {plot_filename}")
        return is_pseudo_orthogonal

    def run_analysis(self, plot_n_modes_spatial=4, plot_n_coeffs_time=2, plot_snapshot_indices=None, plot_modes_for_reconstruction=None, check_orthogonality=True):
        """
        Run the full POD analysis pipeline: load, compute, save, plot, and verify.

        This method orchestrates the entire POD process:
        1. Loads and preprocesses data (calls `BaseAnalyzer.load_and_preprocess`).
        2. Performs POD computation (calls `perform_pod`).
        3. Saves the results (calls `save_results`).
        4. Generates and saves standard POD plots (calls various `plot_*` methods).
        5. Optionally, checks and prints mode/coefficient orthogonality.

        Args:
            plot_n_modes_spatial (int, optional): Number of spatial modes to plot.
                                                Defaults to 4.
            plot_n_coeffs_time (int, optional): Number of temporal coefficients to plot.
                                              Defaults to 2.
            plot_snapshot_indices (list of int, optional): Indices of snapshots for reconstruction plot.
                                                         Defaults to None (auto-selected).
            plot_modes_for_reconstruction (list of int, optional): Number of modes for reconstruction plot.
                                                                 Defaults to None (auto-selected).
            check_orthogonality (bool, optional): If True, perform and print orthogonality checks.
                                                Defaults to True.
        """
        print(f"\n--- Starting POD Analysis for: {os.path.basename(self.file_path)} ---")
        start_total_time = time.time()

        # Load data and calculate weights via BaseAnalyzer's run method.
        # compute_fft=False because POD is time-domain.
        super().run(compute_fft=False)

        # Perform POD
        self.perform_pod()

        # Save results
        self.save_results()  # This already calls super().save_results()

        # Plotting
        self.plot_eigenvalues()
        self.plot_modes(n_modes_to_plot=plot_n_modes_spatial)
        self.plot_time_coefficients(n_coeffs_to_plot=plot_n_coeffs_time)
        self.plot_cumulative_energy()
        self.plot_reconstruction_error()
        self.plot_reconstruction_comparison(snapshot_indices_to_plot=plot_snapshot_indices, modes_for_reconstruction=plot_modes_for_reconstruction)

        if check_orthogonality:
            self.check_spatial_mode_orthogonality()
            self.check_temporal_coefficient_orthogonality()

        end_total_time = time.time()
        print(f"\nPOD analysis and plotting completed successfully in {end_total_time - start_total_time:.2f} seconds.")


# Example usage when the script is run directly
if __name__ == "__main__":
    # --- Configuration ---
    # data_file = "./data/jetLES_small.mat" # Updated data path
    data_file = "./data/jetLES.mat"  # Path to your data file
    # data_file = "./data/cavityPIV.mat" # Path to your data file

    n_modes_to_save_main = 10  # Number of POD modes to save
    n_modes_to_plot_spatial_main = 4  # Number of spatial modes to visualize
    n_coeffs_to_plot_time_main = 5  # Number of temporal coefficients to visualize
    # ---------------------

    # Ensure data file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at '{data_file}'. Please update the path.")
        # Create dummy data for testing if no file exists
        print("Creating dummy data for testing purposes as data file not found.")
        dummy_Ns = 100
        dummy_Nx = 20
        dummy_Ny = 10
        dummy_q = np.random.rand(dummy_Ns, dummy_Nx * dummy_Ny)
        dummy_x = np.linspace(0, 1, dummy_Nx)
        dummy_y = np.linspace(0, 0.5, dummy_Ny)
        dummy_dt = 0.01

        # Save dummy data to a temporary HDF5 file
        data_file = "./data/dummy_pod_data.h5"
        os.makedirs("./data", exist_ok=True)
        with h5py.File(data_file, "w") as hf:
            hf.create_dataset("p", data=dummy_q.reshape(dummy_Ns, dummy_Nx, dummy_Ny))  # Store as 3D for loader
            hf.create_dataset("x", data=dummy_x)
            hf.create_dataset("y", data=dummy_y)
            hf.create_dataset("dt", data=dummy_dt)
        print(f"Saved dummy data to {data_file}")
        # For dummy data, always use the generic loader
        data_loader_main = load_mat_data  # generic h5 loader
        spatial_weights_main = "uniform"
    else:
        # Auto-detect loader and weights based on filename conventions from utils.py
        if "cavity" in data_file.lower():
            data_loader_main = load_mat_data  # Assuming load_mat_data handles cavity .mat files
            spatial_weights_main = "uniform"
            print("Cavity case detected: Using load_mat_data and uniform weights.")
        elif "jet" in data_file.lower():
            data_loader_main = load_jetles_data
            spatial_weights_main = "polar"
            print("Jet case detected: Using load_jetles_data and polar weights.")
        else:
            data_loader_main = load_mat_data  # Default for .mat or other .h5
            spatial_weights_main = auto_detect_weight_type(data_file)
            print(f"Unknown case: Using load_mat_data and '{spatial_weights_main}' weights.")

    # Create POD analyzer instance
    pod_analyzer = PODAnalyzer(file_path=data_file, results_dir=RESULTS_DIR, figures_dir=FIGURES_DIR, data_loader=data_loader_main, spatial_weight_type=spatial_weights_main, n_modes_save=n_modes_to_save_main)

    # Run the full analysis and plotting pipeline
    pod_analyzer.run_analysis(plot_n_modes_spatial=n_modes_to_plot_spatial_main, plot_n_coeffs_time=n_coeffs_to_plot_time_main)

    print("\n--------------------------------------------")
    print("POD analysis script finished.")
    print(f"Results saved in: {os.path.abspath(RESULTS_DIR)}")
    print(f"Figures saved in: {os.path.abspath(FIGURES_DIR)}")
    print("--------------------------------------------")
