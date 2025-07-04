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
import argparse
import os
import time
from typing import Optional

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Third-party imports
import numpy as np
import scipy.linalg  # For eigh

from configs import (
    CMAP_DIV,
    CMAP_SEQ,
    FIG_DPI,
    FIGURES_DIR_POD,
    RESULTS_DIR_POD,
)
from fft.spectral_utils import find_peaks, periodogram_rfft
from parallel_utils import print_optimization_status

# Local application/library specific imports
from utils import (
    BaseAnalyzer,
    auto_detect_weight_type,
    get_fig_aspect_ratio,
    make_result_filename,  # For saving results
    print_summary,
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

    def __init__(self, file_path, results_dir=RESULTS_DIR_POD, figures_dir=FIGURES_DIR_POD, data_loader=None, spatial_weight_type="auto", n_modes_save=10, use_parallel=True):
        """
        Initialize the PODAnalyzer.

        Args:
            file_path (str): Path to the data file (e.g., .mat, .h5).
            results_dir (str, optional): Directory to save analysis results (HDF5 files).
                                         Defaults to `RESULTS_DIR_POD` from `configs.py`.
            figures_dir (str, optional): Directory to save generated plots.
                                         Defaults to `FIGURES_DIR_POD` from `configs.py`.
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
            use_parallel=use_parallel,
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
        num_snapshots, num_space_points = data_matrix.shape

        # Input validation
        if num_snapshots < 2:
            raise ValueError(f"Need at least 2 snapshots for POD, got {num_snapshots}")
        if num_space_points < 1:
            raise ValueError(f"Need at least 1 spatial point, got {num_space_points}")

        # 1. Subtract temporal mean (more efficient with axis parameter)
        self.temporal_mean = np.mean(data_matrix, axis=0, dtype=np.float64)
        data_mean_removed = data_matrix - self.temporal_mean

        # 2. Apply spatial weights with better handling
        if self.spatial_weight_type == "uniform":
            self.W = np.ones(num_space_points, dtype=np.float64)

        # Ensure W is a 1D array for efficient broadcasting
        if self.W.ndim == 2:
            if self.W.shape[0] == self.W.shape[1]:
                weight_vector = np.diag(self.W)
            elif self.W.shape[1] == 1:
                weight_vector = self.W.ravel()
            else:
                raise ValueError(f"Unexpected shape for spatial weights W: {self.W.shape}")
        else:
            weight_vector = self.W

        # Validate weight vector
        if len(weight_vector) != num_space_points:
            raise ValueError(f"Weight vector length {len(weight_vector)} doesn't match spatial points {num_space_points}")

        # Apply weights efficiently using broadcasting
        sqrt_weights = np.sqrt(np.maximum(weight_vector, 1e-12))  # Avoid sqrt of negative/zero
        data_weighted = data_mean_removed * sqrt_weights

        # 3. Choose eigenvalue problem based on efficiency
        use_temporal_kernel = num_snapshots < num_space_points

        if use_temporal_kernel:
            print(f"Using temporal kernel: {num_snapshots} snapshots < {num_space_points} spatial points")
            # K = (1/Ns) * X * X^T where X is weighted data
            K = np.dot(data_weighted, data_weighted.T) / num_snapshots
            eigenvalues_temp, temporal_coeffs_unscaled = scipy.linalg.eigh(K)

            # Sort in descending order
            sort_idx = np.argsort(eigenvalues_temp)[::-1]
            self.eigenvalues = eigenvalues_temp[sort_idx]
            temporal_coeffs_unscaled = temporal_coeffs_unscaled[:, sort_idx]

            # Compute spatial modes efficiently
            # Avoid division by very small eigenvalues
            safe_eigenvals = np.maximum(self.eigenvalues * num_snapshots, 1e-12)
            normalization = 1.0 / np.sqrt(safe_eigenvals)

            modes_temp = np.dot(data_weighted.T, temporal_coeffs_unscaled)
            self.modes = (modes_temp * normalization) / sqrt_weights[:, np.newaxis]
            self.time_coefficients = temporal_coeffs_unscaled * np.sqrt(safe_eigenvals)

        else:
            print(f"Using spatial kernel: {num_space_points} spatial points <= {num_snapshots} snapshots")
            # K = (1/Ns) * X^T * X where X is weighted data
            K = np.dot(data_weighted.T, data_weighted) / num_snapshots
            eigenvalues_spatial, spatial_modes_weighted = scipy.linalg.eigh(K)

            # Sort in descending order
            sort_idx = np.argsort(eigenvalues_spatial)[::-1]
            self.eigenvalues = eigenvalues_spatial[sort_idx]
            spatial_modes_weighted = spatial_modes_weighted[:, sort_idx]

            # Get unweighted modes
            self.modes = spatial_modes_weighted / sqrt_weights[:, np.newaxis]
            # Project data onto modes
            self.time_coefficients = np.dot(data_mean_removed, self.modes)

        # Ensure real values (should be real from eigh on symmetric matrix)
        self.modes = np.real(self.modes)
        self.eigenvalues = np.real(self.eigenvalues)
        self.time_coefficients = np.real(self.time_coefficients)

        # Truncate to requested number of modes
        n_available_modes = len(self.eigenvalues)
        if self.n_modes_save > n_available_modes:
            print(f"Warning: n_modes_save ({self.n_modes_save}) > available modes ({n_available_modes}). Using all available.")
            self.n_modes_save = n_available_modes

        self.modes = self.modes[:, : self.n_modes_save]
        self.eigenvalues = self.eigenvalues[: self.n_modes_save]
        self.time_coefficients = self.time_coefficients[:, : self.n_modes_save]

        end_time = time.time()
        print(f"POD analysis completed in {end_time - start_time:.2f} seconds.")
        print(f"Computed {self.modes.shape[1]} POD modes.")

        # Print energy summary
        total_energy = np.sum(self.eigenvalues)
        if total_energy > 0:
            energy_pct = 100.0 * np.sum(self.eigenvalues) / np.sum(self.eigenvalues)
            print(f"Energy captured by {self.n_modes_save} modes: {energy_pct:.2f}%")

    def load_results(self, filename=None):
        """Load POD modes, eigenvalues, and time coefficients from an HDF5 file."""
        if not filename:
            filename = f"{self.data_root}_{self.data.get('Ns', 0)}snapshots_{self.analysis_type}.hdf5"
        load_path = os.path.join(self.results_dir, filename)
        print(f"Loading POD results from {load_path}")
        with h5py.File(load_path, "r") as f:
            # Load coordinates and weights (if present)
            if "x" in f:
                self.data["x"] = f["x"][:]
            if "y" in f:
                self.data["y"] = f["y"][:]
            if "W" in f:
                self.W = f["W"][:]
            if "temporal_mean" in f:
                self.temporal_mean = f["temporal_mean"][:]
            # Load POD results
            self.modes = f["modes"][:]
            self.eigenvalues = f["eigenvalues"][:]
            self.time_coefficients = f["time_coefficients"][:]
            # Load other attributes if present
            if "dt" in f.attrs:
                self.data["dt"] = f.attrs["dt"]
            if "n_snapshots" in f.attrs:
                self.data["Ns"] = f.attrs["n_snapshots"]
            if "Nspace" in f.attrs:
                self.data["Nspace"] = f.attrs["Nspace"]
        print("POD results loaded.")

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

        # Create figure with better memory management
        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            mode_indices = np.arange(1, len(self.eigenvalues) + 1)
            normalized_eigenvals = self.eigenvalues / np.sum(self.eigenvalues) * 100

            ax.plot(mode_indices, normalized_eigenvals, "o-", linewidth=2, markersize=6)

            # Annotate only first few and last few points to avoid clutter
            n_annotate = min(5, len(mode_indices))
            for idx in range(n_annotate):
                ax.text(mode_indices[idx], normalized_eigenvals[idx], f" {idx + 1}", fontsize=7, va="bottom")
            if len(mode_indices) > n_annotate:
                for idx in range(max(n_annotate, len(mode_indices) - 3), len(mode_indices)):
                    ax.text(mode_indices[idx], normalized_eigenvals[idx], f" {idx + 1}", fontsize=7, va="bottom")

            ax.set_yscale("log")
            ax.set_xlabel("Mode Number")
            ax.set_ylabel("Normalized Eigenvalue (Energy Percentage %)")
            ax.set_title("POD Eigenvalue Spectrum")
            ax.grid(True, which="both", ls="--")

            plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_pod_eigenvalues.png")
            plt.savefig(plot_filename, dpi=FIG_DPI, bbox_inches="tight")
            print(f"Saving figure {plot_filename}")

        finally:
            plt.close(fig)  # Ensure figure is closed even if error occurs

    def plot_modes(self, plot_n_modes: Optional[int] = 10, modes_per_fig: int = 1) -> None:
        """Plot the spatial POD modes.

        Visualizes the first `n_modes_to_plot` dominant spatial modes.
        Requires spatial coordinates (e.g., `self.data['x']`, `self.data['y']`) to be loaded.
        Assumes 2D spatial data that can be reshaped using `Nx` and `Ny` from `self.data`.
        Plots are saved to `self.figures_dir`.

        Args:
            plot_n_modes (int | None, optional): Number of leading spatial modes to
                plot. If ``None`` all available modes are plotted. Defaults to 10.
        """
        if self.modes.size == 0:
            print("No modes to plot. Run perform_pod() first.")
            return

        n_modes = self.modes.shape[1]
        if plot_n_modes is not None:
            n_modes = min(plot_n_modes, n_modes, self.n_modes_save)
        if n_modes == 0:
            print("No modes available to plot.")
            return

        Nx = self.data.get("Nx", int(np.sqrt(self.modes.shape[0])))
        Ny = self.data.get("Ny", int(np.sqrt(self.modes.shape[0])))
        is_2d_plot = (self.modes.shape[0] == Nx * Ny) and (Nx > 1 and Ny > 1)
        x_coords = self.data.get("x", np.arange(Nx))
        y_coords = self.data.get("y", np.arange(Ny))

        fig_aspect = get_fig_aspect_ratio(self.data)

        # Determine if plotting 1D or 2D modes
        is_2d_plot = (self.modes.shape[0] == Nx * Ny) and (Nx > 1 and Ny > 1)

        var_name = self.data.get("metadata", {}).get("var_name", "q")

        for start in range(0, n_modes, modes_per_fig):
            end = min(start + modes_per_fig, n_modes)
            ncols = end - start
            if not is_2d_plot:
                print("plot_modes currently supports 2-D fields only.")
                return

            fig, axes = plt.subplots(
                1,
                ncols,
                figsize=(4 * ncols * fig_aspect, 4),
                squeeze=False,
            )
            for idx, mode_idx in enumerate(range(start, end)):
                ax = axes[0, idx]
                mode = self.modes[:, mode_idx]
                # Reshape mode to 2D
                mode_2d = mode.reshape((Nx, Ny))
                # Get meshgrid for plotting
                if x_coords.ndim == 1 and y_coords.ndim == 1:
                    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")
                else:
                    x_mesh, y_mesh = x_coords, y_coords
                # Cylinder mask and plotting limits
                distance = np.sqrt(x_mesh**2 + y_mesh**2)
                cylinder_mask = distance <= 0.5
                # Compute levels using only values outside the cylinder
                mode_flat = mode_2d[~cylinder_mask]
                vmin = np.nanmin(mode_flat)
                vmax = np.nanmax(mode_flat)
                levels = np.linspace(vmin, vmax, 21)
                # Masked array for plotting
                mode_plot = np.ma.array(mode_2d, mask=cylinder_mask)
                # Plot filled contour
                cf = ax.contourf(x_mesh, y_mesh, mode_plot, levels=levels, cmap=CMAP_SEQ, extend="both")
                # Contour lines
                cs = ax.contour(x_mesh, y_mesh, mode_plot, levels=levels[::4], colors="k", linewidths=0.5, alpha=0.5)
                # Cylinder
                cylinder = plt.Circle((0, 0), 0.5, fill=True, linewidth=0.5, zorder=3, facecolor="lightgray", edgecolor="black")
                ax.add_patch(cylinder)
                # Labels and aspect
                ax.set_xlabel(r"$x/D$")
                ax.set_ylabel(r"$y/D$")
                ax.set_aspect("equal", "box")
                ax.set_xlim(np.min(x_coords), np.max(x_coords))
                ax.set_ylim(np.min(y_coords), np.max(y_coords))
                ax.grid(True, linestyle="--", alpha=0.3)
                # Calculate energy and cumulative energy
                if self.eigenvalues is not None and len(self.eigenvalues) > mode_idx:
                    total_energy = np.sum(self.eigenvalues)
                    energy_pct = 100.0 * self.eigenvalues[mode_idx] / total_energy
                    cum_energy_pct = 100.0 * np.sum(self.eigenvalues[: mode_idx + 1]) / total_energy
                    title_str = f"POD Mode {mode_idx + 1} [{var_name}] | Energy: {energy_pct:.2f}% | Cumulative: {cum_energy_pct:.2f}%"
                else:
                    title_str = f"POD Mode {mode_idx + 1} [{var_name}]"
                ax.set_title(title_str)
                # Colorbar
                fig.colorbar(cf, ax=ax, format="%.2f")

                fig.tight_layout()
                # Save figure as PNG with dpi=FIG_DPI
                plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_pod_mode_{start + 1}_to_{end}.png")
                plt.savefig(plot_filename, dpi=FIG_DPI)
                plt.close(fig)
                print(f"Saving figure {plot_filename}")
                fig.tight_layout()

    def plot_modes_pair_detailed(self, plot_n_modes: int = 4, cmap=CMAP_SEQ) -> None:
        """Plot modes in pairs with an additional magnitude row (2×2 per figure).

        Produces figures where the top row contains the raw spatial fields for a
        pair of modes (e.g. mode 1 and 2) and the bottom row contains their
        magnitudes.  Designed to replicate the 4-panel style the user wants for
        `pod_mode_1_to_2.png`, `pod_mode_3_to_4.png`, etc.
        """
        if self.modes.size == 0:
            print("No modes to plot. Run perform_pod() first.")
            return

        n_modes = min(plot_n_modes, self.modes.shape[1], self.n_modes_save)
        if n_modes == 0:
            print("No modes available to plot.")
            return

        Nx = self.data.get("Nx", int(np.sqrt(self.modes.shape[0])))
        Ny = self.data.get("Ny", int(np.sqrt(self.modes.shape[0])))
        is_2d_plot = (self.modes.shape[0] == Nx * Ny) and (Nx > 1 and Ny > 1)
        x_coords = self.data.get("x", np.arange(Nx))
        y_coords = self.data.get("y", np.arange(Ny))
        fig_aspect = get_fig_aspect_ratio(self.data)
        var_name = self.data.get("metadata", {}).get("var_name", "q")

        # --- Colormap setup for raw and magnitude plots ---
        import matplotlib.cm as cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # Hold contour handles for shared colorbars (set in first loop iteration)
        first_cf = None  # For raw (signed) mode fields

        for start in range(0, n_modes, 2):
            end = min(start + 2, n_modes)
            ncols = end - start
            if not is_2d_plot:
                print("plot_modes_pair_detailed currently supports 2-D fields only.")
                return

            fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols * fig_aspect, 4), squeeze=False, constrained_layout=True)

            total_energy = np.sum(self.eigenvalues)
            for idx, mode_idx in enumerate(range(start, end)):
                # ------------------ Only plot top row: raw mode ------------------
                ax = axes[0, idx]
                mode_vec = self.modes[:, mode_idx]
                mode_2d = mode_vec.reshape((Nx, Ny))
                if x_coords.ndim == 1 and y_coords.ndim == 1:
                    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")
                else:
                    x_mesh, y_mesh = x_coords, y_coords
                dist = np.sqrt(x_mesh**2 + y_mesh**2)
                mask = dist <= 0.5
                field = np.ma.array(mode_2d, mask=mask)
                vmax = np.max(np.abs(field))
                levels = np.linspace(-vmax, vmax, 21)

                cf = ax.contourf(
                    x_mesh,
                    y_mesh,
                    field,
                    levels=levels,
                    cmap=CMAP_DIV,  # diverging colormap for signed mode field
                    extend="both",
                )
                if first_cf is None:
                    first_cf = cf
                ax.add_patch(plt.Circle((0, 0), 0.5, fill=True, facecolor="lightgray", edgecolor="black", linewidth=0.5))
                ax.set_aspect("equal", "box")
                ax.set_xlim(np.min(x_coords), np.max(x_coords))
                ax.set_ylim(np.min(y_coords), np.max(y_coords))
                ax.set_xlabel(r"$x/D$")
                ax.set_ylabel(r"$y/D$")
                ax.grid(True, linestyle="--", alpha=0.3)

                # Add individual small colorbar inside the data area (upper right)
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes

                cax = inset_axes(ax, width="15%", height="6%", loc="upper right", borderpad=3)
                cb = fig.colorbar(cf, cax=cax, orientation="horizontal", format="%.2f")
                cb.ax.tick_params(labelsize=8, pad=1, colors="black")
                cb.ax.xaxis.set_ticks_position("top")
                cb.ax.xaxis.set_label_position("top")
                # Set custom ticks: min, 0, max
                vmin, vmax = -vmax, vmax
                cb.set_ticks([vmin, 0, vmax])
                cb.set_ticklabels([f"{vmin:.2f}", "0", f"{vmax:.2f}"])
                # Make colorbar background semi-transparent
                cax.patch.set_facecolor("black")
                cax.patch.set_alpha(0.7)

                energy_pct = 100.0 * self.eigenvalues[mode_idx] / total_energy
                cum_pct = 100.0 * np.sum(self.eigenvalues[: mode_idx + 1]) / total_energy
                ax.set_title(f"Mode {mode_idx + 1}\nE={energy_pct:.2f}%  Cum={cum_pct:.2f}%", fontsize=8, pad=20)
            plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_pod_mode_{start + 1}_to_{end}.png")
            plt.savefig(plot_filename, dpi=FIG_DPI)
            plt.close(fig)
            print(f"Saving figure {plot_filename}")

    def plot_modes_grid(self, energy_threshold: float = 99.5, cmap=CMAP_DIV) -> None:
        """Plot spatial POD modes side-by-side up to a cumulative energy threshold.

        This produces a single figure containing all modes required to reach the
        specified cumulative energy percentage (default 99.5%).  Each mode is
        displayed with a diverging colormap so positive and negative regions
        are easily distinguished.  The subplot title indicates the mode number,
        its individual energy contribution, and the cumulative energy captured
        up to that mode.  Axes limits, cylinder overlay, and other style
        choices mirror those used in the DMD detailed mode plots so that the
        two decompositions can be compared directly.
        """
        # Preconditions – ensure POD has been performed
        if self.modes.size == 0 or self.eigenvalues.size == 0:
            print("No POD modes/eigenvalues to plot. Run perform_pod() first.")
            return

        total_energy = np.sum(self.eigenvalues)
        cumulative_pct = np.cumsum(self.eigenvalues) / total_energy * 100.0
        # Number of modes needed to reach threshold (inclusive)
        n_modes_plot = int(np.searchsorted(cumulative_pct, energy_threshold, side="right")) + 1
        n_modes_plot = min(n_modes_plot, self.n_modes_save, self.modes.shape[1])
        if n_modes_plot <= 0:
            print("Energy threshold too low – nothing to plot.")
            return

        # Spatial grid information
        Nx = self.data.get("Nx", int(np.sqrt(self.modes.shape[0])))
        Ny = self.data.get("Ny", int(np.sqrt(self.modes.shape[0])))
        is_2d_plot = (self.modes.shape[0] == Nx * Ny) and (Nx > 1 and Ny > 1)
        x_coords = self.data.get("x", np.arange(Nx))
        y_coords = self.data.get("y", np.arange(Ny))
        fig_aspect = get_fig_aspect_ratio(self.data)
        var_name = self.data.get("metadata", {}).get("var_name", "q")

        # Always use two columns so rows list modes sequentially (1-2, 3-4, …)
        ncols = 2 if n_modes_plot > 1 else 1
        nrows = int(np.ceil(n_modes_plot / ncols))

        # Create figure with constrained_layout for better spacing
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols * fig_aspect, 4 * nrows), squeeze=False, constrained_layout=True)

        # Import make_axes_locatable for better colorbar placement
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # Track the first contourf for colorbar
        first_cf = None

        # Plot each mode
        for k in range(n_modes_plot):
            row, col = divmod(k, ncols)
            ax = axes[row][col]
            mode_vec = self.modes[:, k]

            if is_2d_plot:
                # Reshape mode to 2D grid
                mode_2d = mode_vec.reshape((Nx, Ny))

                # Create meshgrid for plotting
                if x_coords.ndim == 1 and y_coords.ndim == 1:
                    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")
                else:
                    x_mesh, y_mesh = x_coords, y_coords

                # Mask interior of cylinder (radius 0.5)
                distance = np.sqrt(x_mesh**2 + y_mesh**2)
                cylinder_mask = distance <= 0.5
                mode_plot = np.ma.array(mode_2d, mask=cylinder_mask)

                # Calculate contour levels with symmetric diverging scale
                vmax = np.max(np.abs(mode_plot))
                levels = np.linspace(-vmax, vmax, 21)

                # Plot contours
                cf = ax.contourf(x_mesh, y_mesh, mode_plot, levels=levels, cmap=cmap, extend="both")

                # Add cylinder overlay
                cyl = plt.Circle((0, 0), 0.5, fill=True, facecolor="lightgray", edgecolor="black", linewidth=0.5)
                ax.add_patch(cyl)

                # Set axis properties
                ax.set_aspect("equal", "box")
                ax.set_xlim(np.min(x_coords), np.max(x_coords))
                ax.set_ylim(np.min(y_coords), np.max(y_coords))
                ax.set_xlabel(r"$x/D$")
                ax.set_ylabel(r"$y/D$")
                ax.grid(True, linestyle="--", alpha=0.3)

                # Add individual small colorbar inside the data area (upper right)
                pos = ax.get_position()
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes

                cax = inset_axes(ax, width="15%", height="6%", loc="upper right", borderpad=3)
                cb = fig.colorbar(cf, cax=cax, orientation="horizontal", format="%.2f")
                cb.ax.tick_params(labelsize=8, pad=1, colors="black")
                cb.ax.xaxis.set_ticks_position("top")
                cb.ax.xaxis.set_label_position("top")
                # Set custom ticks: min, 0, max
                vmin, vmax = -vmax, vmax
                cb.set_ticks([vmin, 0, vmax])
                cb.set_ticklabels([f"{vmin:.2f}", "0", f"{vmax:.2f}"])
                # Make colorbar background semi-transparent
                cax.patch.set_facecolor("black")
                cax.patch.set_alpha(0.7)

                # Add title with energy information
                energy_pct = 100.0 * self.eigenvalues[k] / total_energy
                cum_pct = cumulative_pct[k]
                ax.set_title(f"Mode {k + 1}\nE={energy_pct:.2f}%  Cum={cum_pct:.2f}%", fontsize=8, pad=20)
            else:
                # 1D mode plotting (fallback)
                ax.plot(mode_vec)
                ax.set_xlabel("Spatial Index")
                ax.set_ylabel(f"{var_name} amplitude")
                ax.set_title(f"Mode {k + 1}")

        # Hide any extra subplots
        for idx in range(n_modes_plot, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r][c].axis("off")

        # Add title and save
        fig.suptitle(f"POD Modes up to {energy_threshold:.1f}% cumulative energy ({n_modes_plot} modes)", fontsize=12)

        plot_filename = os.path.join(
            self.figures_dir,
            f"{self.data_root}_pod_modes_grid_{energy_threshold:.1f}perc.png",
        )
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        print(f"Saving figure {plot_filename}")

    def plot_time_coefficients(self, n_coeffs_to_plot=2, n_snapshots_plot=None, L=1.0, U=1.0):
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
            L (float, optional): Characteristic length for Strouhal number conversion. Defaults to ``1.0``.
            U (float, optional): Characteristic velocity for Strouhal number conversion. Defaults to ``1.0``.

        The periodogram axis is always shown in terms of Strouhal number.  The
        frequencies are converted according to ``freqs_st = freqs * L / U``.  With
        the default ``L = U = 1`` the plot displays the unscaled frequencies but
        labeled as Strouhal numbers.
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

        # Prefer explicit time vector if provided in data
        if "t" in self.data and len(self.data["t"]) >= n_snapshots_plot:
            time_vector = self.data["t"][:n_snapshots_plot]
        else:
            time_vector = np.arange(n_snapshots_plot) * self.data.get("dt", 1.0)

        plt.figure(figsize=(12, 3 * n_coeffs_to_plot))
        for i in range(n_coeffs_to_plot):
            plt.subplot(n_coeffs_to_plot, 2, 2 * i + 1)
            coeff = self.time_coefficients[:n_snapshots_plot, i]
            plt.plot(time_vector, coeff, linewidth=1.5)
            plt.xlabel("Time")
            plt.ylabel(f"Amplitude Mode {i + 1}")
            plt.title(f"Temporal Coefficient for POD Mode {i + 1}")
            plt.grid(True, linestyle=":")
            plt.xlim(time_vector.min(), time_vector.max())

            plt.subplot(n_coeffs_to_plot, 2, 2 * i + 2)
            freqs, psd = periodogram_rfft(coeff, self.fs)
            peak_freqs, peak_psd = find_peaks(freqs, psd)
            
            if L is not None and U is not None:
                freqs = freqs * L / U
            xlabel = "Strouhal Number (St)"
            plt.semilogy(freqs, psd)
            if peak_freqs.size > 0:
                plt.plot(peak_freqs, peak_psd, "o")
                for pf, pv in zip(peak_freqs, peak_psd):
                    plt.text(pf, pv, f"{pf:.2f}", fontsize=8, ha="left", va="bottom")
            plt.xscale("log")
            plt.xlim(1e-1, 1e4)
            plt.ylim(1e-6, None)
            plt.xlabel(xlabel)
            plt.ylabel("PSD")
            plt.title(f"Periodogram Mode {i + 1}")
            plt.grid(True, linestyle=":")

        plt.tight_layout()
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_pod_time_coeffs.png")
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close()
        print(f"Saving figure {plot_filename}")

    def check_mode_pair_phase(self, start_mode: int = 1, threshold: float = 0.9):
        """Identify mode pairs with strongly correlated time coefficients.

        Starting from ``start_mode`` (1-indexed), iterate through the saved POD
        modes and compute the Pearson correlation coefficient between the time
        coefficients of mode ``i`` and candidate mode ``j``.  If the absolute
        value of the correlation exceeds ``threshold`` the pair ``(i, j)`` is
        considered a valid phase pair and is yielded.  If the correlation is
        below the threshold the search continues with ``j`` incremented until a
        suitable partner for mode ``i`` is found or the available modes are
        exhausted.

        Parameters
        ----------
        start_mode : int, optional
            First mode index to test (1-indexed).  Defaults to ``1``.
        threshold : float, optional
            Minimum absolute correlation required to accept a pair.  A value of
            ``1.0`` would require perfectly correlated time coefficients while a
            value of ``0`` would accept any pair.  Defaults to ``0.9``.

        Yields
        ------
        tuple[int, int]
            Mode index pairs that satisfy the correlation criterion.
        """

        if self.time_coefficients.size == 0:
            print("No time coefficients available. Run perform_pod() first.")
            return

        n_modes = self.time_coefficients.shape[1]
        i = start_mode
        while i < n_modes:
            found = False
            for j in range(i + 1, n_modes + 1):
                coeff_i = self.time_coefficients[:, i - 1]
                coeff_j = self.time_coefficients[:, j - 1]
                corr = np.corrcoef(coeff_i, coeff_j)[0, 1]
                if np.abs(corr) >= threshold:
                    print(f"Found correlated pair ({i}, {j}) with r={corr:.3f}")
                    yield (i, j)
                    i = j + 1
                    found = True
                    break
            if not found:
                print(f"No correlated partner found for mode {i}")
                i += 1

    def plot_mode_pair_phase(self, start_mode: int = 1, threshold: float = 0.9):
        """Plot phase portraits of automatically detected mode pairs.

        Mode pairs are identified using :meth:`check_mode_pair_phase`.  For each
        accepted pair the temporal coefficients of the two modes are plotted
        against each other to visualize their phase relationship.

        Parameters
        ----------
        start_mode : int, optional
            Initial mode index to search from.  Defaults to ``1``.
        threshold : float, optional
            Correlation threshold passed to :meth:`check_mode_pair_phase`.
            Defaults to ``0.9``.
        """

        pairs = list(self.check_mode_pair_phase(start_mode=start_mode, threshold=threshold))
        if not pairs:
            print("No mode pairs met the correlation threshold.")
            return

        for i, j in pairs:
            coeff_i = self.time_coefficients[:, i - 1]
            coeff_j = self.time_coefficients[:, j - 1]
            plt.figure(figsize=(5, 5))
            plt.plot(coeff_i, coeff_j, "o-", markersize=3, linewidth=0.8)
            plt.xlabel(f"Coefficient {i}")
            plt.ylabel(f"Coefficient {j}")
            plt.title(f"POD Phase Portrait Modes {i} & {j}")
            plt.grid(True)
            fname = os.path.join(self.figures_dir, f"{self.data_root}_pod_phase_pair_{i}_{j}.png")
            plt.savefig(fname, dpi=FIG_DPI)
            plt.close()
            print(f"Saving figure {fname}")

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
        # Annotate cumulative curve with mode numbers
        for idx, (x, y) in enumerate(zip(mode_indices, cumulative_energy)):
            plt.text(x, y, f" {idx + 1}", fontsize=7, va="bottom")
        plt.xlabel("Number of Modes")
        plt.ylabel("Cumulative Energy (%)")
        plt.title("Cumulative Energy of POD Modes")
        plt.grid(True, which="both", ls="--")
        plt.ylim(0, 105)  # Show up to 100% or slightly more
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_pod_cumulative_energy.png")
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close()
        print(f"Saving figure {plot_filename}")

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
        # Annotate each reconstruction error point with mode number
        for idx, (x, y) in enumerate(zip(mode_indices, reconstruction_errors)):
            plt.text(x, y, f" {idx + 1}", fontsize=7, va="bottom")
        plt.xlabel("Number of Modes Used for Reconstruction")
        plt.ylabel("Reconstruction Error (%)")
        plt.title("Data Reconstruction Error vs. Number of POD Modes")
        plt.grid(True, which="both", ls="--")
        plt.yscale("log")  # Error often drops off exponentially
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_pod_reconstruction_error.png")
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close()
        print(f"Saving figure {plot_filename}")

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
        num_snapshots, num_space_points = data_mean_removed.shape

        if snapshot_indices_to_plot is None:
            snapshot_indices_to_plot = [0, num_snapshots // 2, num_snapshots - 1]
            # Ensure indices are unique and within bounds, especially for small datasets
            snapshot_indices_to_plot = sorted(list(set(idx for idx in snapshot_indices_to_plot if 0 <= idx < num_snapshots)))
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
        Nx = self.data.get("Nx", int(np.sqrt(num_space_points)))
        Ny = self.data.get("Ny", int(np.sqrt(num_space_points)))
        is_2d_plot = (num_space_points == Nx * Ny) and (Nx > 1 and Ny > 1)
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
                # TODO: Implement 2D plotting for original snapshot
                pass
            else:
                # TODO: Implement 1D plotting for original snapshot
                pass

    def check_spatial_mode_orthogonality(self, tolerance=1e-9):
        """Check the orthogonality of spatial modes with respect to weights W.

                Verifies that `Modes.T @ W_diag @ Modes` is close to the identity matrix,
        {{ ... }}
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
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close()
        print(f"Saving figure {plot_filename}")
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
        num_snapshots = self.data["Ns"]
        n_saved_coeffs = self.time_coefficients.shape[1]

        # Expected matrix based on POD theory for snapshot POD temporal eigenvectors
        # (Psi_temp.T @ Psi_temp) should be Identity if Psi_temp are normalized eigenvectors of K_t.
        # self.time_coefficients = Psi_temp * sqrt(eigenvalues_temp * Ns)
        # So (1/Ns) * self.time_coefficients.T @ self.time_coefficients =
        # (1/Ns) * sqrt(L*Ns) * Psi_temp.T @ Psi_temp * sqrt(L*Ns) = L (diag(eigenvalues))
        ortho_check_matrix = (1.0 / num_snapshots) * (self.time_coefficients.T @ self.time_coefficients)
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
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close()
        print(f"Saving figure {plot_filename}")
        return is_pseudo_orthogonal

    def run_analysis(self, plot_n_modes_spatial=4, plot_n_coeffs_time=5, plot_snapshot_indices=None, modes_for_reconstruction=None, check_orthogonality=False):
        """
        Main entry point for running POD analysis and plotting.
            check_orthogonality (bool, optional): If True, perform and print orthogonality checks.
                                                Defaults to True.
        """
        print(f"🔎 Starting POD analysis for {os.path.basename(self.file_path)}")
        start_total_time = time.time()

        # Load data and calculate weights via BaseAnalyzer's run method.
        # compute_fft=False because POD is time-domain.
        super().run(compute_fft=False)

        # Perform POD
        self.perform_pod()

        # Identify correlated mode pairs only when plotting

        # Save results
        self.save_results()  # This already calls super().save_results()

        # Plotting
        self.plot_eigenvalues()
        # Detailed 4-panel mode plots (pairs with magnitude)
        self.plot_modes_pair_detailed(plot_n_modes=plot_n_modes_spatial)
        # Phase portraits for correlated pairs
        self.plot_mode_pair_phase()
        # New: comprehensive grid of modes up to cumulative energy threshold for easy DMD comparison
        self.plot_modes_grid(energy_threshold=99.5)
        self.plot_time_coefficients(n_coeffs_to_plot=plot_n_coeffs_time)
        self.plot_cumulative_energy()
        self.plot_reconstruction_error()
        self.plot_reconstruction_comparison(snapshot_indices_to_plot=plot_snapshot_indices, modes_for_reconstruction=modes_for_reconstruction)

        if check_orthogonality:
            self.check_spatial_mode_orthogonality()
            self.check_temporal_coefficient_orthogonality()

        end_total_time = time.time()
        print(f"\nPOD analysis and plotting completed successfully in {end_total_time - start_total_time:.2f} seconds.")
        print_summary("POD", self.results_dir, self.figures_dir)


# Example usage when the script is run directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run POD analysis")
    parser.add_argument("--config", help="Path to JSON/YAML configuration file", default=None)
    parser.add_argument("--prep", action="store_true", help="Load data and prepare for POD")
    parser.add_argument("--compute", action="store_true", help="Perform POD and save results")
    parser.add_argument("--plot", action="store_true", help="Generate default plots")
    args = parser.parse_args()

    # If no arguments are provided, run the full analysis (compute + plot)
    if not any([args.prep, args.compute, args.plot]):
        args.compute = True
        args.plot = True

    print_optimization_status()

    if args.config:
        from configs import load_config

        load_config(args.config)

    # --- Configuration ---
    # data_file = "./data/jetLES_small.mat"  # Updated data path
    # data_file = "./data/jetLES.mat"  # Path to your data file
    # data_file = "./data/cavityPIV.mat"  # Path to your data file
    data_file = "./data/snp1-947_u.npz"  # Path to your data file

    n_modes_to_save_main = 10  # Number of POD modes to save
    n_modes_to_plot_spatial_main = 4  # Number of spatial modes to visualize
    n_coeffs_to_plot_time_main = 5  # Number of temporal coefficients to visualize
    # ---------------------

    # Loop over all available fields in the consolidated npz
    from data_interface import DNamiXNPZLoader

    loader = DNamiXNPZLoader()
    available_fields = loader.get_available_fields(data_file)
    print(f"Available fields in {data_file}: {available_fields}")

    for field in available_fields:
        print(f"\n===== Running POD for variable: {field} =====")
        data = loader.load(data_file, field=field)
        # Set up variable-specific result and figure directories
        results_dir = os.path.join(RESULTS_DIR_POD, field)
        figures_dir = os.path.join(FIGURES_DIR_POD, field)
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        analyzer = PODAnalyzer(
            file_path=data_file,
            results_dir=results_dir,
            figures_dir=figures_dir,
            data_loader=lambda fp: loader.load(fp, field=field),
            n_modes_save=n_modes_to_save_main,
            spatial_weight_type="uniform",
        )
        analyzer.data = data
        analyzer.analysis_type = f"pod_{field}"
        if args.compute:
            analyzer.run_analysis(
                plot_n_modes_spatial=n_modes_to_plot_spatial_main,
                plot_n_coeffs_time=n_coeffs_to_plot_time_main,
            )
        elif args.plot:
            analyzer.load_results()
            analyzer.plot_eigenvalues()
            analyzer.plot_modes_pair_detailed(plot_n_modes=n_modes_to_plot_spatial_main)
            analyzer.plot_modes_grid(energy_threshold=99.7)
            analyzer.plot_time_coefficients(n_coeffs_to_plot=n_coeffs_to_plot_time_main)
            analyzer.plot_cumulative_energy()
            analyzer.plot_reconstruction_error()
            analyzer.plot_reconstruction_comparison()
