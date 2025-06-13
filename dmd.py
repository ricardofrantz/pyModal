#!/usr/bin/env python3
"""
Dynamic Mode Decomposition (exact DMD) implementation.

This follows the style of :mod:`pod.py` but implements the standard
exact DMD algorithm as in the PyDMD project.
"""

# Standard library imports
import argparse
import os
import h5py
import matplotlib
matplotlib.use('Agg')
from typing import Optional
import matplotlib.pyplot as plt
import warnings
# Suppress contour warnings when no levels can be plotted
warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
import numpy as np

from configs import (
    CMAP_DIV,
    CMAP_SEQ,
    FIG_DPI,
    FIGURES_DIR_DMD,
    RESULTS_DIR_DMD,
)

from utils import (
    BaseAnalyzer,
    get_fig_aspect_ratio,
    make_result_filename,
    print_summary,
)

# Try to import DNamiXNPZLoader for npz support
try:
    from data_interface import DNamiXNPZLoader
except ImportError:
    DNamiXNPZLoader = None



class DMDAnalyzer(BaseAnalyzer):
    """Exact Dynamic Mode Decomposition analyzer."""

    def __init__(
        self,
        file_path,
        results_dir=RESULTS_DIR_DMD,
        figures_dir=FIGURES_DIR_DMD,
        data_loader=None,
        spatial_weight_type="auto",
        n_modes_save=10,
        use_parallel=True,
    ):
        super().__init__(
            file_path=file_path,
            nfft=1,
            overlap=0.0,
            results_dir=results_dir,
            figures_dir=figures_dir,
            data_loader=data_loader,
            spatial_weight_type=spatial_weight_type,
            use_parallel=use_parallel,
        )
        self.n_modes_save = n_modes_save
        self.modes = np.array([])
        self.eigenvalues = np.array([])
        self.time_coefficients = np.array([])
        self.analysis_type = "dmd"
        self.temporal_mean = np.array([])
        # Store modal amplitudes (|b|) after perform_dmd()
        self.amplitudes = np.array([])


    def perform_dmd(self):
        """Compute DMD modes, eigenvalues and time coefficients."""
        if "q" not in self.data:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")

        q = self.data["q"]
        n_snapshots = q.shape[0]
        x = q[:-1, :].T
        y = q[1:, :].T

        u, s, vh = np.linalg.svd(x, full_matrices=False)
        r = min(self.n_modes_save, len(s))
        u_r = u[:, :r]
        s_r = np.diag(s[:r])
        v_r = vh.conj().T[:, :r]

        atilde = u_r.conj().T @ y @ v_r @ np.linalg.inv(s_r)
        eigvals, w = np.linalg.eig(atilde)

        modes = y @ v_r @ np.linalg.inv(s_r) @ w

        b = np.linalg.pinv(modes) @ q[0, :]
        t = np.arange(n_snapshots)
        time_dynamics = (b[:, None] * eigvals[:, None] ** t).T

        idx = np.argsort(np.abs(eigvals))[::-1]
        self.eigenvalues = eigvals[idx][: self.n_modes_save]
        self.modes = modes[:, idx][:, : self.n_modes_save]
        self.time_coefficients = time_dynamics[:, idx][:, : self.n_modes_save]
        # Normalized amplitudes of the sorted modes
        self.amplitudes = np.abs(b[idx][: self.n_modes_save])

    def save_results(self, filename=None):
        """Save DMD results to an HDF5 file."""
        if not filename:
            filename = make_result_filename(
                self.data_root,
                self.nfft,
                self.overlap,
                self.data.get("Ns", 0),
                self.analysis_type,
            )
        path = os.path.join(self.results_dir, filename)
        with h5py.File(path, "w") as f:
            f.attrs.update(self._get_metadata())
            f.create_dataset("eigenvalues", data=self.eigenvalues, compression="gzip")
            f.create_dataset("modes", data=self.modes, compression="gzip")
            f.create_dataset("time_coefficients", data=self.time_coefficients, compression="gzip")
            f.create_dataset("amplitudes", data=self.amplitudes, compression="gzip")
            f.create_dataset("x", data=self.data["x"], compression="gzip")
            f.create_dataset("y", data=self.data["y"], compression="gzip")
        print(f"DMD results saved to {path}")

    def load_results(self, filename=None):
        """Load DMD results from an HDF5 file."""
        if not filename:
            filename = make_result_filename(
                self.data_root,
                self.nfft,
                self.overlap,
                self.data.get("Ns", 0),
                self.analysis_type,
            )
        path = os.path.join(self.results_dir, filename)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"DMD results file not found: {path}")
        
        with h5py.File(path, "r") as f:
            self.eigenvalues = f["eigenvalues"][:]
            self.modes = f["modes"][:]
            self.time_coefficients = f["time_coefficients"][:]
            # Load amplitudes if available (backward compatibility)
            if "amplitudes" in f:
                self.amplitudes = f["amplitudes"][:]
            else:
                # Calculate amplitudes from eigenvalues for backward compatibility
                self.amplitudes = np.abs(self.eigenvalues)
            # Load spatial coordinates if they exist
            if "x" in f:
                self.data["x"] = f["x"][:]
            if "y" in f:
                self.data["y"] = f["y"][:]
        print(f"DMD results loaded from {path}")

    def plot_eigenvalues(self):
        """Plot DMD eigenvalues in the complex plane."""
        if self.eigenvalues.size == 0:
            print("No eigenvalues to plot.")
            return
        plt.figure(figsize=(6, 6))
        plt.plot(self.eigenvalues.real, self.eigenvalues.imag, "bo")
        circle = plt.Circle((0, 0), 1.0, color="green", fill=False, linestyle="--")
        ax = plt.gca()
        ax.add_artist(circle)
        ax.axhline(0, color="k", linestyle="--", linewidth=0.5)
        ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
        ax.set_aspect("equal")
        plt.xlabel("Real part")
        plt.ylabel("Imaginary part")
        plt.title("DMD Eigenvalues (Complex Plane)")
        fname = os.path.join(self.figures_dir, f"{self.data_root}_dmd_eigenvalues.png")
        plt.savefig(fname, dpi=FIG_DPI)
        plt.close()
        print(f"Saving figure {fname}")

    def plot_eigenspectra(self):
        """Create composite spectra figure: eigenvalues circle, amplitude vs frequency and growth rate."""
        if self.eigenvalues.size == 0 or self.amplitudes.size == 0:
            print("No eigenvalue data to plot. Run perform_dmd() first.")
            return
        dt = self.data.get("dt", 1.0)
        eigvals = self.eigenvalues
        amps = self.amplitudes
        amps_norm = amps / np.max(amps)
        freq = np.angle(eigvals) / (2 * np.pi * dt)
        growth = np.log(np.abs(eigvals)) / dt

        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
        ax_complex = fig.add_subplot(gs[0, :])
        ax_freq = fig.add_subplot(gs[1, 0])
        ax_growth = fig.add_subplot(gs[1, 1])

        # Complex eigenvalue plot
        ax_complex.plot(eigvals.real, eigvals.imag, "o", mfc="none", mec="brown")
        # Annotate every eigenvalue with its mode number; mark mean explicitly
        for k, lam in enumerate(eigvals):
            label = f"{k+1}"
            if np.isclose(lam, 1+0j, atol=1e-3):
                label += " (mean)"
            ax_complex.text(lam.real, lam.imag, f" {label}", fontsize=7, color="black")
        idx_mean = int(np.argmin(np.abs(eigvals - 1)))
        ax_complex.text(eigvals.real[idx_mean], eigvals.imag[idx_mean], "  mean", color="red", fontsize=8, va="center")
        # Annotate first few oscillatory modes with frequency
        for k in range(min(4, len(eigvals))):
            if k == idx_mean:
                continue
            ax_complex.text(eigvals.real[k], eigvals.imag[k], f"  f={freq[k]:.2f}", fontsize=7, color="black")
        unit_circle = plt.Circle((0, 0), 1.0, color="brown", fill=False, linewidth=1.0)
        ax_complex.add_patch(unit_circle)
        ax_complex.axhline(0.0, color="k", linestyle="--", linewidth=0.5)
        ax_complex.axvline(0.0, color="k", linestyle="--", linewidth=0.5)
        ax_complex.set_xlabel(r"$\mathrm{Re}(\lambda)$")
        ax_complex.set_ylabel(r"$\mathrm{Im}(\lambda)$")
        ax_complex.set_aspect("equal")
        ax_complex.set_title("DMD eigenvalues")

        # Amplitude vs frequency
        ax_freq.stem(freq, amps_norm, linefmt="brown", markerfmt="ro", basefmt=" ", use_line_collection=True)
        for k, (x, y) in enumerate(zip(freq, amps_norm)):
            ax_freq.text(x, y, f" {k+1}", fontsize=6, rotation=45, va="bottom")
        ax_freq.set_xlabel("frequency")
        ax_freq.set_ylabel("normalized amplitude")
        ax_freq.set_yscale("log")
        ax_freq.set_title("Amplitude vs frequency")

        # Amplitude vs growth rate
        ax_growth.stem(growth, amps_norm, linefmt="brown", markerfmt="ro", basefmt=" ", use_line_collection=True)
        for k, (x, y) in enumerate(zip(growth, amps_norm)):
            ax_growth.text(x, y, f" {k+1}", fontsize=6, rotation=45, va="bottom")
        ax_growth.set_xlabel("growth rate")
        ax_growth.set_yscale("log")
        ax_growth.set_title("Amplitude vs growth rate")

        fig.tight_layout()
        fname_spec = os.path.join(self.figures_dir, f"{self.data_root}_dmd_eigenspectra.png")
        fig.savefig(fname_spec, dpi=FIG_DPI)
        plt.close(fig)
        print(f"Saving figure {fname_spec}")

    def plot_modes_detailed(
        self,
        plot_n_modes: int = 8,
        zero_phase_ref: bool = False,
        unwrap_phase: bool = False,
        ref_method: str = "max",
    ):
        """Plot real, imaginary, magnitude, and phase of several modes in a 4-row grid."""
        if self.modes.size == 0:
            print("No modes to plot. Run perform_dmd() first.")
            return
        n_modes = min(plot_n_modes, self.modes.shape[1])
        if n_modes == 0:
            print("No modes available to plot.")
            return

        nx = self.data.get("Nx", int(np.sqrt(self.modes.shape[0])))
        ny = self.data.get("Ny", int(np.sqrt(self.modes.shape[0])))
        if self.modes.shape[0] != nx * ny or nx <= 1 or ny <= 1:
            print("Detailed mode plotting supports 2D data only.")
            return

        x_coords = self.data.get("x", np.arange(nx))
        y_coords = self.data.get("y", np.arange(ny))
        fig_aspect = get_fig_aspect_ratio(self.data)
        var_name = self.data.get("metadata", {}).get("var_name", "q")
        # Compute mode frequencies (Hz) for annotation purposes
        dt = self.data.get("dt", 1.0)
        eigvals_subset = self.eigenvalues[:n_modes]
        freq = np.angle(eigvals_subset) / (2 * np.pi * dt)

        fig, axes = plt.subplots(4, n_modes, figsize=(3 * n_modes * fig_aspect, 12), squeeze=False)
        row_labels = ["real", "imaginary", "magnitude", "phase"]
        cmaps = [CMAP_DIV, CMAP_DIV, CMAP_SEQ, "twilight"]

        if x_coords.ndim == 1 and y_coords.ndim == 1:
            x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")
        else:
            x_mesh, y_mesh = x_coords, y_coords
        distance = np.sqrt(x_mesh**2 + y_mesh**2)
        cylinder_mask = distance <= 0.5

        for m in range(n_modes):
            vec = self.modes[:, m]
            # Phase processing with optional reference and unwrapping
            phase_arr = np.angle(vec)
            if zero_phase_ref:
                if ref_method == "max":
                    phase0 = phase_arr[np.argmax(np.abs(vec))]
                else:  # 'mean'
                    phase0 = np.mean(phase_arr)
                phase_arr = (phase_arr - phase0 + np.pi) % (2 * np.pi) - np.pi
            if unwrap_phase:
                phase_arr = np.unwrap(phase_arr)
            comps = [vec.real, vec.imag, np.abs(vec), phase_arr]
            for r, comp in enumerate(comps):
                ax = axes[r, m]
                comp2d = comp.reshape((nx, ny))
                comp_plot = np.ma.array(comp2d, mask=cylinder_mask)
                if r == 2:
                    vmin, vmax = 0.0, np.nanmax(comp_plot)
                elif r == 3:
                    vmin, vmax = -np.pi, np.pi
                else:
                    vmin, vmax = np.nanmin(comp_plot), np.nanmax(comp_plot)
                # Ensure levels are valid and strictly increasing
                if not np.isfinite(vmin) or not np.isfinite(vmax):
                    continue  # skip if invalid
                if np.isclose(vmin, vmax):
                    vmax = vmin + 1e-12  # tiny range to allow contouring
                levels = np.linspace(vmin, vmax, 21)
                cf = ax.contourf(x_mesh, y_mesh, comp_plot, levels=levels, cmap=cmaps[r], extend="both")
                # Add line contours only if range is significant
                if vmax - vmin > 1e-12:
                    ax.contour(x_mesh, y_mesh, comp_plot, levels=levels[::4], colors="k", linewidths=0.4, alpha=0.4)
                
                # Add individual small colorbar inside the data area (upper right)
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                cax = inset_axes(ax, width="15%", height="6%", loc='upper right', borderpad=3)
                cb = fig.colorbar(cf, cax=cax, orientation='horizontal', format="%.2f")
                cb.ax.tick_params(labelsize=8, pad=1, colors='black')
                cb.ax.xaxis.set_ticks_position('top')
                cb.ax.xaxis.set_label_position('top')
                # Set custom ticks: min, 0, max (except for magnitude which starts at 0)
                if r == 2:  # magnitude
                    cb.set_ticks([0, vmax/2, vmax])
                    cb.set_ticklabels(['0', f'{vmax/2:.2f}', f'{vmax:.2f}'])
                elif r == 3:  # phase
                    cb.set_ticks([-np.pi, 0, np.pi])
                    cb.set_ticklabels(['-π', '0', 'π'])
                else:  # real and imaginary
                    cb.set_ticks([vmin, 0, vmax])
                    cb.set_ticklabels([f'{vmin:.2f}', '0', f'{vmax:.2f}'])
                # Make colorbar background semi-transparent
                cax.patch.set_facecolor('black')
                cax.patch.set_alpha(0.7)
                
                # Cylinder overlay (always)
                cylinder = plt.Circle((0, 0), 0.5, facecolor="lightgray", edgecolor="black", linewidth=0.5)
                ax.add_patch(cylinder)
                # Phase zero-line overlay
                if r == 3 and vmax - vmin > 1e-12:
                    ax.contour(x_mesh, y_mesh, comp_plot, levels=[0.0], colors="white", linewidths=0.6)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                if m == 0:
                    ax.set_ylabel(row_labels[r])
                if r == 0:
                    # Column header annotations
                    if m == 0:
                        header = "1 (mean)"
                    else:
                        header = f"{m+1} (f={freq[m]:.2f})"
                    ax.set_title(header)
        fig.tight_layout()
        fname_modes = os.path.join(self.figures_dir, f"{self.data_root}_dmd_modes_detailed_{n_modes}_{var_name}.png")
        fig.savefig(fname_modes, dpi=FIG_DPI)
        plt.close(fig)
        print(f"Saving figure {fname_modes}")

    def plot_cumulative_energy(self):
        """Plot the cumulative energy captured by DMD modes (using |eigval|^2 as proxy)."""
        if self.eigenvalues.size == 0:
            print("No eigenvalues to plot. Run perform_dmd() first.")
            return
        # Use squared modulus as 'energy' proxy
        eigvals_abs2 = np.abs(self.eigenvalues) ** 2
        cumulative_energy = np.cumsum(eigvals_abs2) / np.sum(eigvals_abs2) * 100
        mode_indices = np.arange(1, len(self.eigenvalues) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(mode_indices, cumulative_energy, "o-", linewidth=2, markersize=6)
        plt.xlabel("Number of Modes")
        plt.ylabel("Cumulative |Eigval|^2 (%)")
        plt.title("Cumulative Energy of DMD Modes (|eigval|^2)")
        plt.grid(True, which="both", ls="--")
        plt.ylim(0, 105)
        fname = os.path.join(self.figures_dir, f"{self.data_root}_dmd_cumulative_energy.png")
        plt.savefig(fname, dpi=FIG_DPI*0.8)
        plt.close()
        print(f"Saving figure {fname}")

    def plot_modes(self, plot_n_modes: Optional[int] = 10, modes_per_fig: int = 1):
        """Plot the spatial DMD modes (1D/2D, like POD)."""
        if self.modes.size == 0:
            print("No modes to plot. Run perform_dmd() first.")
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

        nx = self.data.get("Nx", int(np.sqrt(self.modes.shape[0])))
        ny = self.data.get("Ny", int(np.sqrt(self.modes.shape[0])))
        x_coords = self.data.get("x", np.arange(nx))
        y_coords = self.data.get("y", np.arange(ny))
        is_2d = self.modes.shape[0] == nx * ny and nx > 1 and ny > 1
        fig_aspect = get_fig_aspect_ratio(self.data)
        var_name = self.data.get("metadata", {}).get("var_name", "q")
        # Compute mode frequencies (Hz) for annotation purposes
        dt = self.data.get("dt", 1.0)
        eigvals_subset = self.eigenvalues[:n_modes]
        freq = np.angle(eigvals_subset) / (2 * np.pi * dt)

        for start in range(0, n_modes, modes_per_fig):
            end = min(start + modes_per_fig, n_modes)
            ncols = end - start
            if is_2d:
                fig, axes = plt.subplots(
                    1,
                    ncols,
                    figsize=(4 * ncols * fig_aspect, 4),
                    squeeze=False,
                )
            else:
                fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3), squeeze=False)
            axes = axes.ravel()
            for j, i in enumerate(range(start, end)):
                ax = axes[j]
                mode = self.modes[:, i].real
                if is_2d:
                    # Reshape mode to 2D
                    mode_2d = mode.reshape((nx, ny))
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
                    cf = ax.contourf(x_mesh, y_mesh, mode_plot, levels=levels, cmap=CMAP_DIV, extend="both")
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
                    # Colorbar
                    fig.colorbar(cf, ax=ax, format="%.2f")
                else:
                    ax.plot(mode)
                    ax.set_xlabel("Spatial index")
                    ax.set_ylabel("Amplitude")
                ax.set_title(f"DMD Mode {i + 1} [{var_name}]")

            fig.tight_layout()
            fname = os.path.join(
                self.figures_dir,
                f"{self.data_root}_dmd_modes_{start + 1}_to_{end}_{var_name}.png",
            )
            fig.savefig(fname, dpi=FIG_DPI)
            plt.close(fig)
            print(f"Saving figure {fname}")

    def plot_time_coefficients(self, n_coeffs_to_plot=2):
        """Plot DMD temporal coefficients."""
        if self.time_coefficients.size == 0:
            print("No time coefficients to plot. Run perform_dmd() first.")
            return
        n_coeffs_to_plot = min(n_coeffs_to_plot, self.time_coefficients.shape[1], self.n_modes_save)
        if n_coeffs_to_plot == 0:
            print("No coefficients available to plot.")
            return
        Ns_total = self.time_coefficients.shape[0]
        t = np.arange(Ns_total) * self.data.get("dt", 1.0)
        plt.figure(figsize=(10, 3 * n_coeffs_to_plot))
        for i in range(n_coeffs_to_plot):
            plt.subplot(n_coeffs_to_plot, 1, i + 1)
            plt.plot(t, self.time_coefficients[:Ns_total, i].real, linewidth=1.5)
            plt.xlabel("Time")
            plt.ylabel(f"Amplitude Mode {i + 1}")
            plt.title(f"Temporal Coefficient for DMD Mode {i + 1}")
            plt.grid(True, linestyle=":")
            plt.xlim(t.min(), t.max())
        plt.tight_layout()
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_dmd_time_coeffs.png")
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close()
        print(f"Saving figure {plot_filename}")

    def plot_reconstruction_error(self):
        """Plot the data reconstruction error using an increasing number of DMD modes."""
        if self.modes.size == 0 or self.time_coefficients.size == 0 or "q" not in self.data:
            print("Data, modes, or time coefficients not available. Run perform_dmd() first.")
            return
        data_matrix = self.data["q"]
        # DMD reconstruction: sum_k a_k(t) * phi_k
        n_modes_check = self.modes.shape[1]
        reconstruction_errors = []
        for k in range(1, n_modes_check + 1):
            reconstructed_data_k_modes = self.time_coefficients[:, :k] @ self.modes[:, :k].T
            error = np.linalg.norm(data_matrix - reconstructed_data_k_modes, "fro") / np.linalg.norm(data_matrix, "fro")
            reconstruction_errors.append(error * 100)
        mode_indices = np.arange(1, n_modes_check + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(mode_indices, reconstruction_errors, "s-", linewidth=2, markersize=6)
        plt.xlabel("Number of Modes Used for Reconstruction")
        plt.ylabel("Reconstruction Error (%)")
        plt.title("Data Reconstruction Error vs. Number of DMD Modes")
        plt.grid(True, which="both", ls="--")
        plt.yscale("log")
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_dmd_reconstruction_error.png")
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close()
        print(f"Saving figure {plot_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DMD analysis")
    parser.add_argument("--config", help="Path to JSON/YAML configuration file", default=None)
    parser.add_argument("--prep", action="store_true", help="Load data and prepare for DMD")
    parser.add_argument("--compute", action="store_true", help="Perform DMD and save results")
    parser.add_argument("--plot", action="store_true", help="Generate default plots")
    args = parser.parse_args()

    if args.config:
        from configs import load_config
        load_config(args.config)

    # Example: data_file = "./data/consolidated_data.npz"
    data_file = "./data/consolidated_data.npz"
    n_modes_to_save_main = 8
    n_modes_to_plot_spatial_main = 8
    n_coeffs_to_plot_time_main = 8

    # Support batch field analysis for npz files
    if DNamiXNPZLoader is not None and data_file.endswith('.npz'):
        loader = DNamiXNPZLoader()
        available_fields = loader.get_available_fields(data_file)
        print(f"Available fields in {data_file}: {available_fields}")
        for field in available_fields:
            print(f"\n===== Running DMD for variable: {field} =====")
            data = loader.load(data_file, field=field)
            results_dir = os.path.join(RESULTS_DIR_DMD, field)
            figures_dir = os.path.join(FIGURES_DIR_DMD, field)
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(figures_dir, exist_ok=True)
            analyzer = DMDAnalyzer(
                file_path=data_file,
                results_dir=results_dir,
                figures_dir=figures_dir,
                data_loader=lambda fp: loader.load(fp, field=field),
                n_modes_save=n_modes_to_save_main,
                spatial_weight_type='uniform',
            )
            analyzer.data = data
            analyzer.analysis_type = f"dmd_{field}"
            
            if args.plot:
                # Only load results and plot, do not recompute
                analyzer.load_results()
                analyzer.plot_eigenspectra()
                analyzer.plot_modes_detailed(plot_n_modes=n_modes_to_plot_spatial_main)
                analyzer.plot_time_coefficients(n_coeffs_to_plot=n_coeffs_to_plot_time_main)
                analyzer.plot_cumulative_energy()
                analyzer.plot_reconstruction_error()
            else:
                # Run full pipeline (default behavior when no arguments)
                analyzer.perform_dmd()
                analyzer.save_results()
                analyzer.plot_eigenspectra()
                analyzer.plot_modes_detailed(plot_n_modes=n_modes_to_plot_spatial_main)
                analyzer.plot_time_coefficients(n_coeffs_to_plot=n_coeffs_to_plot_time_main)
                analyzer.plot_cumulative_energy()
                analyzer.plot_reconstruction_error()
            print_summary("DMD", analyzer.results_dir, analyzer.figures_dir)
    else:
        # Fallback for legacy .mat/.h5 files
        from utils import load_mat_data
        loader = load_mat_data
        analyzer = DMDAnalyzer(
            file_path=data_file,
            results_dir=RESULTS_DIR_DMD,
            figures_dir=FIGURES_DIR_DMD,
            data_loader=loader,
            spatial_weight_type='uniform',
            n_modes_save=n_modes_to_save_main,
        )
        run_all = not (args.prep or args.compute or args.plot)
        if run_all or args.prep:
            analyzer.load_and_preprocess()
        if run_all or args.compute:
            if analyzer.data == {}:
                analyzer.load_and_preprocess()
            analyzer.perform_dmd()
            analyzer.save_results()
        if run_all or args.plot:
            if analyzer.eigenvalues.size == 0:
                print("No DMD results to plot. Run with --compute first.")
            else:
                analyzer.plot_eigenspectra()
                analyzer.plot_modes_detailed(plot_n_modes=n_modes_to_plot_spatial_main)
                analyzer.plot_time_coefficients(n_coeffs_to_plot=n_coeffs_to_plot_time_main)
                analyzer.plot_cumulative_energy()
                analyzer.plot_reconstruction_error()
        if run_all:
            print_summary("DMD", analyzer.results_dir, analyzer.figures_dir)
