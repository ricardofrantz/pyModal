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
import matplotlib.pyplot as plt

# Third-party imports
import numpy as np
import scipy.linalg

from configs import (
    CMAP_DIV,
    FIGURES_DIR_DMD,
    RESULTS_DIR_DMD,
)
from utils import (
    BaseAnalyzer,
    auto_detect_weight_type,
    load_jetles_data,
    load_mat_data,
    make_result_filename,
    print_summary,
)


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
            f.create_dataset("x", data=self.data["x"], compression="gzip")
            f.create_dataset("y", data=self.data["y"], compression="gzip")
        print(f"DMD results saved to {path}")

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
        plt.title("DMD Eigenvalues")
        plt.grid(True, linestyle="-.")
        fname = os.path.join(self.figures_dir, f"{self.data_root}_dmd_eigs.png")
        plt.savefig(fname)
        plt.close()
        print(f"Eigenvalue plot saved to {fname}")

    def plot_modes(self, n_modes_to_plot=4):
        """Plot spatial DMD modes."""
        if self.modes.size == 0:
            print("No modes to plot.")
            return
        n_modes_to_plot = min(n_modes_to_plot, self.modes.shape[1])
        nx = self.data.get("Nx", int(np.sqrt(self.modes.shape[0])))
        ny = self.data.get("Ny", int(np.sqrt(self.modes.shape[0])))
        x_coords = self.data.get("x", np.arange(nx))
        y_coords = self.data.get("y", np.arange(ny))
        is_2d = self.modes.shape[0] == nx * ny and nx > 1 and ny > 1

        plt.figure(figsize=(4 * n_modes_to_plot, 4))
        for i in range(n_modes_to_plot):
            plt.subplot(1, n_modes_to_plot, i + 1)
            mode = self.modes[:, i].real
            if is_2d:
                img = mode.reshape(nx, ny).T
                extent = (
                    x_coords.min(),
                    x_coords.max(),
                    y_coords.min(),
                    y_coords.max(),
                )
                plt.imshow(img, origin="lower", extent=extent, cmap=CMAP_DIV)
                plt.colorbar(label="Mode amplitude")
            else:
                plt.plot(mode)
            plt.title(f"Mode {i + 1}")
        plt.tight_layout()
        fname = os.path.join(self.figures_dir, f"{self.data_root}_dmd_modes.png")
        plt.savefig(fname)
        plt.close()
        print(f"Mode plot saved to {fname}")

    def plot_time_coefficients(self, n_coeffs_to_plot=2):
        """Plot DMD temporal coefficients."""
        if self.time_coefficients.size == 0:
            print("No time coefficients to plot.")
            return
        n_coeffs_to_plot = min(n_coeffs_to_plot, self.time_coefficients.shape[1])
        t = np.arange(self.time_coefficients.shape[0]) * self.data.get("dt", 1.0)
        plt.figure(figsize=(8, 3 * n_coeffs_to_plot))
        for i in range(n_coeffs_to_plot):
            plt.subplot(n_coeffs_to_plot, 1, i + 1)
            plt.plot(t, self.time_coefficients[:, i].real)
            plt.xlabel("Time")
            plt.ylabel(f"Coeff {i + 1}")
            plt.grid(True)
        plt.tight_layout()
        fname = os.path.join(self.figures_dir, f"{self.data_root}_dmd_time_coeffs.png")
        plt.savefig(fname)
        plt.close()
        print(f"Time coefficient plot saved to {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DMD analysis")
    parser.add_argument("--prep", action="store_true", help="Load data")
    parser.add_argument("--compute", action="store_true", help="Compute DMD")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()

    data_file = "./data/jetLES.mat"
    if "jet" in data_file.lower():
        loader = load_jetles_data
        weight = "polar"
    else:
        loader = load_mat_data
        weight = "uniform"

    analyzer = DMDAnalyzer(
        file_path=data_file,
        results_dir=RESULTS_DIR_DMD,
        figures_dir=FIGURES_DIR_DMD,
        data_loader=loader,
        spatial_weight_type=weight,
        n_modes_save=10,
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
            analyzer.plot_eigenvalues()
            analyzer.plot_modes()
            analyzer.plot_time_coefficients()

    if run_all:
        print_summary("DMD", analyzer.results_dir, analyzer.figures_dir)
