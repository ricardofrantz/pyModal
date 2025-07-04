# Standard library imports
import argparse
import os
import time
import warnings

import h5py
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# Third-party imports
import numpy as np
from tqdm import tqdm

from configs import (
    CMAP_DIV,
    FIG_DPI,
    FIG_FORMAT,
    FIGURES_DIR_SPOD,
    RESULTS_DIR_SPOD,
    WINDOW_NORM,
    WINDOW_TYPE,
)
from parallel_utils import print_optimization_status

# Local application/library specific imports
from utils import (
    BaseAnalyzer,
    auto_detect_weight_type,  # For example in __main__ or if SPODAnalyzer needs it directly
    get_fig_aspect_ratio,
    get_num_threads,  # Utility to get number of available CPU threads
    load_jetles_data,  # For example in __main__
    load_mat_data,  # For example in __main__
    make_result_filename,
    print_summary,
    spod_function,  # Core SPOD routine for SPODAnalyzer
)

# Try to import DNamiXNPZLoader for npz support
try:
    from data_interface import DNamiXNPZLoader
except ImportError:
    DNamiXNPZLoader = None


class SPODAnalyzer(BaseAnalyzer):
    """
    Spectral Proper Orthogonal Decomposition (SPOD) analyzer.

    This class implements the SPOD method to decompose a data sequence into
    modes that are optimal in terms of energy for each frequency. It is
    particularly useful for analyzing spatio-temporal data from fluid dynamics
    simulations or experiments.

    ## Note: we need to test different inner products SPOD-u (using TKE) and for SPOD-p (known as extended SPOD which includes the acoustic radiation pressure)

    The SPOD algorithm involves:
    1. Computing the cross-spectral density (CSD) matrix from blocked FFTs of the data.
    2. Performing an eigenvalue decomposition of the CSD matrix for each frequency.
    The eigenvalues represent the energy of each mode at a given frequency, and
    the eigenvectors are the SPOD modes.

    Key Attributes:
        eigenvalues (np.ndarray): SPOD eigenvalues (energy) for each mode and frequency.
                                  Shape: (n_freq_bins, n_modes_saved_per_freq).
        modes (np.ndarray): SPOD spatial modes. Shape: (n_freq_bins, n_spatial_points, n_modes_saved_per_freq).
        time_coefficients (np.ndarray): SPOD temporal coefficients (reconstructed from modes and qhat).
                                        Shape: (n_freq_bins, n_modes_saved_per_freq, n_blocks).
        freq (np.ndarray): Array of frequencies corresponding to FFT bins.
        St (np.ndarray): Array of Strouhal numbers corresponding to `freq`.
        dst (float): Strouhal number step, used for integral weights in `spod_function`.
        qhat_cached (bool): Flag indicating if FFT blocks (q_hat) were loaded from cache.
        data_matrix (np.ndarray): Preprocessed data matrix [time, space].
        W (np.ndarray): Spatial weighting matrix (diagonal).
        fs (float): Sampling frequency of the data.
        L (float): Characteristic length for Strouhal number calculation.
        U (float): Characteristic velocity for Strouhal number calculation.

    Inherits from:
        BaseAnalyzer: Provides common functionalities for data loading, preprocessing, and FFT computation.
    """

    ############################################################
    # Initialization and Core Parameters                       #
    ############################################################
    def __init__(
        self,
        file_path,
        nfft=128,
        overlap=0.5,
        results_dir=RESULTS_DIR_SPOD,
        figures_dir=FIGURES_DIR_SPOD,
        blockwise_mean=False,
        normvar=False,
        window_norm=WINDOW_NORM,
        window_type=WINDOW_TYPE,
        data_loader=None,
        spatial_weight_type="auto",
        n_threads=None,
        use_parallel=True,
    ):
        """
        Initializes the SPODAnalyzer instance.

        Args:
            file_path (str): Path to the data file (e.g., .mat, .h5).
            nfft (int, optional): Number of points per FFT block. Defaults to 128.
            overlap (float, optional): Overlap fraction between FFT blocks (0 to <1).
                                     Defaults to 0.5 (50% overlap).
            results_dir (str, optional): Directory to save analysis results (HDF5 files).
                                         Defaults to `RESULTS_DIR_SPOD` from `configs.py`.
            figures_dir (str, optional): Directory to save generated plots.
                                         Defaults to `FIGURES_DIR_SPOD` from `configs.py`.
            blockwise_mean (bool, optional): If True, subtracts the mean of each block before FFT.
                                           If False, subtracts the global mean. Defaults to False.
            normvar (bool, optional): If True, normalizes FFT blocks by variance.
                                      Defaults to False.
            window_norm (str, optional): Normalization type for the window function ('amplitude' or 'power').
                                         Defaults to `WINDOW_NORM` from `configs.py`.
            window_type (str, optional): Type of window function to use (e.g., 'hamming', 'hanning', 'sine').
                                         Defaults to `WINDOW_TYPE` from `configs.py`.
            data_loader (callable, optional): Custom function to load data from `file_path`.
                                              If None, `BaseAnalyzer` attempts to auto-detect.
                                              Defaults to None.
            spatial_weight_type (str, optional): Type of spatial weights to apply ('auto', 'uniform', 'polar').
                                                 'auto' attempts to detect from filename.
                                                 Defaults to 'auto'.
        """
        super().__init__(
            file_path=file_path,
            nfft=nfft,
            overlap=overlap,
            results_dir=results_dir,
            figures_dir=figures_dir,
            data_loader=data_loader,
            spatial_weight_type=spatial_weight_type,
            n_threads=n_threads,
            use_parallel=use_parallel,
        )

        self._validate_inputs()
        # SPOD specific attributes
        self.blockwise_mean = blockwise_mean
        self.normvar = normvar
        self.window_norm = window_norm
        self.window_type = window_type

        self.St_normalization_factor = 1.0  # For Strouhal number calculation
        self.analysis_type = "spod"

        self.eigenvalues = np.array([])  # SPOD eigenvalues (L_d)
        self.modes = np.array([])  # SPOD spatial modes (Phi)
        self.time_coefficients = np.array([])  # SPOD temporal coefficients (Psi)
        self.freq = np.array([])  # Frequencies (from rfft)
        self.St = np.array([])  # Strouhal numbers
        self.dst = 0.0  # Strouhal step (for spod_function integral weight)
        self.qhat_cached = False  # Flag whether FFT blocks were loaded from cache

    def _validate_inputs(self):
        """
        Validates SPOD-specific input parameters.

        Ensures that `overlap` is within the range [0, 1) and `nfft` is positive.

        Raises:
            ValueError: If `overlap` is not in [0, 1) or `nfft` is not positive.
        """
        if not (0 <= self.overlap < 1):
            raise ValueError("Overlap must be between 0 (inclusive) and 1 (exclusive).")
        if self.nfft <= 0:
            raise ValueError("NFFT must be positive.")

    ############################################################
    # Data Loading and Preprocessing                           #
    ############################################################
    def load_and_preprocess(self):
        """
        Loads data, preprocesses it, and sets SPOD-specific parameters.

        This method extends `BaseAnalyzer.load_and_preprocess()` by:
        1. Calling the parent method to load data, apply spatial weights,
           subtract the mean, and store `self.data_matrix`, `self.W`, `self.fs`.
        2. Setting characteristic length (`self.L`) and velocity (`self.U`)
           based on filename conventions (e.g., 'cavity' or 'jet') for
           Strouhal number calculation.
        3. Calculating the frequency array (`self.freq`) from `rfftfreq`,
           the Strouhal numbers (`self.St`), and the Strouhal step (`self.dst`).
        """
        super().load_and_preprocess()  # Handles data loading, weighting, mean subtraction.
        # Sets self.data, self.W, self.fs, self.data_matrix

        # Set normalization constants for Strouhal number (L, U)
        if "cavity" in self.file_path.lower():
            self.L = 0.0381  # Characteristic length for cavity (e.g. diameter)
            self.U = 230.0  # Characteristic velocity for cavity (e.g. inflow velocity)
            print(f"Cavity case detected: Using L={self.L} m, U={self.U} m/s for Strouhal normalization.")
        elif "jet" in self.file_path.lower():
            self.L = self.data.get("D", 1.0)  # Characteristic length for jet (e.g. Diameter)
            self.U = self.data.get("U0", 1.0)  # Characteristic velocity for jet (e.g. U0)
            print(f"Jet case detected: Using L={self.L}, U={self.U} for Strouhal normalization.")
        else:
            self.L = 1.0  # Default characteristic length
            self.U = 1.0  # Default characteristic velocity
            print(f"Unknown case or L, U not in data: Using L={self.L}, U={self.U} for Strouhal normalization.")

        # Calculate Strouhal vector and frequency axis (self.freq is set by BaseAnalyzer)
        # Here, we ensure self.freq and self.St are set before perform_spod
        # BaseAnalyzer.compute_fft_blocks sets self.freq based on rfftfreq
        # If super().run() calls compute_fft_blocks, self.freq should be populated.
        # For safety, we can calculate it here if not already done or to ensure consistency.
        if not hasattr(self, "freq") or self.freq.size == 0:
            self.freq = np.fft.rfftfreq(self.nfft, d=self.data["dt"])

        self.St = self.freq * self.L / self.U

        if len(self.St) > 1:
            self.dst = self.St[1] - self.St[0]
        elif len(self.St) == 1:
            self.dst = self.St[0]  # Or some other appropriate non-zero value if St[0] can be 0
        else:
            self.dst = 0

    ############################################################
    # FFT Block Caching                                        #
    ############################################################
    def compute_fft_blocks(self):
        """Compute or load FFT blocks (qhat), with disk caching."""

        filename = make_result_filename(
            self.data_root,
            self.nfft,
            self.overlap,
            self.data.get("Ns", 0),
            self.analysis_type,
        )
        cache_path = os.path.join(self.results_dir, filename)

        # Try loading cached FFT blocks from previous SPOD run
        if os.path.exists(cache_path):
            try:
                with h5py.File(cache_path, "r") as f:
                    if "FFTBlocks" in f:
                        qhat_cached = f["FFTBlocks"][:]
                        if qhat_cached.shape[0] == self.nfft // 2 + 1:
                            self.qhat = qhat_cached
                            self.nblocks = qhat_cached.shape[2]
                            self.qhat_cached = True
                            print(f"Loaded cached FFT blocks from {cache_path}")
                            return
            except OSError as exc:
                print(f"Failed to load cached FFT blocks ({exc}), recomputing.")

        # Otherwise compute and save to cache
        super().compute_fft_blocks()

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

    ############################################################
    # Core SPOD Computation                                    #
    ############################################################
    def perform_spod(self, weights=None):
        """
        Performs the core SPOD analysis (eigenvalue decomposition for each frequency).

        This method computes the SPOD modes and eigenvalues by performing an
        eigenvalue decomposition of the cross-spectral density (CSD) matrix
        at each frequency bin. The CSD matrix is constructed from the
        Fourier-transformed data blocks (`self.qhat`).

        The actual computation is delegated to the `spod_function` imported from `utils.py`.

        Attributes set:
            eigenvalues (np.ndarray): SPOD eigenvalues.
            modes (np.ndarray): SPOD spatial modes.
            time_coefficients (np.ndarray): SPOD time coefficients.
        """
        # Make sure qhat has been computed
        if self.qhat is None or self.qhat.size == 0:
            print("Error: qhat not computed. Call super().run(compute_fft=True) first.")
            return

        start_time = time.time()

        num_space_points = self.qhat.shape[1]
        num_freq_bins = self.qhat.shape[0]

        # Check if self.freq and self.St are consistent with qhat's frequency bins
        if len(self.freq) != num_freq_bins:
            print(f"Warning: self.freq length ({len(self.freq)}) mismatch with qhat bins ({num_freq_bins}). Recalculating.")
            # Recalculate freq and St based on nfft and fs (from BaseAnalyzer)
            self.freq = np.fft.rfftfreq(self.nfft, d=1.0 / self.fs)[:num_freq_bins]
            self.St = self.freq * self.L / self.U
            print(f"Realigned self.freq to {len(self.freq)} elements and self.St.")

        # Initialize result arrays using num_freq_bins
        self.eigenvalues = np.zeros((num_freq_bins, self.nblocks))
        self.modes = np.zeros((num_freq_bins, num_space_points, self.nblocks), dtype=complex)  # Spatial modes
        self.time_coefficients = np.zeros((num_freq_bins, self.nblocks, self.nblocks), dtype=complex)  # Temporal coefficients

        print("Performing SPOD for each frequency...")

        weights = weights if weights is not None else self.W
        for i in tqdm(range(num_freq_bins), desc="SPOD Computation", unit="freq"):
            qhat_freq = self.qhat[i, :, :]
            phi_freq, lambda_freq, psi_freq = spod_function(
                qhat_freq,
                self.nblocks,
                self.dst,
                weights,
                return_psi=True,
                use_parallel=self.use_parallel,
            )
            self.modes[i, :, :] = phi_freq
            self.eigenvalues[i, :] = lambda_freq
            self.time_coefficients[i, :, :] = psi_freq
        print(f"SPOD eigenvalue decomposition completed in {time.time() - start_time:.2f} seconds")

    ############################################################
    # Results Handling                                         #
    ############################################################
    def save_results(self):
        """
        Saves SPOD modes, eigenvalues, frequencies, and Strouhal numbers to an HDF5 file.

        The results are saved in the `self.results_dir` directory. The filename is
        generated using `make_result_filename` based on the input data file name,
        `nfft`, `overlap`, and the analysis type ('spod').

        Datasets saved:
            'Eigenvalues': SPOD eigenvalues.
            'Modes': SPOD spatial modes.
            'TimeCoefficients': (Optional) SPOD time coefficients.
            'Frequencies': Frequency array.
            'Strouhal': Strouhal number array.
            'dt': Time step of the original data.
            'nfft': NFFT used for the analysis.
            'overlap': Overlap fraction used.
            'window_type': Window type used for FFT.
        """
        filename = make_result_filename(self.data_root, self.nfft, self.overlap, self.data.get("Ns", 0), self.analysis_type)
        save_path = os.path.join(self.results_dir, filename)
        os.makedirs(self.results_dir, exist_ok=True)

        # Check if the file exists from a previous BaseAnalyzer save
        # If it does, open in append mode, otherwise write mode
        mode = "a" if os.path.exists(save_path) else "w"

        print(f"Saving SPOD-specific results to {save_path} (mode: {mode})")
        with h5py.File(save_path, mode) as f:
            if "Eigenvalues" in f:
                del f["Eigenvalues"]
            if "Modes" in f:
                del f["Modes"]
            if "TimeCoefficients" in f:
                del f["TimeCoefficients"]
            if "Freq" in f:
                del f["Freq"]
            if "St" in f:
                del f["St"]
            if "FFTBlocks" in f:
                del f["FFTBlocks"]

            f.create_dataset("Eigenvalues", data=self.eigenvalues, compression="gzip")
            f.create_dataset("Modes", data=self.modes, compression="gzip")
            if self.time_coefficients is not None and self.time_coefficients.size > 0:
                f.create_dataset("TimeCoefficients", data=self.time_coefficients, compression="gzip")
            f.create_dataset("Freq", data=self.freq, compression="gzip")
            f.create_dataset("St", data=self.St, compression="gzip")
            if self.qhat is not None and self.qhat.size > 0:
                f.create_dataset("FFTBlocks", data=self.qhat, compression="gzip")

            # Ensure base attributes are also saved if this is the first save operation (mode 'w')
            if mode == "w":
                if "x_coords" in self.data and "x_coords" not in f:
                    f.create_dataset("x_coords", data=self.data["x_coords"])
                if "y_coords" in self.data and "y_coords" not in f:
                    f.create_dataset("y_coords", data=self.data["y_coords"])
                if "z_coords" in self.data and "z_coords" not in f:
                    f.create_dataset("z_coords", data=self.data["z_coords"])
                for key, value in self._get_metadata().items():
                    if key not in f.attrs:
                        f.attrs[key] = value
            else:  # Append mode, just update SPOD specific attributes if necessary
                f.attrs["blockwise_mean"] = self.blockwise_mean
                f.attrs["normvar"] = self.normvar
                f.attrs["window_norm"] = self.window_norm
                f.attrs["window_type"] = self.window_type

                # Coordinates and Weights might have been saved by BaseAnalyzer.save_results if it was called.
                # Here, we ensure they are present if SPODAnalyzer.save_results is the primary save method.
                if "x_coords" not in f and self.data.get("x") is not None:
                    f.create_dataset("x_coords", data=self.data["x"], compression="gzip")
                if "y_coords" not in f and self.data.get("y") is not None:
                    f.create_dataset("y_coords", data=self.data["y"], compression="gzip")

                # Save weights if not already saved by BaseAnalyzer or this method
                if self.W is not None and self.W.size > 0 and "Weights" not in f:
                    f.create_dataset("Weights", data=self.W, compression="gzip")

        print(f"SPOD results saved to {save_path} (HDF5 attributes updated/created)")

    ############################################################
    # Main Analysis Pipeline Orchestration                     #
    ############################################################
    def run_analysis(self, plot_modes_options=None, plot_reconstruction_options=None):
        """
        Run the full SPOD analysis pipeline: load, compute, save, and plot.

        Args:
            plot_modes_options (dict, optional): Options for `plot_modes`.
                                               Example: {'modes_to_plot': [0, 1], 'freqs_to_plot': [self.St[0], self.St[1]]}
            plot_reconstruction_options (dict, optional): Options for `plot_reconstruction_error`.
                                                        Example: {'n_modes_max_error': 50}
        """
        print(f"🔎 Starting SPOD analysis for {os.path.basename(self.file_path)}")
        start_total_time = time.time()

        self.load_and_preprocess()
        super().run(compute_fft=True)  # Compute/load qhat before SPOD
        self.perform_spod()
        self.save_results()

        # Generate plots
        self.plot_eigenvalues_v2()  # Call the new plotting function
        self.plot_cumulative_energy()

        if plot_modes_options:
            self.plot_modes(**plot_modes_options)
        else:
            self.plot_modes()
        self.plot_time_coeffs()
        self.plot_reconstruction_error(**(plot_reconstruction_options or {}))
        self.plot_eig_complex_plane()

        print_summary("SPOD", self.results_dir, self.figures_dir)

    def plot_eigenvalues_v2(self, n_modes_line_plot=20, shading_cmap="inferno_r"):
        """Plot the SPOD eigenvalue spectrum (energy vs. Strouhal number) - Version 2.

        This version aims for a style similar to a user-provided example,
        featuring a shaded background for the eigenvalue bundle and grayscale lines for modes.

        Args:
            n_modes_line_plot (int, optional): Number of dominant mode eigenvalue lines to plot.
                                             Defaults to 20.
            shading_cmap (str, optional): Colormap for the background shading of the eigenvalue bundle.
                                        Defaults to 'inferno_r' (yellow at bottom, dark at top).
        """
        if self.eigenvalues.size == 0 or self.St.size == 0:
            print("Eigenvalues (self.eigenvalues) or Strouhal numbers (self.St) not computed. Run perform_spod() first.")
            return

        L_plot = self.eigenvalues  # Use self.eigenvalues for the eigenvalue matrix
        St_plot = self.St

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Determine y-range for pcolormesh shading (covering the bundle)
        # Ensure l_min is positive for log scale
        l_min_positive = L_plot[L_plot > 0].min() if np.any(L_plot > 0) else 1e-6
        y_fill_min = l_min_positive * 0.1
        y_fill_max = L_plot[:, 0].max()  # Shade up to the first mode's max
        if y_fill_min >= y_fill_max:
            y_fill_min = y_fill_max * 0.01  # Ensure min < max

        # Create a grid for pcolormesh
        # y_coords_mesh should be log-spaced for even coloring in log y-scale
        num_y_points_mesh = 100
        y_coords_mesh = np.logspace(np.log10(y_fill_min), np.log10(y_fill_max), num_y_points_mesh)

        # Fill the background shading without explicit Python loops
        first_mode = L_plot[:, 0]
        mask = y_coords_mesh[:, None] <= first_mode[None, :]
        C_fill_data = np.where(
            mask,
            np.broadcast_to(y_coords_mesh[:, None], (num_y_points_mesh, len(St_plot))),
            np.nan,
        )
        # Mask NaNs to avoid matplotlib warnings when using LogNorm
        C_fill_data = np.ma.masked_invalid(C_fill_data)

        norm_vmin = np.min(y_coords_mesh[y_coords_mesh > 0])
        norm_vmax = np.max(y_coords_mesh)
        if norm_vmin >= norm_vmax:
            norm_vmin = norm_vmax * 0.1  # ensure vmin < vmax and positive

        if norm_vmin > 0 and norm_vmax > 0 and norm_vmin < norm_vmax:
            mesh = ax.pcolormesh(St_plot, y_coords_mesh, C_fill_data, shading="gouraud", cmap=shading_cmap, norm=colors.LogNorm(vmin=norm_vmin, vmax=norm_vmax), rasterized=True)  # Rasterize for potentially large mesh
        else:
            print("Warning: Could not generate pcolormesh shading due to invalid vmin/vmax for LogNorm.")

        # Plot individual modal energies (eigenvalues) - grayscale lines
        num_modes_actual = min(n_modes_line_plot, L_plot.shape[1])
        mode_colors = plt.cm.gray(np.linspace(0.0, 0.7, num_modes_actual))
        for i in range(num_modes_actual):
            ax.plot(St_plot, L_plot[:, i], color=mode_colors[i], linewidth=0.8, alpha=0.9)

        # Plot total energy (sum of eigenvalues) - red line
        ax.plot(St_plot, np.sum(L_plot, axis=1), color="red", linewidth=1.5, label="Total Energy")

        ax.set_xlabel("Strouhal number")
        ax.set_ylabel(r"$\lambda$")  # Use raw string for LaTeX
        ax.set_title("SPOD Eigenvalue Spectrum (v2)")

        # Set plot limits - adjust as necessary
        if np.any(St_plot > 0):
            st_min = St_plot[St_plot > 0].min()
        else:
            st_min = St_plot.min()
        ax.set_xlim(st_min, St_plot.max())
        plot_y_min = y_fill_min * 0.5
        plot_y_max = np.max(np.sum(L_plot, axis=1)) * 2.0
        if plot_y_min > 0 and plot_y_max > 0 and plot_y_min < plot_y_max:
            ax.set_ylim(plot_y_min, plot_y_max)
        else:  # Fallback if limits are problematic
            current_st_min, current_st_max = ax.get_xlim()
            if np.any(L_plot[L_plot > 0]):
                ax.set_ylim(L_plot[L_plot > 0].min() * 0.1, np.sum(L_plot, axis=1).max() * 2.0)

        # Use settings from configs.py for saving
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_SPOD_eigenvalues_v2_nfft{self.nfft}_noverlap{self.overlap}.{FIG_FORMAT}")  # Corrected self.novlap

        # Save the figure
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        print(f"SPOD eigenvalue plot (v2) saved to {plot_filename}")

    from typing import Optional

    def plot_modes(self, freqs_to_plot=None, plot_n_modes: Optional[int] = 10, modes_per_fig: int = 1) -> None:
        """Plot spatial modes for selected frequencies as individual figures."""

        if self.modes.size == 0 or self.St.size == 0:
            print("No modes to plot. Run perform_spod() first.")
            return

        n_modes_total = self.modes.shape[2]
        if plot_n_modes is None:
            n_modes = n_modes_total
        else:
            n_modes = min(plot_n_modes, n_modes_total)
        modes_to_plot = list(range(n_modes))

        if freqs_to_plot is None:
            dominant_idx = int(np.argmax(self.eigenvalues[:, 0]))
            freqs_to_plot = [dominant_idx]

        freq_indices = []
        for f in freqs_to_plot:
            if isinstance(f, (int, np.integer)) and 0 <= f < len(self.St):
                freq_indices.append(int(f))
            else:
                freq_indices.append(int(np.argmin(np.abs(self.St - float(f)))))

        Nx = self.data.get("Nx", int(np.sqrt(self.modes.shape[1])))
        Ny = self.data.get("Ny", int(np.sqrt(self.modes.shape[1])))
        x_coords = self.data.get("x", np.arange(Nx))
        y_coords = self.data.get("y", np.arange(Ny))

        fig_aspect = get_fig_aspect_ratio(self.data)
        var_name = self.data.get("metadata", {}).get("var_name", "q")

        for f_idx in freq_indices:
            st_val = self.St[f_idx]
            for start in range(0, n_modes, modes_per_fig):
                end = min(start + modes_per_fig, n_modes)
                ncols = end - start
                if Nx * Ny == self.modes.shape[1] and Nx > 1 and Ny > 1:
                    fig, axes = plt.subplots(
                        1,
                        ncols,
                        figsize=(4 * ncols * fig_aspect, 4),
                        squeeze=False,
                    )
                else:
                    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3), squeeze=False)
                axes = axes.ravel()
                for j, m_idx in enumerate(range(start, end)):
                    if m_idx >= self.modes.shape[2]:
                        continue
                    ax = axes[j]
                    mode_real = self.modes[f_idx, :, m_idx].real
                    if Nx * Ny == mode_real.size and Nx > 1 and Ny > 1:
                        mode_2d = mode_real.reshape(Nx, Ny)
                        if x_coords.ndim == 1 and y_coords.ndim == 1:
                            X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
                        else:
                            X, Y = x_coords, y_coords
                        dist = np.sqrt(X**2 + Y**2)
                        cyl_mask = dist <= 0.5
                        mode_plot = np.ma.array(mode_2d, mask=np.isnan(mode_2d) | cyl_mask)
                        vmax = np.max(np.abs(mode_plot))
                        levels = np.linspace(-vmax, vmax, 21)
                        im = ax.contourf(X, Y, mode_plot, levels=levels, cmap=CMAP_DIV, extend="both")
                        ax.contour(X, Y, mode_plot, levels=levels[::4], colors="k", linewidths=0.5, alpha=0.5)
                        ax.add_patch(plt.Circle((0, 0), 0.5, facecolor="lightgray", edgecolor="black", linewidth=0.5))
                        ax.set_aspect("equal", "box")
                        ax.set_xlim(np.min(x_coords), np.max(x_coords))
                        ax.set_ylim(np.min(y_coords), np.max(y_coords))
                        fig.colorbar(im, ax=ax, shrink=0.8, label=r"Re($\Phi$)")
                        ax.set_xlabel(r"$x/D$")
                        ax.set_ylabel(r"$y/D$")
                        ax.grid(True, linestyle="--", alpha=0.3)
                    else:
                        ax.plot(mode_real)
                        ax.set_xlabel("Spatial index")
                        ax.set_ylabel("Amplitude")
                    ax.set_title(f"SPOD Mode {m_idx + 1} at St={st_val:.4f} [{var_name}]")

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    fig.tight_layout()
                if modes_per_fig == 1 and ncols == 1:
                    fname = os.path.join(
                        self.figures_dir,
                        f"{self.data_root}_SPOD_mode{start + 1}_freq{f_idx}_{var_name}.png",
                    )
                else:
                    fname = os.path.join(
                        self.figures_dir,
                        f"{self.data_root}_SPOD_modes_{start + 1}_to_{end}_freq{f_idx}_{var_name}.png",
                    )
                fig.savefig(fname, dpi=FIG_DPI)
                plt.close(fig)
                print(f"SPOD modes {start + 1}-{end} at St={st_val:.4f} (freq index {f_idx}) saved to {fname}")

    def plot_cumulative_energy(self, freq_idx=None):
        """Plot cumulative energy captured by modes at a given frequency."""
        if self.eigenvalues.size == 0:
            print("No eigenvalues to plot. Run perform_spod() first.")
            return
        if freq_idx is None:
            freq_idx = int(np.argmax(self.eigenvalues[:, 0]))
        lambdas = self.eigenvalues[freq_idx, :]
        cumulative = np.cumsum(lambdas)
        cumulative /= cumulative[-1]
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, len(cumulative) + 1), cumulative, "o-")
        ax.set_xlabel("Mode index")
        ax.set_ylabel("Cumulative energy")
        ax.set_title(f"Cumulative energy at St={self.St[freq_idx]:.3f}")
        plot_filename = os.path.join(
            self.figures_dir,
            f"{self.data_root}_SPOD_cumulative_freq{freq_idx}.{FIG_FORMAT}",
        )
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        print(f"Cumulative energy plot saved to {plot_filename}")

    def plot_time_coeffs(self, modes_to_plot=None, freq=None, n_blocks=None):
        """Plot temporal coefficients for selected modes."""
        if self.time_coefficients.size == 0:
            print("No time coefficients to plot. Run perform_spod() first.")
            return
        if freq is None:
            freq_idx = int(np.argmax(self.eigenvalues[:, 0]))
        else:
            freq_idx = int(np.argmin(np.abs(self.St - float(freq))))
        coeffs = self.time_coefficients[freq_idx, :, :]
        if n_blocks is not None:
            coeffs = coeffs[:n_blocks, :]
        if modes_to_plot is None:
            modes_to_plot = list(range(min(4, coeffs.shape[1])))
        fig, ax = plt.subplots()
        for m in modes_to_plot:
            if m < coeffs.shape[1]:
                ax.plot(coeffs[:, m].real, label=f"Mode {m + 1} (real)")
                ax.plot(coeffs[:, m].imag, "--", label=f"Mode {m + 1} (imag)")
        ax.set_xlabel("Block index")
        ax.set_ylabel("Coefficient")
        ax.legend()
        ax.set_title(f"SPOD Time Coefficients at St={self.St[freq_idx]:.4f}")
        plot_filename = os.path.join(
            self.figures_dir,
            f"{self.data_root}_SPOD_timecoeffs_freq{freq_idx}_nfft{self.nfft}_noverlap{self.overlap}.{FIG_FORMAT}",
        )
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        print(f"Time coefficients plot saved to {plot_filename}")

    def plot_reconstruction_error(self, st_target=None, n_modes_max=None):
        """Plot reconstruction error of qhat as a function of modes used."""
        if self.qhat.size == 0 or self.modes.size == 0:
            print("No data to plot reconstruction error. Run perform_spod() first.")
            return
        if st_target is None:
            freq_idx = int(np.argmax(self.eigenvalues[:, 0]))
        else:
            freq_idx = int(np.argmin(np.abs(self.St - float(st_target))))
        qhat_f = self.qhat[freq_idx, :, :]
        modes_f = self.modes[freq_idx, :, :]
        n_modes_avail = modes_f.shape[1]
        if n_modes_max is None:
            n_modes_max = n_modes_avail
        n_modes_max = min(n_modes_max, n_modes_avail)
        W = np.diagflat(self.W) if self.W.ndim == 1 or self.W.shape[1] == 1 else self.W
        norm_orig = np.linalg.norm(qhat_f, "fro")
        errors = []
        mode_counts = range(1, n_modes_max + 1)
        for k in mode_counts:
            phi_k = modes_f[:, :k]
            coeffs = phi_k.conj().T @ W @ qhat_f
            if coeffs.shape != (k, qhat_f.shape[1]):
                raise ValueError(f"Coefficient matrix has unexpected shape: {coeffs.shape}, expected ({k}, {qhat_f.shape[1]})")
            qrec = phi_k @ coeffs
            errors.append(np.linalg.norm(qhat_f - qrec, "fro") / norm_orig)
        fig, ax = plt.subplots()
        ax.plot(list(mode_counts), errors, "o-")
        ax.set_yscale("log")
        ax.set_xlabel("Number of SPOD Modes")
        ax.set_ylabel("Relative Error")
        ax.set_title(f"Reconstruction Error at St={self.St[freq_idx]:.4f}")
        plot_filename = os.path.join(
            self.figures_dir,
            f"{self.data_root}_SPOD_reconstruction_error_freq{freq_idx}_nfft{self.nfft}_noverlap{self.overlap}.{FIG_FORMAT}",
        )
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        print(f"Reconstruction error plot saved to {plot_filename}")

    def plot_eig_complex_plane(self, freq=None, n_modes=4):
        """Plot modes in the complex plane for a given frequency."""
        if self.modes.size == 0:
            print("No modes to plot. Run perform_spod() first.")
            return
        if freq is None:
            freq_idx = int(np.argmax(self.eigenvalues[:, 0]))
        else:
            freq_idx = int(np.argmin(np.abs(self.St - float(freq))))
        modes = self.modes[freq_idx, :, :n_modes]
        fig, ax = plt.subplots()
        for i in range(modes.shape[1]):
            ax.scatter(modes[:, i].real, modes[:, i].imag, s=10, alpha=0.7, label=f"Mode {i + 1}")
        ax.set_xlabel("Real")
        ax.set_ylabel("Imag")
        ax.set_title(f"SPOD Modes Complex Plane St={self.St[freq_idx]:.4f}")
        ax.legend()
        plot_filename = os.path.join(
            self.figures_dir,
            f"{self.data_root}_SPOD_complex_freq{freq_idx}_nfft{self.nfft}_noverlap{self.overlap}.{FIG_FORMAT}",
        )
        plt.savefig(plot_filename, dpi=FIG_DPI)
        plt.close(fig)
        print(f"Complex plane plot saved to {plot_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SPOD analysis")
    parser.add_argument("--config", help="Path to JSON/YAML configuration file", default=None)
    parser.add_argument("--prep", action="store_true", help="Load data and compute FFT blocks")
    parser.add_argument("--compute", action="store_true", help="Perform SPOD and save results")
    parser.add_argument("--plot", action="store_true", help="Generate default plots")
    args = parser.parse_args()

    from parallel_utils import get_threadpool_summary

    print(f"Thread pools: {get_threadpool_summary()}")

    print_optimization_status()

    if args.config:
        from configs import load_config

        load_config(args.config)

    # --- Configuration ---
    # data_file = "./data/jetLES_small.mat"  # Updated data path
    # data_file = "./data/jetLES.mat"  # Path to your data file
    # data_file = "./data/cavityPIV.mat"  # Path to your data file
    data_file = "./data/snp1-947_u.npz"

    # Default parameters
    nfft_param = 128  # FFT block size
    overlap_param = 0.5  # Overlap fraction (50%)
    freq_target_for_plots = None  # Target St for plotting, if None, picks dominant
    # ---------------------

    # Check if data file exists, use dummy if not found
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at '{data_file}'")
        print("Attempting to use dummy data generator...")
        try:
            from utils import generate_dummy_data_like_jetles  # Assumed utility

            data_file = "./data/dummy_jetles_data.h5"  # Create/use an HDF5 dummy file
            if not os.path.exists(data_file):
                # Generate a small coherent dataset for quick testing
                generate_dummy_data_like_jetles(
                    output_path=data_file,
                    Ns=200,
                    Nx=30,
                    Ny=20,
                    dt=0.01,
                    save_mat=False,
                )
            print(f"Using dummy data: {data_file}")
        except ImportError:
            print("Dummy data generator 'generate_dummy_data_like_jetles' not found in utils.py. Exiting.")
            exit(1)
        except Exception as e:
            print(f"Error generating/using dummy data: {e}. Exiting.")
            exit(1)

    # Set case-specific parameters and data loader
    if DNamiXNPZLoader is not None and data_file.endswith(".npz"):
        loader = DNamiXNPZLoader()
        available_fields = loader.get_available_fields(data_file)
        print(f"Available fields in {data_file}: {available_fields}")
        for field in available_fields:
            print(f"\n===== Running SPOD for variable: {field} =====")
            data = loader.load(data_file, field=field)
            results_dir = os.path.join(RESULTS_DIR_SPOD, field)
            figures_dir = os.path.join(FIGURES_DIR_SPOD, field)
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(figures_dir, exist_ok=True)
            analyzer = SPODAnalyzer(
                file_path=data_file,
                nfft=nfft_param,
                overlap=overlap_param,
                data_loader=lambda fp: loader.load(fp, field=field),
                spatial_weight_type="uniform",
            )
            analyzer.data = data
            analyzer.analysis_type = f"spod_{field}"
            run_all = not (args.prep or args.compute or args.plot)
            if run_all or args.prep:
                analyzer.load_and_preprocess()
                analyzer.compute_fft_blocks()
            if run_all or args.compute:
                if analyzer.qhat.size == 0:
                    analyzer.load_and_preprocess()
                    analyzer.compute_fft_blocks()
                analyzer.perform_spod()
                analyzer.save_results()
            if run_all or args.plot:
                if analyzer.eigenvalues.size == 0 or analyzer.St.size == 0:
                    print("No SPOD results to plot. Run with --compute first.")
                else:
                    analyzer.plot_eigenvalues_v2()
                    analyzer.plot_modes()
            if run_all:
                print_summary("SPOD", analyzer.results_dir, analyzer.figures_dir)
        exit(0)
    elif "cavity" in data_file.lower():
        nfft_param = 256
        overlap_param = 128 / 256  # 0.5
        window_type_param = "sine"
        spatial_weight_type_param = "uniform"
        data_loader_param = load_mat_data
        print("Cavity case: Using nfft=256, sine window, uniform weights, load_mat_data.")
    elif "jet" in data_file.lower() or "dummy_jetles_data" in data_file.lower():
        nfft_param = 256
        overlap_param = 0.5
        window_type_param = WINDOW_TYPE
        spatial_weight_type_param = auto_detect_weight_type(data_file)
        data_loader_param = load_jetles_data
        print(f"Jet case: Using nfft={nfft_param}, overlap={overlap_param * 100}%, {window_type_param} window, {spatial_weight_type_param} weights, load_jetles_data.")
    else:
        print(f"Warning: Unknown data case for '{data_file}'. Using default jet parameters.")
        nfft_param = 256
        overlap_param = 0.5
        window_type_param = WINDOW_TYPE
        spatial_weight_type_param = "auto"
        data_loader_param = load_jetles_data

    # Further configuration
    n_modes_to_save = 50  # Number of SPOD modes to save

    analyzer = SPODAnalyzer(
        file_path=data_file,
        nfft=nfft_param,
        overlap=overlap_param,
        data_loader=data_loader_param,
        spatial_weight_type=spatial_weight_type_param,
    )

    run_all = not (args.prep or args.compute or args.plot)

    if run_all or args.prep:
        analyzer.load_and_preprocess()
        analyzer.compute_fft_blocks()

    if run_all or args.compute:
        if analyzer.qhat.size == 0:
            analyzer.load_and_preprocess()
            analyzer.compute_fft_blocks()
        analyzer.perform_spod()
        analyzer.save_results()

    if run_all or args.plot:
        if analyzer.eigenvalues.size == 0 or analyzer.St.size == 0:
            print("No SPOD results to plot. Run with --compute first.")
        else:
            analyzer.plot_eigenvalues_v2()
            analyzer.plot_modes()

    if run_all:
        print_summary("SPOD", analyzer.results_dir, analyzer.figures_dir)

    # Example of calling plot_eigenvalues_v2 directly if needed for specific params
    # analyzer.plot_eigenvalues_v2(n_modes_line_plot=15, shading_cmap='viridis_r')
