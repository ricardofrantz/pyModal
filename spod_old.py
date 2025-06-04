#!/usr/bin/env python3
"""
Extract modes with Spectral Proper Orthogonal Decomposition (SPOD)

For now we only use standard version and NOT streaming version, as available in
https://github.com/MathEXLab/PySPOD/tree/main/pyspod/spod

Author: R. Frantz

Reference codes:
    - https://github.com/SpectralPOD/spod_matlab/tree/master
    - https://github.com/MathEXLab/PySPOD/blob/main/tutorials/tutorial1/tutorial1.ipynb
"""

# All core imports and configs are available via utils
from utils import *


class SPODAnalyzer(BaseAnalyzer):
    ############################################################
    # Disk caching for FFT blocks                             #
    #                                                        #
    # Motivation: For very large datasets, computing FFT      #
    # blocks is expensive and may exceed memory limits.       #
    # By saving FFT blocks to disk (caching), we avoid        #
    # recomputation on repeated runs with the same data and   #
    # parameters. This is especially beneficial for large     #
    # simulations or parameter sweeps, where the FFT step     #
    # dominates runtime.                                      #
    #                                                        #
    # Note: The first run will be slightly slower due to      #
    # saving to disk, but subsequent runs are much faster     #
    # as FFT blocks are loaded directly from cache.           #
    # For small datasets or one-off runs, caching can be      #
    # disabled if desired.                                    #
    ############################################################
    # Disk caching for FFT blocks #
    ###############################
    def _get_blocks_directory(self):
        # All FFT block cache is stored in the central cache directory
        blocks_dir = os.path.join(CACHE_DIR, f"{self.data_root}_blocks")
        os.makedirs(blocks_dir, exist_ok=True)
        return blocks_dir

    def _check_cached_blocks_exist(self):
        blocks_dir = self._get_blocks_directory()
        metadata_file = os.path.join(blocks_dir, "metadata.json")
        if not os.path.exists(metadata_file):
            return False
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            current_metadata = {
                "nfft": self.nfft,
                "overlap": self.overlap,
                "blockwise_mean": getattr(self, "blockwise_mean", False),
                "normvar": getattr(self, "normvar", False),
                "window_norm": getattr(self, "window_norm", "power"),
                "window_type": getattr(self, "window_type", "hamming"),
                "file_path": self.file_path,
                "n_blocks": self.nblocks,
            }
            for key, value in current_metadata.items():
                if key not in metadata or metadata[key] != value:
                    return False
            expected_count = self.nblocks * (self.nfft // 2 + 1)
            block_files = glob.glob(os.path.join(blocks_dir, "block_*.npy"))
            if len(block_files) < expected_count:
                return False
            return True
        except Exception:
            return False

    def _save_fft_block(self, freq_idx, block_idx, data):
        blocks_dir = self._get_blocks_directory()
        filename = f"block_{freq_idx:04d}_{block_idx:04d}.npy"
        np.save(os.path.join(blocks_dir, filename), data)

    def _load_fft_block(self, freq_idx, block_idx):
        blocks_dir = self._get_blocks_directory()
        filename = f"block_{freq_idx:04d}_{block_idx:04d}.npy"
        return np.load(os.path.join(blocks_dir, filename))

    def _save_blocks_metadata(self):
        blocks_dir = self._get_blocks_directory()
        metadata = {
            "nfft": self.nfft,
            "overlap": self.overlap,
            "blockwise_mean": getattr(self, "blockwise_mean", False),
            "normvar": getattr(self, "normvar", False),
            "window_norm": getattr(self, "window_norm", "power"),
            "window_type": getattr(self, "window_type", "hamming"),
            "file_path": self.file_path,
            "n_blocks": self.nblocks,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(blocks_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    ###############################
    # Robust error handling       #
    ###############################
    def _validate_inputs(self):
        if not hasattr(self, "data") or "q" not in self.data:
            raise ValueError("Data not loaded or 'q' field missing.")
        if self.nfft <= 0:
            raise ValueError(f"Invalid nfft: {self.nfft}. Must be positive.")
        if self.overlap < 0 or self.overlap >= 1:
            raise ValueError(f"Invalid overlap: {self.overlap}. Must be in [0, 1).")
        valid_windows = ["hamming", "hann", "rectangular", "sine"]
        if self.window_type not in valid_windows:
            raise ValueError(f"Invalid window_type: {self.window_type}. Must be one of {valid_windows}.")
        valid_norms = ["power", "amplitude"]
        if self.window_norm not in valid_norms:
            raise ValueError(f"Invalid window_norm: {self.window_norm}. Must be one of {valid_norms}.")
        if self.data["Ns"] < self.nfft:
            raise ValueError(f"Too few time samples ({self.data['Ns']}) for FFT size ({self.nfft}).")
        required_fields = ["x", "y", "Nx", "Ny", "dt"]
        missing_fields = [field for field in required_fields if field not in self.data]
        if missing_fields:
            raise ValueError(f"Missing required data fields: {', '.join(missing_fields)}")

    ###############################
    # Flexible data handling      #
    ###############################
    def load_and_preprocess(self):
        super().load_and_preprocess()
        self._validate_inputs()
        # Flexible handling for ndarray input
        if isinstance(self.data, dict):
            pass
        elif isinstance(self.data, np.ndarray):
            if self.data.ndim >= 2:
                if self.data.ndim > 2:
                    time_steps = self.data.shape[0]
                    spatial_points = np.prod(self.data.shape[1:])
                    self.data = {"q": self.data.reshape(time_steps, spatial_points)}
                else:
                    self.data = {"q": self.data}
            else:
                raise ValueError("Input array must have at least 2 dimensions (time, space)")
        else:
            raise ValueError(f"Unsupported data type: {type(self.data)}. Must be dict or ndarray.")
        # Strouhal normalization
        if "cavity" in self.file_path.lower():
            self.L = 0.0381
            self.U = 230.0
            print(f"Cavity case detected: Using L={self.L} m, U={self.U} m/s for Strouhal normalization.")
        else:
            self.L = 1.0
            self.U = 1.0
            print("Jet case or unknown: Using L=1, U=1 for Strouhal normalization.")
        self.fs = 1 / self.data["dt"]
        f = np.linspace(0, self.fs - self.fs / self.nfft, self.nfft)
        St = f * self.L / self.U
        self.St = St[0 : self.nfft // 2 + 1]
        self.dst = self.St[1] - self.St[0]
        self.strouhal = St

    ###############################
    # Progress reporting + caching#
    ###############################
    def compute_fft_blocks(self):
        if self._check_cached_blocks_exist():
            print("Loading FFT blocks from disk cache...")
            n_freq = self.nfft // 2 + 1
            nq = self.data["Nx"] * self.data["Ny"]
            self.qhat = np.zeros((n_freq, nq, self.nblocks), dtype=complex)
            for i in tqdm(range(n_freq), desc="Loading FFT blocks", unit="freq"):
                for j in range(self.nblocks):
                    self.qhat[i, :, j] = self._load_fft_block(i, j)
            print("FFT blocks loaded from cache successfully.")
            return
        if "q" not in self.data:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")
        print(f"Computing FFT with {self.nblocks} blocks...")
        self.qhat = blocksfft(self.data["q"], self.nfft, self.nblocks, self.novlap, blockwise_mean=getattr(self, "blockwise_mean", False), normvar=getattr(self, "normvar", False), window_norm=getattr(self, "window_norm", "power"), window_type=getattr(self, "window_type", "hamming"))
        print("Saving FFT blocks to disk cache...")
        n_freq = self.nfft // 2 + 1
        for i in tqdm(range(n_freq), desc="Saving FFT blocks", unit="freq"):
            for j in range(self.nblocks):
                self._save_fft_block(i, j, self.qhat[i, :, j])
        self._save_blocks_metadata()
        print("FFT blocks saved to disk cache successfully.")
        print("FFT computation complete.")

    def perform_spod(self):
        if self.qhat.size == 0:
            raise ValueError("FFT blocks not computed. Call compute_fft_blocks() first.")
        start_time = time.time()
        nq = self.data["Nx"] * self.data["Ny"]
        n_freq = self.nfft // 2 + 1
        self.lambda_values = np.zeros((n_freq, self.nblocks))
        self.phi = np.zeros((n_freq, nq, self.nblocks), dtype=complex)
        self.psi = np.zeros((n_freq, self.nblocks, self.nblocks), dtype=complex)
        print("Performing SPOD for each frequency...")
        for i in tqdm(range(n_freq), desc="SPOD Computation", unit="freq"):
            qhat_freq = self.qhat[i, :, :]
            phi_freq, lambda_freq, psi_freq = spod_function(qhat_freq, self.nblocks, self.dst, self.W, return_psi=True)
            self.phi[i, :, :] = phi_freq
            self.lambda_values[i, :] = lambda_freq
            self.psi[i, :, :] = psi_freq
        print(f"SPOD eigenvalue decomposition completed in {time.time() - start_time:.2f} seconds")

    def save_results(self):
        if self.phi.size == 0 or self.lambda_values.size == 0:
            raise ValueError("SPOD not performed. Call perform_spod() first.")
        from utils import make_result_filename

        save_path = os.path.join(self.results_dir, make_result_filename(self.data_root, self.nfft, self.overlap, self.data["Ns"], "spod"))
        print(f"Saving results to {save_path}")
        with h5py.File(save_path, "w") as fsnap:
            fsnap.create_dataset("Phi", data=self.phi, compression="gzip")
            fsnap.create_dataset("Lambda", data=self.lambda_values, compression="gzip")
            fsnap.create_dataset("St", data=self.St, compression="gzip")
            fsnap.create_dataset("x", data=self.data["x"], compression="gzip")
            fsnap.create_dataset("y", data=self.data["y"], compression="gzip")
            fsnap.attrs["Nfft"] = self.nfft
            fsnap.attrs["overlap"] = self.overlap
            fsnap.attrs["Ns"] = self.data["Ns"]
            fsnap.attrs["fs"] = self.fs
            fsnap.attrs["nblocks"] = self.nblocks
            fsnap.attrs["dt"] = self.data["dt"]

    def plot_eigenvalues(self, n_modes=10, highlight_St=None):
        """Plot the SPOD eigenvalue spectrum (energy vs. St) for leading modes.
        Optionally highlight a specific St value (e.g., the selected mode peak)."""
        if self.lambda_values.size == 0:
            raise ValueError("SPOD not performed. Call perform_spod() first.")
        print("Plotting SPOD eigenvalues...")
        plt.figure(figsize=(10, 6))
        plt.rc("text", usetex=USE_LATEX)
        plt.rc("font", family=FONT_FAMILY, size=FONT_SIZE)
        n_modes_to_plot = min(n_modes, self.lambda_values.shape[1])
        for i in range(n_modes_to_plot):
            plt.loglog(self.St, self.lambda_values[:, i], label=f"Mode {i + 1}", marker="o", markersize=3, linestyle="-")
        if highlight_St is not None:
            idx = np.argmin(np.abs(self.St - highlight_St))
            plt.scatter(self.St[idx], self.lambda_values[idx, 0], color="red", s=80, edgecolor="k", zorder=10, label=f"Peak St={self.St[idx]:.3f}")
        plt.legend()
        plt.xlabel(r"St")
        plt.ylabel(r"SPOD Eigenvalue $\lambda$")
        plt.title(r"SPOD Eigenvalue Spectrum vs. St")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        os.makedirs(self.figures_dir, exist_ok=True)
        filename = f"{self.data_root}_eigenvalues_Nfft{self.nfft}_ovlap{self.overlap}_{self.data['Ns']}snapshots.{FIG_FORMAT}"
        plt.savefig(os.path.join(self.figures_dir, os.path.basename(filename)), bbox_inches="tight", dpi=FIG_DPI, format=FIG_FORMAT)
        plt.close()
        print(f"Eigenvalue plot saved to {filename}")

    def plot_modes(self, st_target, n_modes=4):
        """Plot the real part of spatial SPOD modes (Phi) for a target Strouhal number."""
        if self.phi.size == 0:
            raise ValueError("SPOD not performed. Call perform_spod() first.")

        # Find the index (st_idx) of the Strouhal number in self.St closest to st_target
        st_idx = np.argmin(np.abs(self.St - st_target))
        st_value = self.St[st_idx]
        print(f"Plotting SPOD modes for St ≈ {st_value:.4f} (target: {st_target:.4f})...", flush=True)

        # Setup grid layout for subplots
        n_modes_to_plot = min(n_modes, self.phi.shape[2], 4)

        if n_modes_to_plot == 0:
            print("  Warning: No modes available to plot.")
            return

        # Set layout to 2x2 grid
        nrows = 2
        ncols = 2

        # Create figure and axes
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4), squeeze=False, constrained_layout=True)
        axes = axes.flatten()

        plt.rc("text", usetex=False)
        plt.rc("font", family="serif", size=10)
        cmap = plt.get_cmap("bwr")

        # Plot each requested mode
        max_abs_val = 0
        phi_modes_real = []
        for i in range(n_modes_to_plot):
            phi_real = self.phi[st_idx, :, i].real
            phi_2d = np.reshape(phi_real, (self.data["Nx"], self.data["Ny"])).T
            phi_modes_real.append(phi_2d)
            max_abs_val = max(max_abs_val, np.max(np.abs(phi_real)))

        interval = max_abs_val * 1.0
        levels = np.linspace(-interval, interval, 61)

        for i in range(n_modes_to_plot):
            phi_2d = phi_modes_real[i]
            ax = axes[i]
            im = ax.contourf(self.data["x"], self.data["y"], phi_2d, levels=levels, cmap=cmap, vmin=-interval, vmax=interval, extend="both")
            ax.set_aspect("equal")
            ax.set_title(f"Mode {i + 1}", size=12)
            if i >= (nrows - 1) * ncols:
                ax.set_xlabel("$x$", fontsize=12)
            if i % ncols == 0:
                ax.set_ylabel("$r$", fontsize=12)
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis="both", which="major", labelsize=10)

        for i in range(n_modes_to_plot, len(axes)):
            axes[i].axis("off")

        fig.colorbar(im, ax=axes[:n_modes_to_plot], shrink=0.8, label=r"Real($\Phi$)")
        fig.suptitle(rf"SPOD Modes at $St \approx {st_value:.4f}$", fontsize=14)

        # Ensure figures directory exists
        os.makedirs(self.figures_dir, exist_ok=True)
        filename = f"{self.data_root}_modes_Nfft{self.nfft}_ovlap{self.overlap}_{self.data['Ns']}snapshots_St{st_value:.4f}.png"
        fig.savefig(os.path.join(self.figures_dir, os.path.basename(filename)), bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  Mode plot saved to {filename}")

    def plot_eig_complex_plane(self, n_modes=4, st_target=None):
        """
        Plot the eigenvalues for a given Strouhal number (or the dominant one) in the complex plane:
        - x-axis: Real part
        - y-axis: Imaginary part
        Each eigenvalue corresponds to a mode at the selected frequency.
        """
        if self.lambda_values.size == 0 or self.phi.size == 0:
            raise ValueError("SPOD not performed. Call perform_spod() first.")
        # Determine which St to use
        if st_target is None:
            # Use the dominant peak as in plot_modes
            peaks, _ = find_peaks(self.lambda_values[:, 0])
            valid_peaks = [i for i in peaks if self.St[i] > 0.1]
            if valid_peaks:
                idx_peak = valid_peaks[np.argmax(self.lambda_values[valid_peaks, 0])]
                st_idx = idx_peak
                st_value = self.St[st_idx]
            else:
                st_idx = np.argmax(self.lambda_values[:, 0])
                st_value = self.St[st_idx]
        else:
            st_idx = np.argmin(np.abs(self.St - st_target))
            st_value = self.St[st_idx]
        # Get eigenvalues (complex) for the selected frequency
        # For SPOD, lambda_values are real, but phi (modes) are complex
        # We'll plot the first n_modes eigenvectors' first spatial point (as a simple example)
        eigvecs = self.phi[st_idx, :, :n_modes]  # shape: (space, n_modes)
        # For each mode, plot the real vs imag part of all spatial points
        plt.figure(figsize=(6, 6))
        for i in range(eigvecs.shape[1]):
            plt.scatter(eigvecs[:, i].real, eigvecs[:, i].imag, label=f"Mode {i + 1}", alpha=0.7, s=18)
        plt.xlabel("Real part of Mode")
        plt.ylabel("Imaginary part of Mode")
        plt.title(rf"SPOD Mode Eigenvectors in Complex Plane ($St \approx {st_value:.4f}$)")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        # Ensure figures directory exists
        os.makedirs(self.figures_dir, exist_ok=True)
        filename = f"{self.data_root}_modes_complex_plane_Nfft{self.nfft}_ovlap{self.overlap}_{self.data['Ns']}snapshots_St{st_value:.4f}.png"
        plt.savefig(os.path.join(self.figures_dir, os.path.basename(filename)), bbox_inches="tight", dpi=300)
        plt.close()
        print(f"  Complex plane plot saved to {filename}")

    def plot_time_coeffs(self, st_target=None, coeffs_idx=[0], n_blocks_plot=None):
        """
        Plot the real and imaginary parts of the time coefficients (psi) for selected modes at a given St.
        coeffs_idx: list of mode indices to plot (e.g., [0, 1])
        n_blocks_plot: if not None, limit number of blocks/time samples to plot
        """
        if not hasattr(self, "psi") or self.psi.size == 0:
            print("No time coefficients found. Run SPOD first.")
            return
        if st_target is None:
            # Use the dominant peak as in plot_modes
            peaks, _ = find_peaks(self.lambda_values[:, 0])
            valid_peaks = [i for i in peaks if self.St[i] > 0.1]
            if valid_peaks:
                idx_peak = valid_peaks[np.argmax(self.lambda_values[valid_peaks, 0])]
                st_idx = idx_peak
                st_value = self.St[st_idx]
            else:
                st_idx = np.argmax(self.lambda_values[:, 0])
                st_value = self.St[st_idx]
        else:
            st_idx = np.argmin(np.abs(self.St - st_target))
            st_value = self.St[st_idx]
        time_coeffs = self.psi[st_idx, :, :]  # shape: (nblocks, nblocks)
        if n_blocks_plot is not None:
            time_coeffs = time_coeffs[:n_blocks_plot, :]
        plt.figure(figsize=(8, 5))
        for idx in coeffs_idx:
            plt.plot(time_coeffs[:, idx].real, label=f"Mode {idx + 1} (real)")
            plt.plot(time_coeffs[:, idx].imag, "--", label=f"Mode {idx + 1} (imag)")
        plt.xlabel("Block index (time)")
        plt.ylabel("Time coefficient")
        plt.title(rf"SPOD Time Coefficients at $St \approx {st_value:.4f}$")
        plt.legend()
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        # Ensure figures directory exists
        os.makedirs(self.figures_dir, exist_ok=True)
        filename = f"{self.data_root}_time_coeffs_Nfft{self.nfft}_ovlap{self.overlap}_{self.data['Ns']}snapshots_St{st_value:.4f}.png"
        plt.savefig(os.path.join(self.figures_dir, os.path.basename(filename)), bbox_inches="tight", dpi=300)
        plt.close()
        print(f"  Time coefficients plot saved to {filename}")

    def plot_cumulative_energy_per_frequency(self, st_target=None):
        """Plot the cumulative energy captured by SPOD modes for a specific frequency."""
        if self.lambda_values is None or self.St is None:
            print("Eigenvalues or Strouhal numbers not available. Run perform_spod() first.")
            return

        if st_target is None:
            # Default to the frequency with max energy in the first mode if not specified
            dominant_freq_idx = np.argmax(self.lambda_values[:, 0])
            st_target = self.St[dominant_freq_idx]
            print(f"No st_target provided for cumulative energy plot. Using St = {st_target:.4f} (dominant for mode 0).")

        freq_idx = np.argmin(np.abs(self.St - st_target))
        actual_st = self.St[freq_idx]
        eigenvalues_at_freq = self.lambda_values[freq_idx, :]

        if eigenvalues_at_freq.size == 0:
            print(f"No eigenvalues found for St = {actual_st:.4f}.")
            return

        cumulative_energy = np.cumsum(eigenvalues_at_freq) / np.sum(eigenvalues_at_freq) * 100
        mode_indices = np.arange(1, len(eigenvalues_at_freq) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(mode_indices, cumulative_energy, "o-", linewidth=2, markersize=6)
        plt.xlabel("Number of Modes")
        plt.ylabel("Cumulative Energy (%)")
        plt.title(f"Cumulative Energy of SPOD Modes at St = {actual_st:.4f}")
        plt.grid(True, which="both", ls="--")
        plt.ylim(0, 105)
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_spod_cumulative_energy_St{actual_st:.4f}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"SPOD cumulative energy plot for St={actual_st:.4f} saved to {plot_filename}")

    def check_spatial_mode_orthogonality(self, st_target=None, tolerance=1e-9):
        """Check orthogonality of spatial SPOD modes for a specific frequency.
        Verifies Modes_f.conj().T @ W @ Modes_f is close to Identity.
        """
        if self.phi is None or self.phi.size == 0 or self.W is None or self.St is None:
            print("Modes (self.phi), weights, or Strouhal numbers not available. Run perform_spod() first.")
            return False

        if st_target is None:
            dominant_freq_idx = np.argmax(self.lambda_values[:, 0])
            st_target = self.St[dominant_freq_idx]
            print(f"No st_target provided for spatial orthogonality check. Using St = {st_target:.4f} (dominant for mode 0).")

        freq_idx = np.argmin(np.abs(self.St - st_target))
        actual_st = self.St[freq_idx]
        # self.phi has shape (Nfreq, Nspace, Nmodes_at_freq)
        # So, self.phi[freq_idx, :, :] gives (Nspace, Nmodes_at_freq)
        modes_at_freq = self.phi[freq_idx, :, :]
        n_modes_at_freq = modes_at_freq.shape[1]

        if n_modes_at_freq == 0:
            print(f"No modes found for St = {actual_st:.4f} to check orthogonality.")
            return False

        print(f"\nChecking SPOD spatial mode orthogonality for St = {actual_st:.4f} (Modes.conj().T @ W @ Modes)...")

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

        ortho_check_matrix = modes_at_freq.conj().T @ W_diag_matrix @ modes_at_freq
        identity_matrix = np.eye(n_modes_at_freq)

        diag_diff = np.abs(np.diag(ortho_check_matrix) - 1.0)
        max_diag_deviation = np.max(diag_diff)

        off_diag_mask = ~np.eye(n_modes_at_freq, dtype=bool)
        max_off_diag_val = np.max(np.abs(ortho_check_matrix[off_diag_mask])) if n_modes_at_freq > 1 else 0.0

        is_orthogonal = (max_diag_deviation < tolerance) and (max_off_diag_val < tolerance)

        print(f"  Max deviation of diagonal elements from 1: {max_diag_deviation:.2e}")
        print(f"  Max absolute value of off-diagonal elements: {max_off_diag_val:.2e}")
        if is_orthogonal:
            print(f"  SPOD spatial modes at St={actual_st:.4f} appear to be W-orthogonal.")
        else:
            print(f"  Warning: SPOD spatial modes at St={actual_st:.4f} may not be perfectly W-orthogonal.")

        plt.figure(figsize=(7, 6))
        # Use a consistent scaling for the colorbar if possible, e.g., vmin=-1, vmax=1 if expecting identity-like matrix
        # However, deviations can be small, so autoscaling might be better initially.
        abs_max_val = np.max(np.abs(ortho_check_matrix))
        plt.imshow(ortho_check_matrix.real, cmap=CMAP_DIV, vmin=-abs_max_val, vmax=abs_max_val)
        plt.colorbar(label="Value (Real Part)")
        plt.title(f"SPOD Spatial Mode Orthogonality (St={actual_st:.4f})")
        plt.xlabel("Mode Index")
        plt.ylabel("Mode Index")
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_spod_spatial_ortho_St{actual_st:.4f}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"  SPOD spatial orthogonality check plot for St={actual_st:.4f} saved to {plot_filename}")
        return is_orthogonal

    def check_temporal_coefficient_orthogonality(self, st_target=None, tolerance=1e-9):
        """Check orthogonality of SPOD temporal coefficients (Psi_f) for a specific frequency.
        Verifies Psi_f.conj().T @ Psi_f is close to Identity.
        """
        if self.psi is None or self.psi.size == 0 or self.St is None:
            print("Temporal coefficients (self.psi) or Strouhal numbers not available. Run perform_spod() first.")
            return False

        if st_target is None:
            dominant_freq_idx = np.argmax(self.lambda_values[:, 0])  # Assuming lambda_values exists if psi does
            st_target = self.St[dominant_freq_idx]
            print(f"No st_target provided for temporal coefficient orthogonality check. Using St = {st_target:.4f} (dominant for mode 0).")

        freq_idx = np.argmin(np.abs(self.St - st_target))
        actual_st = self.St[freq_idx]
        # self.psi has shape (Nfreq, Nmodes_at_freq, Nblocks_or_Nmodes_again)
        # For standard SPOD, Nmodes_at_freq is Nblocks. Psi_f is (Nblocks, Nblocks)
        # psi_at_freq should be (Nblocks, Nmodes_at_freq)
        psi_at_freq = self.psi[freq_idx, :, :]
        n_coeffs = psi_at_freq.shape[0]  # Should be nblocks
        n_modes_check = psi_at_freq.shape[1]  # Also nblocks for full rank

        if n_coeffs == 0 or n_modes_check == 0:
            print(f"No temporal coefficients found for St = {actual_st:.4f} to check orthogonality.")
            return False

        # We are checking L.conj().T @ L = I, where L is psi_at_freq
        # If n_modes_check (columns) < n_coeffs (rows), the result won't be square Identity of size n_coeffs
        # but rather Identity of size n_modes_check

        print(f"\nChecking SPOD temporal coefficient orthogonality for St = {actual_st:.4f} (Psi_f.conj().T @ Psi_f)...")
        ortho_check_matrix = psi_at_freq.conj().T @ psi_at_freq

        # The resulting matrix should be n_modes_check x n_modes_check (e.g. nblocks x nblocks)
        identity_matrix = np.eye(n_modes_check)

        diag_diff = np.abs(np.diag(ortho_check_matrix) - 1.0)
        max_diag_deviation = np.max(diag_diff) if diag_diff.size > 0 else 0.0

        off_diag_mask = ~np.eye(n_modes_check, dtype=bool)
        max_off_diag_val = np.max(np.abs(ortho_check_matrix[off_diag_mask])) if n_modes_check > 1 and off_diag_mask.any() else 0.0

        is_orthogonal = (max_diag_deviation < tolerance) and (max_off_diag_val < tolerance)

        print(f"  Matrix shape: {ortho_check_matrix.shape}")
        print(f"  Max deviation of diagonal elements from 1: {max_diag_deviation:.2e}")
        print(f"  Max absolute value of off-diagonal elements: {max_off_diag_val:.2e}")
        if is_orthogonal:
            print(f"  SPOD temporal coefficients at St={actual_st:.4f} appear to be orthogonal (Psi_f.conj().T @ Psi_f = I).")
        else:
            print(f"  Warning: SPOD temporal coefficients at St={actual_st:.4f} may not be perfectly orthogonal.")

        plt.figure(figsize=(7, 6))
        abs_max_val = np.max(np.abs(ortho_check_matrix.real))  # Plot real part for visualization
        plt.imshow(ortho_check_matrix.real, cmap=CMAP_DIV, vmin=-abs_max_val, vmax=abs_max_val)
        plt.colorbar(label="Value (Real Part)")
        plt.title(f"SPOD Temporal Coeff Orthogonality (St={actual_st:.4f})")
        plt.xlabel("Mode Index")
        plt.ylabel("Mode Index")
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_spod_temporal_ortho_St{actual_st:.4f}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"  SPOD temporal coefficient orthogonality check plot for St={actual_st:.4f} saved to {plot_filename}")
        return is_orthogonal

    def plot_reconstruction_error_per_frequency(self, st_target=None, max_modes_to_check=None):
        """Plot reconstruction error of frequency-domain data (q_hat_f) vs. number of SPOD modes.
        The error is the Frobenius norm of (q_hat_f - q_hat_f_reconstructed).
        """
        if not hasattr(self, "qhat") or self.qhat is None or self.qhat.size == 0:
            print("Frequency-domain data 'self.qhat' not available. Run compute_fft_blocks() or ensure cache is loaded.")
            return
        if self.phi is None or self.phi.size == 0 or self.St is None or self.W is None:
            print("Modes (self.phi), Strouhal numbers, or weights not available. Run perform_spod() first.")
            return

        if st_target is None:
            dominant_freq_idx = np.argmax(self.lambda_values[:, 0])
            st_target = self.St[dominant_freq_idx]
            print(f"No st_target provided for reconstruction error plot. Using St = {st_target:.4f} (dominant for mode 0).")

        freq_idx = np.argmin(np.abs(self.St - st_target))
        actual_st = self.St[freq_idx]

        qhat_at_freq = self.qhat[freq_idx, :, :]  # Shape (Nspace, Nblocks)
        modes_at_freq = self.phi[freq_idx, :, :]  # Shape (Nspace, Nmodes_at_this_freq)
        n_available_modes = modes_at_freq.shape[1]

        if max_modes_to_check is None:
            max_modes_to_check = n_available_modes
        else:
            max_modes_to_check = min(max_modes_to_check, n_available_modes)

        if self.W.ndim == 1:
            W_diag = np.diag(self.W)
        elif self.W.ndim == 2 and self.W.shape[0] == self.W.shape[1] and np.allclose(self.W, np.diag(np.diag(self.W))):
            W_diag = self.W
        elif self.W.ndim == 2 and self.W.shape[1] == 1:
            W_diag = np.diag(self.W.flatten())
        else:
            print(f"  Warning: Unexpected W shape for reconstruction error: {self.W.shape}. Using identity for W_diag.")
            W_diag = np.eye(qhat_at_freq.shape[0])  # Fallback, may not be correct

        errors = []
        mode_counts = range(1, max_modes_to_check + 1)

        norm_qhat_original = np.linalg.norm(qhat_at_freq, "fro")
        if norm_qhat_original == 0:  # Avoid division by zero if qhat_at_freq is all zeros
            print(f"Warning: Original q_hat data at St={actual_st:.4f} has zero norm. Cannot compute relative error.")
            # errors will remain empty, plot will be skipped or show zero error
            return

        print(f"\nCalculating reconstruction error for q_hat at St = {actual_st:.4f}...")
        for k in mode_counts:
            current_modes = modes_at_freq[:, :k]
            # Project qhat_at_freq onto the current_modes (W-weighted projection)
            # Coeffs = Modes_k^H * W * Q_hat_f
            projection_coeffs = current_modes.conj().T @ W_diag @ qhat_at_freq
            # Reconstruct: Q_hat_f_rec = Modes_k * Coeffs
            qhat_reconstructed = current_modes @ projection_coeffs

            error = np.linalg.norm(qhat_at_freq - qhat_reconstructed, "fro") / norm_qhat_original
            errors.append(error)
            if k % (max_modes_to_check // 10 if max_modes_to_check > 10 else 1) == 0:
                print(f"  Computed error for k={k} modes: {error:.3e}")

        plt.figure(figsize=(8, 5))
        plt.plot(list(mode_counts), errors, "o-", linewidth=2, markersize=5)
        plt.xlabel("Number of SPOD Modes")
        plt.ylabel("Relative Reconstruction Error (Frobenius Norm)")
        plt.title(f"SPOD Reconstruction Error of $q_{{hat}}$ at St = {actual_st:.4f}")
        plt.grid(True, which="both", ls="--")
        plt.yscale("log")  # Errors often span orders of magnitude
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_spod_reconstruction_error_St{actual_st:.4f}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"SPOD reconstruction error plot for St={actual_st:.4f} saved to {plot_filename}")

    def plot_reconstruction_comparison_per_frequency(self, st_target=None, block_idx_to_plot=0, modes_counts_to_compare=None):
        """Visually compare original q_hat_f[:, block_idx] with reconstructions using varying numbers of SPOD modes."""
        if not hasattr(self, "qhat") or self.qhat is None or self.qhat.size == 0:
            print("Frequency-domain data 'self.qhat' not available. Run compute_fft_blocks() or ensure cache is loaded.")
            return
        if self.phi is None or self.phi.size == 0 or self.St is None or self.W is None:
            print("Modes (self.phi), Strouhal numbers, or weights not available. Run perform_spod() first.")
            return

        if st_target is None:
            dominant_freq_idx = np.argmax(self.lambda_values[:, 0])
            st_target = self.St[dominant_freq_idx]
            print(f"No st_target provided for reconstruction comparison. Using St = {st_target:.4f} (dominant for mode 0).")

        freq_idx = np.argmin(np.abs(self.St - st_target))
        actual_st = self.St[freq_idx]

        qhat_at_freq = self.qhat[freq_idx, :, :]  # Shape (Nspace, Nblocks)
        modes_at_freq = self.phi[freq_idx, :, :]  # Shape (Nspace, Nmodes_at_this_freq)
        n_available_modes = modes_at_freq.shape[1]
        n_blocks = qhat_at_freq.shape[1]

        if block_idx_to_plot < 0 or block_idx_to_plot >= n_blocks:
            print(f"Error: block_idx_to_plot={block_idx_to_plot} is out of range (0-{n_blocks - 1}). Using block 0.")
            block_idx_to_plot = 0

        if modes_counts_to_compare is None:
            modes_counts_to_compare = sorted(list(set([1, n_available_modes // 4 if n_available_modes > 3 else 1, n_available_modes // 2 if n_available_modes > 1 else 1, n_available_modes])))
            # Ensure values are at least 1 and unique
            modes_counts_to_compare = [max(1, m) for m in modes_counts_to_compare]
            modes_counts_to_compare = sorted(list(set(m for m in modes_counts_to_compare if m <= n_available_modes)))
            if not modes_counts_to_compare:
                modes_counts_to_compare = [n_available_modes]

        if self.W.ndim == 1:
            W_diag = np.diag(self.W)
        # ... (W_diag handling as in previous method) ...
        elif self.W.ndim == 2 and self.W.shape[0] == self.W.shape[1] and np.allclose(self.W, np.diag(np.diag(self.W))):
            W_diag = self.W
        elif self.W.ndim == 2 and self.W.shape[1] == 1:
            W_diag = np.diag(self.W.flatten())
        else:
            print(f"  Warning: Unexpected W shape for reconstruction comparison: {self.W.shape}. Using identity.")
            W_diag = np.eye(qhat_at_freq.shape[0])

        original_qhat_block = qhat_at_freq[:, block_idx_to_plot]  # Shape (Nspace)

        Nx = self.data.get("Nx")
        Ny = self.data.get("Ny")

        if Nx is None or Ny is None or original_qhat_block.shape[0] != Nx * Ny:
            print(f"Warning: Cannot reshape data for 2D plot in SPOD reconstruction comparison. Nx={Nx}, Ny={Ny}, DataSize={original_qhat_block.shape[0]}. Skipping 2D plot.")
            # Potentially add a 1D plot fallback here if desired or simply return
            return

        original_qhat_block_spatial = original_qhat_block.real.reshape(Nx, Ny)

        num_plots = len(modes_counts_to_compare) + 1
        # Try to make a somewhat square layout of subplots
        ncols = int(np.ceil(np.sqrt(num_plots)))
        nrows = int(np.ceil(num_plots / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5), squeeze=False)
        axes_flat = axes.flatten()

        # Plot Original
        ax = axes_flat[0]
        im = ax.contourf(self.data["x"], self.data["y"], original_qhat_block_spatial.T, cmap=CMAP_DIV, levels=100)
        fig.colorbar(im, ax=ax)
        ax.set_title(f"Original $Re(q_{{hat}})$ Block {block_idx_to_plot}\nSt={actual_st:.4f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        print(f"\nPlotting reconstruction comparison for q_hat (Block {block_idx_to_plot}) at St = {actual_st:.4f}...")
        for i, k in enumerate(modes_counts_to_compare):
            if k > n_available_modes:
                continue  # Should not happen with filtering above
            current_modes = modes_at_freq[:, :k]
            projection_coeffs = current_modes.conj().T @ W_diag @ original_qhat_block  # Coeffs for this specific block
            reconstructed_qhat_block = current_modes @ projection_coeffs
            reconstructed_qhat_block_spatial = reconstructed_qhat_block.real.reshape(Nx, Ny)

            ax = axes_flat[i + 1]
            im = ax.contourf(self.data["x"], self.data["y"], reconstructed_qhat_block_spatial.T, cmap=CMAP_DIV, levels=100)
            fig.colorbar(im, ax=ax)
            ax.set_title(f"Reconstructed ({k} modes)")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            print(f"  Plotted reconstruction with k={k} modes.")

        # Hide any unused subplots
        for j in range(i + 2, nrows * ncols):
            fig.delaxes(axes_flat[j])

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for suptitle
        fig.suptitle(f"SPOD $Re(q_{{hat}})$ Reconstruction Comparison at St={actual_st:.4f}, Block {block_idx_to_plot}", fontsize=14)
        plot_filename = os.path.join(self.figures_dir, f"{self.data_root}_spod_reconstruction_compare_St{actual_st:.4f}_Block{block_idx_to_plot}.png")
        plt.savefig(plot_filename)
        plt.close(fig)
        print(f"SPOD reconstruction comparison plot for St={actual_st:.4f}, Block {block_idx_to_plot} saved to {plot_filename}")

    def run_analysis(self, plot_st_target=None, plot_n_modes_eig=10, plot_n_modes_spatial=4, use_disk_cache=True, perform_verifications=True):
        """
        Run the full SPOD analysis pipeline: load, preprocess, compute FFTs,
        perform SPOD, save results, and plot.

        Args:
            plot_st_target (float, optional): Target Strouhal number for mode plots.
                                              If None, uses the St with max energy in mode 1.
            plot_n_modes_eig (int): Number of modes to show in the eigenvalue vs. St plot.
            plot_n_modes_spatial (int): Number of spatial modes to plot for the target St.
            use_disk_cache (bool): Whether to use disk caching for FFT blocks.
            perform_verifications (bool): Whether to run and plot additional verification checks.
        """
        print("Starting SPOD analysis pipeline...")
        start_total_time = time.time()
        try:
            print("\nStep 1/4: Loading and preprocessing data...")
            self.load_and_preprocess()
            print("\nStep 2/4: Computing FFT blocks...")
            if not use_disk_cache:
                import shutil

                blocks_dir = self._get_blocks_directory()
                if os.path.exists(blocks_dir):
                    shutil.rmtree(blocks_dir)
                    print("Removed existing cache, will recompute FFT blocks.")
            self.compute_fft_blocks()
            print("\nStep 3/4: Performing SPOD eigenvalue decomposition...")
            self.perform_spod()
            print("\nStep 4/4: Saving results and generating visualizations...")
            self.save_results()
            print("\nGenerating visualizations...")
            highlight_St = None
            if plot_st_target is None:
                # Find all local maxima (peaks) in mode 1
                peaks, _ = find_peaks(self.lambda_values[:, 0])
                # Filter peaks to those with St > 0.1
                valid_peaks = [i for i in peaks if self.St[i] > 0.1]
                if valid_peaks:
                    # Take the peak with the highest eigenvalue among valid peaks
                    idx_peak = valid_peaks[np.argmax(self.lambda_values[valid_peaks, 0])]
                    plot_st_target = self.St[idx_peak]
                    highlight_St = plot_st_target
                    print(f"No target St provided, plotting modes for dominant PEAK St > 0.1: {plot_st_target:.4f}")
                else:
                    # Fallback: just use the global maximum
                    dominant_freq_idx = np.argmax(self.lambda_values[:, 0])
                    plot_st_target = self.St[dominant_freq_idx]
                    highlight_St = plot_st_target
                    print(f"No St > 0.1 found, plotting modes for dominant St: {plot_st_target:.4f}")
            else:
                highlight_St = plot_st_target

            self.plot_eigenvalues(n_modes=min(plot_n_modes_eig, self.nblocks), highlight_St=highlight_St)
            self.plot_modes(st_target=plot_st_target, n_modes=min(plot_n_modes_spatial, self.nblocks))
            self.plot_eig_complex_plane(n_modes=min(plot_n_modes_spatial, self.nblocks), st_target=plot_st_target)
            self.plot_time_coeffs(st_target=plot_st_target)  # Default plots mode 0

            if perform_verifications:
                print("\n--- Performing Additional Verifications ---")
                # Use the same plot_st_target for verifications, or the dominant one if None
                verification_st_target = plot_st_target
                if verification_st_target is None:
                    # Find dominant St if not provided (similar to how plot_modes does it)
                    if self.lambda_values is not None and self.St is not None:
                        valid_peaks_indices = np.where(self.St > 0.1)[0]
                        if valid_peaks_indices.size > 0:
                            lambda_at_valid_peaks = self.lambda_values[valid_peaks_indices, 0]
                            if lambda_at_valid_peaks.size > 0:
                                dominant_peak_idx_in_subset = np.argmax(lambda_at_valid_peaks)
                                actual_dominant_idx = valid_peaks_indices[dominant_peak_idx_in_subset]
                                verification_st_target = self.St[actual_dominant_idx]
                        if verification_st_target is None:  # Fallback if no peak > 0.1 or other issues
                            verification_st_target = self.St[np.argmax(self.lambda_values[:, 0])]

                self.plot_cumulative_energy_per_frequency(st_target=verification_st_target)
                self.check_spatial_mode_orthogonality(st_target=verification_st_target)
                self.check_temporal_coefficient_orthogonality(st_target=verification_st_target)
                if hasattr(self, "qhat") and self.qhat is not None and self.qhat.size > 0:
                    self.plot_reconstruction_error_per_frequency(st_target=verification_st_target)
                    self.plot_reconstruction_comparison_per_frequency(st_target=verification_st_target, block_idx_to_plot=0)  # Plot for block 0 by default
                else:
                    print("Skipping q_hat reconstruction plots as self.qhat is not available.")

            end_total_time = time.time()
            print(f"\nSPOD analysis completed successfully in {end_total_time - start_total_time:.2f} seconds.")
        except Exception as e:
            print(f"\nError during SPOD analysis: {str(e)}")
            import traceback

            print("\nTraceback:")
            traceback.print_exc()
            print("\nPlease check the inputs and try again.")
            raise

    """Class for performing Spectral Proper Orthogonal Decomposition (SPOD) analysis.

    **Expected Input Data Structure:**

    The core analysis methods (compute_fft_blocks, perform_spod) expect the 
    primary input data (e.g., pressure, velocity) to be preprocessed into a 
    2D NumPy array `q` stored in `self.data['q']`. 

    The required format for `self.data['q']` is:
        - Shape: `(Ns, Nspatial)`
        - `Ns`: Number of time snapshots.
        - `Nspatial`: Total number of spatial points (e.g., Nx * Ny for a 2D grid).
        - The first dimension (axis 0) must represent time.
        - The second dimension (axis 1) must represent the flattened spatial domain.
        - **Crucially, the flattening of the spatial points must be consistent 
          across all time snapshots.**

    Additionally, the following need to be provided in `self.data`:
        - `x`: 1D NumPy array of coordinates for the first spatial dimension (length Nx).
        - `y`: 1D NumPy array of coordinates for the second spatial dimension (length Ny).
        - `Nx`, `Ny`: Integers representing the dimensions of the original spatial grid.
        - `dt`: Float representing the time step between snapshots.

    The `load_jetles_data` function in this script handles loading from a specific 
    HDF5 format and performs the necessary transpose and reshape. If loading from
    a different source (like CGNS), you would need to implement a similar loading 
    function that extracts the data and metadata, then reshapes the primary data 
    variable into the required `(Ns, Nspatial)` format before assigning it and 
    the other metadata to `self.data`.
    """

    def __init__(self, file_path, nfft=128, overlap=0.5, results_dir=RESULTS_DIR_SPOD, figures_dir=FIGURES_DIR_SPOD, blockwise_mean=False, normvar=False, window_norm="power", window_type="hamming", data_loader=None, spatial_weight_type="auto"):
        super().__init__(file_path=file_path, nfft=nfft, overlap=overlap, results_dir=results_dir, figures_dir=figures_dir, data_loader=data_loader, spatial_weight_type=spatial_weight_type)
        self.blockwise_mean = blockwise_mean
        self.normvar = normvar
        self.window_norm = window_norm
        self.window_type = window_type
        # SPOD-specific fields
        self.phi = np.array([])
        self.lambda_values = np.array([])
        self.frequencies = np.array([])
        self.psi = np.array([])
        self.St = np.array([])
        self.dst = 0.0
        self.L = 1.0
        self.U = 1.0
        """Initialize the SPOD analyzer.

        Args:
            file_path (str): Path to the HDF5 data file.
            nfft (int): Number of snapshots per FFT block.
            overlap (float): Overlap fraction between blocks (0 to < 1).
            results_dir (str): Directory to save numerical results.
            figures_dir (str): Directory to save plots.
            blockwise_mean (bool): If True, use blockwise mean subtraction.
            normvar (bool): If True, normalize by variance.
            window_norm (str): Normalization for window function ('power' or 'amplitude').
            window_type (str): Type of window ('hamming', 'hann', 'rectangular').
            data_loader (callable, optional): Custom data loading function.
            spatial_weight_type (str): Type of spatial weighting ('polar', 'uniform', 'auto').
        """

    def load_and_preprocess(self):
        super().load_and_preprocess()
        # Set normalization constants for Strouhal number
        if "cavity" in self.file_path.lower():
            self.L = 0.0381
            self.U = 230.0
            print(f"Cavity case detected: Using L={self.L} m, U={self.U} m/s for Strouhal normalization.")
        else:
            self.L = 1.0
            self.U = 1.0
            print("Jet case or unknown: Using L=1, U=1 for Strouhal normalization.")
        # Calculate Strouhal vector
        self.fs = 1 / self.data["dt"]
        f = np.linspace(0, self.fs - self.fs / self.nfft, self.nfft)
        St = f * self.L / self.U
        self.St = St[0 : self.nfft // 2 + 1]
        self.dst = self.St[1] - self.St[0]
        self.strouhal = St

    def perform_spod(self):
        """Perform SPOD analysis (eigenvalue decomposition) for each frequency."""
        if self.qhat.size == 0:
            raise ValueError("FFT blocks not computed. Call compute_fft_blocks() first.")

        start_time = time.time()
        # Total number of spatial points
        nq = self.data["Nx"] * self.data["Ny"]
        # Number of frequencies to compute (only positive frequencies 0 to fs/2)
        n_freq = self.nfft // 2 + 1

        # Initialize arrays to store results
        # Eigenvalues (energy) for each mode at each frequency
        self.lambda_values = np.zeros((n_freq, self.nblocks))
        # Spatial modes for each mode at each frequency
        self.phi = np.zeros((n_freq, nq, self.nblocks), dtype=complex)
        # Time coefficients for each mode at each frequency
        self.psi = np.zeros((n_freq, self.nblocks, self.nblocks), dtype=complex)

        print("Performing SPOD for each frequency...")
        # Compute SPOD for each frequency f in the one-sided spectrum (0 to fs/2)
        for i in range(n_freq):
            # Extract FFT data for the current frequency i
            # qhat has shape [frequency, space, block]
            # We need [space, block] for spod_function
            qhat_freq = self.qhat[i, :, :]

            # Call the core SPOD function for this frequency
            phi_freq, lambda_freq, psi_freq = spod_function(qhat_freq, self.nblocks, self.dst, self.W, return_psi=True)

            # Store results
            self.phi[i, :, :] = phi_freq
            self.lambda_values[i, :] = lambda_freq
            self.psi[i, :, :] = psi_freq

            # Print progress (optional)
            if (i + 1) % 10 == 0 or i == n_freq - 1:
                print(f"  Processed frequency {i + 1}/{n_freq} (St = {self.St[i]:.4f})")

        print(f"SPOD eigenvalue decomposition completed in {time.time() - start_time:.2f} seconds")

    def run_analysis(self, plot_st_target=None, plot_n_modes_eig=10, plot_n_modes_spatial=4, use_disk_cache=True, perform_verifications=True):
        print("Starting SPOD analysis...")
        start_total_time = time.time()

        self.load_and_preprocess()
        self.compute_fft_blocks()
        self.perform_spod()
        self.save_results()

        print("\nGenerating visualizations...")
        highlight_St = None

        if plot_st_target is None:
            # Find all local maxima (peaks) in mode 1
            peaks, _ = find_peaks(self.lambda_values[:, 0])
            # Filter peaks to those with St > 0.1
            valid_peaks = [i for i in peaks if self.St[i] > 0.1]
            if valid_peaks:
                # Take the peak with the highest eigenvalue among valid peaks
                idx_peak = valid_peaks[np.argmax(self.lambda_values[valid_peaks, 0])]
                plot_st_target = self.St[idx_peak]
                highlight_St = plot_st_target
                print(f"No target St provided, plotting modes for dominant PEAK St > 0.1: {plot_st_target:.4f}")
            else:
                # Fallback: just use the global maximum
                dominant_freq_idx = np.argmax(self.lambda_values[:, 0])
                plot_st_target = self.St[dominant_freq_idx]
                highlight_St = plot_st_target
                print(f"No St > 0.1 found, plotting modes for dominant St: {plot_st_target:.4f}")
        else:
            highlight_St = plot_st_target

        self.plot_eigenvalues(n_modes=min(plot_n_modes_eig, self.nblocks), highlight_St=highlight_St)
        self.plot_modes(st_target=plot_st_target, n_modes=min(plot_n_modes_spatial, self.nblocks))
        self.plot_eig_complex_plane(n_modes=min(plot_n_modes_spatial, self.nblocks), st_target=plot_st_target)
        self.plot_time_coeffs(st_target=plot_st_target)
        if perform_verifications:
            print("\n--- Performing Additional Verifications ---")
            # Use the same plot_st_target for verifications, or the dominant one if None
            verification_st_target = plot_st_target
            if verification_st_target is None:
                # Find dominant St if not provided (similar to how plot_modes does it)
                if self.lambda_values is not None and self.St is not None:
                    valid_peaks_indices = np.where(self.St > 0.1)[0]
                    if valid_peaks_indices.size > 0:
                        lambda_at_valid_peaks = self.lambda_values[valid_peaks_indices, 0]
                        if lambda_at_valid_peaks.size > 0:
                            dominant_peak_idx_in_subset = np.argmax(lambda_at_valid_peaks)
                            actual_dominant_idx = valid_peaks_indices[dominant_peak_idx_in_subset]
                            verification_st_target = self.St[actual_dominant_idx]
                    if verification_st_target is None:  # Fallback if no peak > 0.1 or other issues
                        verification_st_target = self.St[np.argmax(self.lambda_values[:, 0])]

            self.plot_cumulative_energy_per_frequency(st_target=verification_st_target)
            self.check_spatial_mode_orthogonality(st_target=verification_st_target)
            self.check_temporal_coefficient_orthogonality(st_target=verification_st_target)
            if hasattr(self, "qhat") and self.qhat is not None and self.qhat.size > 0:
                self.plot_reconstruction_error_per_frequency(st_target=verification_st_target)
                self.plot_reconstruction_comparison_per_frequency(st_target=verification_st_target, block_idx_to_plot=0)  # Plot for block 0 by default
            else:
                print("Skipping q_hat reconstruction plots as self.qhat is not available.")

        end_total_time = time.time()
        print(f"\nSPOD analysis completed successfully in {end_total_time - start_total_time:.2f} seconds.")


# Example usage when the script is run directly
if __name__ == "__main__":
    # --- Configuration ---
    data_file = "./data/jetLES_small.mat"  # Updated data path
    # data_file = "./data/jetLES.mat" # Path to your data file
    # data_file = "./data/cavityPIV.mat" # Path to your data file

    results_dir = "./preprocess"  # Directory for HDF5 results
    figures_dir = "./figs"  # Directory for plots
    # Default parameters
    nfft_param = 128  # FFT block size
    overlap_param = 0.5  # Overlap fraction (50%)
    # Optional: Specify a frequency target for mode plots
    # If None, it will plot modes at the frequency with peak energy in Mode 1.
    freq_target_for_plots = None
    # ---------------------
    # Set case-specific parameters for cavity (to match MATLAB reference)
    if "cavity" in data_file.lower():
        nfft_param = 256
        overlap_param = 128 / 256  # 0.5
        window_type_param = "sine"  # Sine window for cavity, as in MATLAB
        spatial_weight_type_param = "uniform"  # Rectangular grid for cavity
        print("Cavity case detected: Using nfft=256, overlap=128 (50%), sine window to match MATLAB reference.")
    elif "jet" in data_file.lower():
        window_type_param = "hamming"  # Default for jet case
        spatial_weight_type_param = "polar"  # Cylindrical grid for jet
        print("Jet case detected: Using default nfft, overlap, and Hamming window.")
    else:
        window_type_param = "hamming"
        spatial_weight_type_param = "uniform"
        print("Unknown case: Using default nfft, overlap, and Hamming window.")
    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file not found at '{data_file}'")
        exit()

    # Set case-specific data loader
    if "cavity" in data_file.lower():
        data_loader_param = load_mat_data
        print("Cavity case detected: Using load_mat_data.")
    elif "jet" in data_file.lower():
        data_loader_param = load_jetles_data
        print("Jet case detected: Using load_jetles_data.")
    else:
        data_loader_param = load_mat_data
        print("Unknown case: Using load_mat_data.")

    # Create SPOD analyzer instance
    spod_analyzer = SPODAnalyzer(file_path=data_file, nfft=nfft_param, overlap=overlap_param, results_dir=results_dir, figures_dir=figures_dir, window_type=window_type_param, data_loader=data_loader_param, spatial_weight_type=spatial_weight_type_param)

    # Run the full analysis and plotting pipeline
    spod_analyzer.run_analysis(
        plot_st_target=freq_target_for_plots,
        plot_n_modes_eig=10,  # Number of modes in eigenvalue plot
        plot_n_modes_spatial=4,  # Number of modes in spatial plot
    )
