#!/usr/bin/env python3
"""
Common utilities for modal decomposition methods.

All imports are centralized here to keep the code clean and consistent.
"""

from configs import *
from fft.fft_backends import get_fft_func


def make_result_filename(root, nfft, overlap, Ns, analysis):
    """
    Generate a harmonized result filename for analysis outputs.
    Args:
        root (str): Base name of the dataset (no extension)
        nfft (int): FFT block size
        overlap (float): Overlap fraction (0-1)
        Ns (int): Number of snapshots
        analysis (str): Analysis type (e.g., 'spod', 'bsmd')
    Returns:
        str: Result filename (always .hdf5)
    """
    return f"{root}_Nfft{nfft}_ovlap{overlap}_{Ns}snapshots_{analysis}.hdf5"


def load_jetles_data(file_path):
    """Load and preprocess data from HDF5 file with JetLES format."""
    print(f"Loading data from {file_path}")
    with h5py.File(file_path, "r") as fread:
        # Get dimensions and data
        q = fread["p"][:]  # pressure field (or other variable)
        x = fread["x"][:, 0]  # x-coordinates (axial)
        y = fread["r"][0, :]  # y-coordinates (radial)
        dt = fread["dt"][0][0]  # time step

    # Transpose q: original shape (Nx, Ny, Ns) -> (Ns, Nx, Ny)
    # Reshape q: flatten spatial dimensions -> (Ns, Nx * Ny) for SPOD
    q = np.transpose(q, (2, 0, 1))
    Nx, Ny = x.shape[0], y.shape[0]
    Ns = q.shape[0]  # number of snapshots (time steps)
    q_reshaped = q.reshape(Ns, Nx * Ny)  # Use reshape for safety

    # Return all the data in a dictionary for easy access
    return {
        "q": q_reshaped,  # Data reshaped for SPOD: [time, space]
        "x": x,  # Axial coordinates
        "y": y,  # Radial coordinates
        "dt": dt,  # Time step
        "Nx": Nx,  # Number of points in x
        "Ny": Ny,  # Number of points in y (radial)
        "Ns": Ns,  # Number of snapshots
    }


def load_mat_data(file_path):
    """Flexible loader for .mat files with different variable names and shapes."""
    with h5py.File(file_path, "r") as fread:
        # Try common variable names for the main field
        for var in ["p", "u", "v", "data"]:
            if var in fread:
                q = fread[var][:]
                break
        else:
            raise KeyError("No recognized data variable ('p', 'u', 'v', 'data') in file.")
        x = fread["x"][:]
        y = fread["y"][:]
        dt = np.array(fread["dt"])[0][0] if "dt" in fread else 1.0
    print(f"Loaded variable shape: q={q.shape}, x={x.shape}, y={y.shape}")

    # If x and y are 2D (meshgrid), reduce to 1D vectors
    if x.ndim == 2:
        x_vec = x[:, 0]
    else:
        x_vec = x
    if y.ndim == 2:
        y_vec = y[0, :]
    else:
        y_vec = y
    Nx, Ny = x_vec.shape[0], y_vec.shape[0]

    # Special handling for (Nx, Ny, Ns)
    if q.shape == (Nx, Ny, q.shape[2]):
        Ns = q.shape[2]
        q = np.transpose(q, (2, 0, 1))  # (Ns, Nx, Ny)
        q_reshaped = q.reshape(Ns, Nx * Ny)
        print(f"Data interpreted as (Nx, Ny, Ns) and transposed to (Ns, Nx, Ny) = {q.shape}")
    # Standard (Ns, Nx, Ny)
    elif q.shape == (q.shape[0], Nx, Ny):
        Ns = q.shape[0]
        q_reshaped = q.reshape(Ns, Nx * Ny)
        print(f"Data interpreted as (Ns, Nx, Ny) = {q.shape}")
    # Try all permutations if above does not match
    else:
        for axes in [(0, 1, 2), (2, 0, 1), (2, 1, 0), (0, 2, 1), (1, 0, 2), (1, 2, 0)]:
            try:
                arr = np.transpose(q, axes)
                Ns, Nxx, Nyy = arr.shape
                if Nxx == Nx and Nyy == Ny:
                    q_reshaped = arr.reshape(Ns, Nx * Ny)
                    print(f"Data interpreted as (Ns, Nx, Ny) = {arr.shape} via permutation {axes}")
                    break
            except Exception:
                continue
        else:
            # Try if already 2D (Ns, Nspace)
            if q.ndim == 2 and q.shape[1] == Nx * Ny:
                q_reshaped = q
                Ns = q.shape[0]
                print(f"Data interpreted as (Ns, Nspace) = {q.shape}")
            else:
                raise ValueError(f"Cannot interpret data shape: q={q.shape}, x={x.shape}, y={y.shape}. Please check the file.")
    return {"q": q_reshaped, "x": x_vec, "y": y_vec, "dt": dt, "Nx": Nx, "Ny": Ny, "Ns": q_reshaped.shape[0]}


def load_data(file_path):
    """Smart data loader that selects appropriate loader based on file name."""
    if "jet" in file_path.lower():
        return load_jetles_data(file_path)
    else:
        return load_mat_data(file_path)


def calculate_polar_weights(x, y):
    """Calculate integration weights for a 2D cylindrical grid (x, r)."""
    Nx, Ny = x.shape[0], y.shape[0]

    # Calculate y-direction (r-direction) integration weights (Wy)
    Wy = np.zeros((Ny, 1))

    # First point (centerline)
    if Ny > 1:
        y_mid_right = (y[0] + y[1]) / 2
        Wy[0] = np.pi * y_mid_right**2
    else:
        Wy[0] = np.pi * y[0] ** 2

    # Middle points
    for i in range(1, Ny - 1):
        y_mid_left = (y[i - 1] + y[i]) / 2
        y_mid_right = (y[i] + y[i + 1]) / 2
        Wy[i] = np.pi * (y_mid_right**2 - y_mid_left**2)

    # Last point
    if Ny > 1:
        y_mid_left = (y[-2] + y[-1]) / 2
        Wy[Ny - 1] = np.pi * (y[-1] ** 2 - y_mid_left**2)

    # Calculate x-direction integration weights (Wx)
    Wx = np.zeros((Nx, 1))

    # First point
    if Nx > 1:
        Wx[0] = (x[1] - x[0]) / 2
    else:
        Wx[0] = 1.0

    # Middle points
    for i in range(1, Nx - 1):
        Wx[i] = (x[i + 1] - x[i - 1]) / 2

    # Last point
    if Nx > 1:
        Wx[Nx - 1] = (x[Nx - 1] - x[Nx - 2]) / 2

    # Combine weights
    W = np.reshape(Wx @ np.transpose(Wy), (Nx * Ny, 1))

    return W


def calculate_uniform_weights(x, y):
    """Return uniform weights for a 2D grid (Cartesian)."""
    Nx, Ny = x.shape[0], y.shape[0]
    return np.ones((Nx * Ny, 1))


def sine_window(n):
    """Return a sine window of length n."""
    return np.sin(np.pi * (np.arange(n) + 0.5) / n)


def blocksfft(q, nfft, nblocks, novlap, blockwise_mean=False, normvar=False, window_norm="power", window_type="hamming"):
    """
    Compute blocked FFT using Welch's method for CSD estimation.

    Parameters:
    q (np.ndarray): Input data [time, space]
    nfft (int): Number of FFT points
    nblocks (int): Number of blocks
    novlap (int): Number of overlapping points between blocks
    blockwise_mean (bool): Subtract blockwise mean if True
    normvar (bool): Normalize variance if True
    window_norm (str): Window normalization type ('amplitude' or 'power')
    window_type (str): Window type ('hamming' or 'sine')

    Returns:
    q_hat (np.ndarray): FFT coefficients [freq, space, block]

    ---
    IMPORTANT:
    - This function assumes the FFT backend (numpy, scipy, pyfftw, etc.) does NOT normalize the FFT by default (which is true for standard backends).
    - If you use a backend or option that applies normalization (e.g., norm='ortho'), REMOVE the division by nfft below to avoid double normalization.
    - For correct SPOD scaling, ensure that dst (frequency resolution) is set as fs / nfft, where fs is the sampling frequency.
    ---
    """
    # Select window function
    if window_type == "sine":
        window = sine_window(nfft)
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

    # Process each block
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
        fft_func = get_fft_func()
        # Compute full FFT
        full_fft_result = fft_func(block_centered * window_broadcast, axis=0)

        # --- Normalization explanation ---
        # The FFT result is normalized by:
        #   - cw: window normalization constant (either amplitude or power, see above)
        #   - nfft: block length, to match standard FFT conventions (NumPy/SciPy FFTs are unnormalized by default)
        # This ensures that:
        #   - For 'power' normalization, the output power spectrum is consistent with Parseval's theorem (total energy preserved)
        #   - For 'amplitude' normalization, the amplitude spectrum matches the input amplitude scaling
        # Note: No further normalization by the total signal length is needed here, because Welch's method treats each block independently.
        # When combining blocks (e.g., averaging periodograms), normalization by the number of blocks is handled outside this function.
        # Store only the one-sided spectrum (first n_freq_out points)
        q_hat[:, :, iblk] = (cw / nfft) * full_fft_result[:n_freq_out, :]

    return q_hat


def auto_detect_weight_type(file_path):
    """Auto-detect weight type based on file name."""
    if "cavity" in file_path.lower():
        return "uniform"
    else:
        return "polar"


def spod_function(qhat, nblocks, dst, w, return_psi=False):
    """
    Compute SPOD modes and eigenvalues for a single frequency.
    Args:
        qhat (np.ndarray): FFT coefficients for this frequency [space, block].
        nblocks (int): Number of blocks.
        dst (float): Frequency resolution (delta f).
        w (np.ndarray): Spatial integration weights [space, 1].
        return_psi (bool): If True, also return psi (time coefficients).
    Returns:
        tuple: (phi, lambda_tilde[, psi])
            phi (np.ndarray): Spatial SPOD modes for this frequency [space, mode].
            lambda_tilde (np.ndarray): SPOD eigenvalues (energy) for this frequency [mode].
            psi (np.ndarray, optional): Time coefficients for this frequency [block, mode].
    """
    # Normalize FFT coefficients to get fluctuation matrix X_f for this frequency f.
    x = qhat / np.sqrt(nblocks * dst)
    # Compute the weighted cross-spectral density (CSD) matrix M_f.
    xprime_w = np.transpose(np.conj(x)) * np.transpose(w)  # X_f^H * W
    m = xprime_w @ x  # (X_f^H * W) * X_f = M_f
    del xprime_w
    # Solve the eigenvalue problem: M_f * Psi_f = Psi_f * Lambda_f
    lambda_tilde, psi = np.linalg.eigh(m)
    # Sort eigenvalues and eigenvectors in descending order
    idx = lambda_tilde.argsort()[::-1]
    lambda_tilde = lambda_tilde[idx]
    psi = psi[:, idx]
    # Compute spatial SPOD modes (Phi_f) of the direct problem.
    inv_sqrt_lambda = np.zeros_like(lambda_tilde)
    mask = lambda_tilde > 1e-12
    inv_sqrt_lambda[mask] = 1.0 / np.sqrt(lambda_tilde[mask])
    phi = x @ psi @ np.diag(inv_sqrt_lambda)
    if return_psi:
        return phi, np.abs(lambda_tilde), psi
    return phi, np.abs(lambda_tilde)


class BaseAnalyzer:
    """Base class for modal decomposition analyzers."""

    def __init__(self, file_path, nfft=128, overlap=0.5, results_dir="./preprocess", figures_dir="./figs", data_loader=None, spatial_weight_type="auto"):
        """Initialize the analyzer.

        Args:
            file_path (str): Path to data file.
            nfft (int): Number of snapshots per FFT block.
            overlap (float): Overlap fraction between blocks.
            results_dir (str): Directory to save results.
            figures_dir (str): Directory to save figures.
            data_loader (callable): Function to load data.
            spatial_weight_type (str): Type of spatial weighting.
        """
        self.file_path = file_path
        self.nfft = nfft
        self.overlap = overlap
        self.results_dir = results_dir
        self.figures_dir = figures_dir

        # Set default data loader based on file type
        self.data_loader = data_loader or load_data

        # Set default weight type
        if spatial_weight_type == "auto":
            self.spatial_weight_type = auto_detect_weight_type(file_path)
        else:
            self.spatial_weight_type = spatial_weight_type

        # Calculated later
        self.novlap = int(overlap * nfft)
        self.data = {}
        self.W = np.array([])
        self.nblocks = 0
        self.fs = 0.0
        self.qhat = np.array([])

        # Extract root name for output files
        base = os.path.basename(file_path)
        self.data_root = os.path.splitext(base)[0]

        # Ensure output directories exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def load_and_preprocess(self):
        """Load data and calculate weights."""
        # Load data from file
        self.data = self.data_loader(self.file_path)

        # Calculate spatial weights
        if self.spatial_weight_type == "polar":
            self.W = calculate_polar_weights(self.data["x"], self.data["y"])
            print("Using polar (cylindrical) spatial weights.")
        else:
            self.W = calculate_uniform_weights(self.data["x"], self.data["y"])
            print("Using uniform spatial weights (rectangular grid).")

        # Calculate derived parameters
        self.nblocks = int(np.ceil((self.data["Ns"] - self.novlap) / (self.nfft - self.novlap)))
        self.fs = 1 / self.data["dt"]

        print(f"Data loaded: {self.data['Ns']} snapshots, {self.data['Nx']}×{self.data['Ny']} spatial points")
        print(f"FFT parameters: {self.nfft} points, {self.overlap * 100}% overlap, {self.nblocks} blocks")

    def compute_fft_blocks(self):
        """Compute blocked FFT using Welch's method."""
        if "q" not in self.data:
            raise ValueError("Data not loaded. Call load_and_preprocess() first.")

        print(f"Computing FFT with {self.nblocks} blocks...")
        self.qhat = blocksfft(self.data["q"], self.nfft, self.nblocks, self.novlap, blockwise_mean=getattr(self, "blockwise_mean", False), normvar=getattr(self, "normvar", False), window_norm=getattr(self, "window_norm", "power"), window_type=getattr(self, "window_type", "hamming"))
        print("FFT computation complete.")

    def save_results(self, filename=None, analysis_type="spod"):
        """Save results to HDF5 file with harmonized filename and format.
        Args:
            filename (str, optional): Custom filename. If not provided, uses harmonized scheme.
            analysis_type (str): Analysis type for filename (e.g., 'spod', 'bsmd').
        """
        if not filename:
            filename = make_result_filename(self.data_root, self.nfft, self.overlap, self.data.get("Ns", 0), analysis_type)
        save_path = os.path.join(self.results_dir, filename)
        print(f"Saving results to {save_path}")
        # This is a placeholder - subclasses should implement specific saving logic
        with h5py.File(save_path, "w") as f:
            f.attrs["nfft"] = self.nfft
            f.attrs["overlap"] = self.overlap
            f.attrs["nblocks"] = self.nblocks
            f.attrs["fs"] = self.fs
            # Save coordinates
            f.create_dataset("x", data=self.data["x"], compression="gzip")
            f.create_dataset("y", data=self.data["y"], compression="gzip")
            # Save weights
            f.create_dataset("W", data=self.W, compression="gzip")

    def run(self, compute_fft=True):
        """Run the full analysis pipeline."""
        start_time = time.time()

        # Load data and calculate weights
        self.load_and_preprocess()

        # Compute FFT blocks if requested
        if compute_fft:
            self.compute_fft_blocks()

        end_time = time.time()
        print(f"Completed in {end_time - start_time:.2f} seconds.")

        return self
