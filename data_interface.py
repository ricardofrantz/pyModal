#!/usr/bin/env python3
"""
Modular Data Interface for Modal Decomposition

This module provides a unified interface for loading different data formats
(.mat, .h5, .cgns, etc.) with consistent output structure for all analysis methods.

The goal is to have a single place to add new data format support, and all
analysis methods (POD, SPOD, BSMD) will automatically work with the new format.
"""

import glob
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import h5py
import numpy as np


def natural_sort_key(string: str):
    """Return a key for natural sorting of strings with numbers."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", string)]


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Load data from file and return standardized format.

        Returns:
        --------
        dict with keys:
            'q': np.ndarray, shape (Ns, Nspace) - data reshaped for analysis
            'x': np.ndarray, shape (Nx,) - x-coordinates
            'y': np.ndarray, shape (Ny,) - y-coordinates
            'z': np.ndarray, shape (Nz,) - z-coordinates (optional for 2D)
            'dt': float - time step
            'Nx': int - number of x points
            'Ny': int - number of y points
            'Nz': int - number of z points (optional)
            'Ns': int - number of snapshots
            'metadata': dict - additional format-specific information
        Additional keyword arguments may be used by specific loaders.
        """
        pass

    @abstractmethod
    def supports_format(self, file_path: str) -> bool:
        """Check if this loader supports the given file format."""
        pass


class MATDataLoader(DataLoader):
    """Loader for MATLAB .mat files."""

    def supports_format(self, file_path: str) -> bool:
        """Check if file is a .mat file."""
        return file_path.lower().endswith(".mat")

    def load(self, file_path: str, preview_ns: int = None, **kwargs) -> Dict[str, Any]:
        """Load data from .mat file with flexible variable detection. Optionally, only load first preview_ns snapshots."""
        file_size = format_file_size(file_path)
        print(f"📂 Loading .mat data from {file_path} ({file_size})")

        with h5py.File(file_path, "r") as fread:
            # Try common variable names for the main field
            q = None
            var_name = None
            for var in ["p", "u", "v", "data", "field"]:
                if var in fread:
                    var_name = var
                    if preview_ns is not None:
                        # Only load first preview_ns along each axis (for preview mode)
                        shape = fread[var].shape
                        time_axis = int(np.argmax(shape))
                        index = [slice(None)] * len(shape)
                        index[time_axis] = slice(0, preview_ns)
                        q = fread[var][tuple(index)]
                    else:
                        q = fread[var][:]
                    print(f"   Found data variable: '{var}'")
                    break

            if q is None:
                # List available variables
                available_vars = list(fread.keys())
                raise KeyError(f"No recognized data variable in file. Available: {available_vars}")

            # Optionally cast to float32 to save memory
            if q.dtype == np.float64:
                q = q.astype(np.float32, copy=False)

            # Load coordinates
            x = fread["x"][:] if "x" in fread else np.arange(q.shape[1])
            y = fread["y"][:] if "y" in fread else np.arange(q.shape[2] if q.ndim > 2 else 1)
            z = fread["z"][:] if "z" in fread else None

            # Load time step
            if "dt" in fread:
                dt_data = np.array(fread["dt"])
                dt = dt_data[0][0] if dt_data.ndim > 1 else float(dt_data)
            else:
                dt = 1.0
                print("   Warning: No 'dt' found, using dt=1.0")

        # Process coordinates to 1D vectors
        x_vec = x[:, 0] if x.ndim == 2 else x
        y_vec = y[0, :] if y.ndim == 2 else y
        z_vec = z[0, 0, :] if (z is not None and z.ndim == 3) else z

        # --- Simplified: largest axis is time, others are space ---
        q_shape = q.shape
        time_axis = np.argmax(q_shape)
        Ns = q_shape[time_axis]
        if time_axis != 0:
            q = np.moveaxis(q, time_axis, 0)
        spatial_shapes = q.shape[1:]
        coords = [x_vec, y_vec]
        if z_vec is not None:
            coords.append(z_vec)
        # Assign coordinate vectors by order, fallback to np.arange if length mismatches
        for i, s in enumerate(spatial_shapes):
            if i < len(coords):
                if len(coords[i]) != s:
                    coords[i] = np.arange(s)
            else:
                coords.append(np.arange(s))
        x_vec = coords[0] if len(coords) > 0 else np.arange(q.shape[1] if len(q.shape) > 1 else 1)
        y_vec = coords[1] if len(coords) > 1 else np.arange(q.shape[2] if len(q.shape) > 2 else 1)
        z_vec = coords[2] if len(coords) > 2 else None
        Nx = len(x_vec)
        Ny = len(y_vec)
        Nz = len(z_vec) if z_vec is not None else 1
        # Only reshape if needed
        if q.ndim == 2:
            if q.shape == (Ns, Nx * Ny * Nz):
                q_reshaped = q
            elif q.shape == (Nx * Ny * Nz, Ns):
                q_reshaped = q.T
            else:
                q_reshaped = q.reshape(Ns, Nx * Ny * Nz)
        else:
            q_reshaped = q.reshape(Ns, Nx * Ny * Nz)
        Ns = q_reshaped.shape[0]
        print(f"   Processed shape: q={q_reshaped.shape}, Nx={Nx}, Ny={Ny}, Nz={Nz}, Ns={Ns}")
        return {"q": q_reshaped, "x": x_vec, "y": y_vec, "z": z_vec, "dt": dt, "Nx": Nx, "Ny": Ny, "Nz": Nz, "Ns": Ns, "metadata": {"format": "mat", "original_shape": q.shape, "file_path": file_path, "var_name": var_name}}

    def _reshape_to_standard_format(self, q: np.ndarray, Nx: int, Ny: int, Nz: int) -> np.ndarray:
        """Reshape data array to standard (Ns, Nspace) format."""
        if q.ndim == 2:
            # Already 2D - check if it's (Ns, Nspace) or (Nspace, Ns)
            if q.shape[1] == Nx * Ny * Nz:
                return q  # Already correct format
            elif q.shape[0] == Nx * Ny * Nz:
                return q.T  # Transpose to correct format
            else:
                raise ValueError(f"Cannot interpret 2D data shape {q.shape} with spatial dimensions {Nx}×{Ny}×{Nz}")

        elif q.ndim == 3:
            # Try different permutations to find (Ns, Nx, Ny)
            for axes in [(0, 1, 2), (2, 0, 1), (2, 1, 0), (0, 2, 1), (1, 0, 2), (1, 2, 0)]:
                try:
                    arr = np.transpose(q, axes)
                    if arr.shape[1:] == (Nx, Ny):
                        return arr.reshape(arr.shape[0], Nx * Ny)
                except Exception:
                    continue

            # Special case: (Nx, Ny, Ns) - common in some formats
            if q.shape[:2] == (Nx, Ny):
                return np.transpose(q, (2, 0, 1)).reshape(q.shape[2], Nx * Ny)

            raise ValueError(f"Cannot interpret 3D data shape {q.shape} with spatial dimensions {Nx}×{Ny}")

        elif q.ndim == 4:
            # 4D data (Ns, Nx, Ny, Nz) or similar
            for axes in [(0, 1, 2, 3), (3, 0, 1, 2), (3, 2, 1, 0)]:
                try:
                    arr = np.transpose(q, axes)
                    if arr.shape[1:] == (Nx, Ny, Nz):
                        return arr.reshape(arr.shape[0], Nx * Ny * Nz)
                except Exception:
                    continue

            raise ValueError(f"Cannot interpret 4D data shape {q.shape} with spatial dimensions {Nx}×{Ny}×{Nz}")

        else:
            raise ValueError(f"Unsupported data dimensionality: {q.ndim}D")


class HDF5DataLoader(DataLoader):
    """Loader for HDF5/JetLES format files."""

    def supports_format(self, file_path: str) -> bool:
        """Check if file is an HDF5 file."""
        return file_path.lower().endswith((".h5", ".hdf5")) or "jet" in file_path.lower()

    def load(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Load data from HDF5 file with JetLES format."""
        file_size = format_file_size(file_path)
        print(f"📂 Loading HDF5/JetLES data from {file_path} ({file_size})")

        with h5py.File(file_path, "r") as fread:
            # JetLES format specifics
            q = fread["p"][:]  # pressure field
            x = fread["x"][:, 0]  # x-coordinates (axial)
            y = fread["r"][0, :]  # y-coordinates (radial)
            dt = fread["dt"][0][0]  # time step

        # Transpose q: original (Nx, Ny, Ns) -> (Ns, Nx, Ny)
        q = np.transpose(q, (2, 0, 1))
        Nx, Ny = len(x), len(y)
        Ns = q.shape[0]
        q_reshaped = q.reshape(Ns, Nx * Ny)

        print(f"   Processed shape: q={q_reshaped.shape}, Nx={Nx}, Ny={Ny}, Ns={Ns}")

        return {"q": q_reshaped, "x": x, "y": y, "z": None, "dt": dt, "Nx": Nx, "Ny": Ny, "Nz": 1, "Ns": Ns, "metadata": {"format": "hdf5_jetles", "original_shape": (Nx, Ny, Ns), "file_path": file_path}}


class CGNSDataLoader(DataLoader):
    """Loader for CGNS files - placeholder for future implementation."""

    def supports_format(self, file_path: str) -> bool:
        """Check if file is a CGNS file."""
        return file_path.lower().endswith(".cgns")

    def load(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Load data from CGNS file."""
        file_size = format_file_size(file_path)
        print(f"📂 Loading CGNS data from {file_path} ({file_size})")

        # TODO: Implement CGNS loading using python-cgns or similar
        # This is where you'll add CGNS support later

        # Placeholder implementation
        raise NotImplementedError("CGNS support not yet implemented. To add CGNS support, implement the loading logic here and ensure output follows the standard format.")

        # Template for what the implementation should return:
        return {
            "q": None,  # np.ndarray (Ns, Nspace)
            "x": None,  # np.ndarray (Nx,)
            "y": None,  # np.ndarray (Ny,)
            "z": None,  # np.ndarray (Nz,) or None
            "dt": None,  # float
            "Nx": None,  # int
            "Ny": None,  # int
            "Nz": None,  # int
            "Ns": None,  # int
            "metadata": {
                "format": "cgns",
                "file_path": file_path,
                # Add CGNS-specific metadata here
            },
        }


class DataInterfaceManager:
    """
    Manager class that automatically selects the appropriate data loader
    based on file format and provides a unified interface for all analysis methods.
    """

    def __init__(self):
        """Initialize with all available data loaders."""
        self.loaders = [
            MATDataLoader(),
            HDF5DataLoader(),
            CGNSDataLoader(),
            DNamiXNPZLoader(),
            # Add new loaders here in the future
        ]

    def load_data(self, file_path: str, loader_type: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Load data using automatic format detection or specified loader.
        Parameters:
        -----------
        file_path : str
            Path to the data file
        loader_type : str, optional
            Force specific loader ('mat', 'hdf5', 'cgns')
        Returns:
        --------
        dict
            Standardized data format for all analysis methods
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        # If specific loader requested
        if loader_type:
            loader_map = {
                "mat": MATDataLoader,
                "hdf5": HDF5DataLoader,
                "cgns": CGNSDataLoader,
                "dnamiX_npz": DNamiXNPZLoader,
            }
            if loader_type not in loader_map:
                raise ValueError(f"Unknown loader type: {loader_type}")
            loader = loader_map[loader_type]()
            return loader.load(file_path, **kwargs)
        # Auto-detect format
        for loader in self.loaders:
            if loader.supports_format(file_path):
                return loader.load(file_path, **kwargs)
        # No suitable loader found
        file_ext = os.path.splitext(file_path)[1].lower()
        supported_formats = [".mat", ".h5", ".hdf5", ".cgns", ".npz"]
        raise ValueError(f"No loader found for file extension '{file_ext}'. Supported formats: {supported_formats}")


class DNamiXNPZLoader(DataLoader):
    """Loader for dNamiX consolidated .npz files (output of interp_cart.py consolidation)."""

    def supports_format(self, file_path: str) -> bool:
        return file_path.lower().endswith(".npz")

    def get_available_fields(self, file_path: str):
        npz = np.load(file_path)
        return [k for k in ("u", "v", "p") if k in npz]

    def load(self, file_path: str, field: str = None, load_single: bool = False, **kwargs) -> Dict[str, Any]:
        """Load one or multiple dNamiX ``.npz`` files."""
        directory = os.path.dirname(file_path)
        if load_single:
            files = [file_path]
        else:
            pattern = os.path.join(directory, "*.npz")
            files = sorted(glob.glob(pattern), key=natural_sort_key)

        print("📂 Loading npz files in order:")
        for i, f in enumerate(files, 1):
            print(f"   {i}. {os.path.basename(f)} ({format_file_size(f)})")

        q_list = []
        times_list = []
        x = y = None
        dt = None
        available_fields = None
        Nx = Ny = None

        from utils import get_num_threads, parallel_map

        npz_list = parallel_map(np.load, files, threads=get_num_threads())

        for npz, f in zip(npz_list, files):
            if x is None:
                x = npz["x"]
                y = npz["y"]
                if x.ndim == 2:
                    x = x[:, 0]
                if y.ndim == 2:
                    y = y[0, :]
            if dt is None and "dt" in npz:
                dt_val = npz["dt"]
                dt = float(np.mean(dt_val)) if dt_val.size > 0 else None
            if available_fields is None:
                available_fields = [k for k in ("u", "v", "p") if k in npz]
            if field is None:
                for candidate in ["u", "v", "p"]:
                    if candidate in npz:
                        field = candidate
                        break
            if field not in npz:
                raise KeyError(f"Requested field '{field}' not found in {f}")
            arr = npz[field]
            times = npz["times"]
            if Nx is None:
                Nx, Ny = arr.shape[1], arr.shape[2]
            q_list.append(arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2]))
            times_list.append(times)

        q = np.concatenate(q_list, axis=0)
        times = np.concatenate(times_list)
        Ns = times.shape[0]
        if dt is None:
            if len(times) > 1:
                diffs = np.diff(times)
                diffs = diffs[np.nonzero(diffs)]
                dt = float(np.mean(diffs)) if diffs.size > 0 else 1.0
            else:
                dt = 1.0
        print(f"   Processed shape: q={q.shape}, Nx={Nx}, Ny={Ny}, Ns={Ns}, dt={dt}, field={field}")
        return {
            "q": q,
            "x": x,
            "y": y,
            "z": None,
            "dt": dt,
            "Nx": Nx,
            "Ny": Ny,
            "Nz": 1,
            "Ns": Ns,
            "metadata": {
                "format": "dnamiX_npz",
                "file_path": file_path,
                "var_name": field,
                "available_fields": available_fields,
                "loaded_files": files,
            },
        }

    def load_data(self, file_path: str, loader_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Load data using automatic format detection or specified loader.

        Parameters:
        -----------
        file_path : str
            Path to the data file
        loader_type : str, optional
            Force specific loader ('mat', 'hdf5', 'cgns')

        Returns:
        --------
        dict
            Standardized data format for all analysis methods
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # If specific loader requested
        if loader_type:
            loader_map = {"mat": MATDataLoader, "hdf5": HDF5DataLoader, "cgns": CGNSDataLoader}
            if loader_type not in loader_map:
                raise ValueError(f"Unknown loader type: {loader_type}")

            loader = loader_map[loader_type]()
            return loader.load(file_path)

        # Auto-detect format
        for loader in self.loaders:
            if loader.supports_format(file_path):
                return loader.load(file_path)

        # No suitable loader found
        file_ext = os.path.splitext(file_path)[1].lower()
        supported_formats = [".mat", ".h5", ".hdf5", ".cgns"]
        raise ValueError(f"No loader found for file extension '{file_ext}'. Supported formats: {supported_formats}")

    def get_weight_type(self, data: Dict[str, Any], file_path: str) -> str:
        """
        Always return 'uniform' for dNamiX consolidated .npz files (Cartesian mesh), else use legacy logic.
        """
        if file_path.lower().endswith(".npz") or data.get("metadata", {}).get("format") == "dnamiX_npz":
            return "uniform"
        # --- legacy/other logic ---
        if "cavity" in file_path.lower():
            return "uniform"
        elif "jet" in file_path.lower() or data["metadata"].get("format") == "hdf5_jetles":
            return "polar"
        else:
            return "uniform"

    def list_supported_formats(self) -> Dict[str, str]:
        """Return dictionary of supported formats and their descriptions."""
        return {
            ".mat": "MATLAB data files",
            ".h5/.hdf5": "HDF5 format (including JetLES)",
            ".cgns": "CGNS format (future implementation)",
        }


def format_file_size(file_path: str) -> str:
    """Format file size in human-readable format (MB or GB)."""
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    if size_mb >= 1024:
        size_gb = size_mb / 1024
        return f"{size_gb:.1f} GB"
    else:
        return f"{size_mb:.1f} MB"


# Global instance for easy access
data_manager = DataInterfaceManager()


def load_data(file_path: str, loader_type: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Convenience function for loading data.

    This is the main entry point that all analysis methods should use.
    """
    return data_manager.load_data(file_path, loader_type, **kwargs)


def get_weight_type(data: Dict[str, Any], file_path: str) -> str:
    """Convenience function for determining weight type."""
    return data_manager.get_weight_type(data, file_path)


# Legacy compatibility functions - these maintain backward compatibility
# while redirecting to the new modular system


def load_jetles_data(file_path: str, **kwargs) -> Dict[str, Any]:
    """Legacy compatibility function for JetLES data."""
    return load_data(file_path, loader_type="hdf5", **kwargs)


def load_mat_data(file_path: str, **kwargs) -> Dict[str, Any]:
    """Legacy compatibility function for .mat data."""
    return load_data(file_path, loader_type="mat", **kwargs)


def auto_detect_weight_type(file_path: str) -> str:
    """Legacy compatibility function for weight detection."""
    # For backward compatibility, we need to load the data first
    data = load_data(file_path)
    return get_weight_type(data, file_path)
