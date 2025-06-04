#!/usr/bin/env python3
"""
Modular Data Interface for Modal Decomposition

This module provides a unified interface for loading different data formats
(.mat, .h5, .cgns, etc.) with consistent output structure for all analysis methods.

The goal is to have a single place to add new data format support, and all
analysis methods (POD, SPOD, BSMD) will automatically work with the new format.
"""

import numpy as np
import h5py
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class DataLoader(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def load(self, file_path: str) -> Dict[str, Any]:
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
        return file_path.lower().endswith('.mat')
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load data from .mat file with flexible variable detection."""
        print(f"📂 Loading .mat data from {file_path}")
        
        with h5py.File(file_path, "r") as fread:
            # Try common variable names for the main field
            q = None
            for var in ["p", "u", "v", "data", "field"]:
                if var in fread:
                    q = fread[var][:]
                    print(f"   Found data variable: '{var}'")
                    break
            
            if q is None:
                # List available variables
                available_vars = list(fread.keys())
                raise KeyError(f"No recognized data variable in file. Available: {available_vars}")
            
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
        
        Nx, Ny = len(x_vec), len(y_vec)
        Nz = len(z_vec) if z_vec is not None else 1
        
        # Reshape data to standard format: (Ns, Nspace)
        q_reshaped = self._reshape_to_standard_format(q, Nx, Ny, Nz)
        Ns = q_reshaped.shape[0]
        
        print(f"   Processed shape: q={q_reshaped.shape}, Nx={Nx}, Ny={Ny}, Nz={Nz}, Ns={Ns}")
        
        return {
            'q': q_reshaped,
            'x': x_vec,
            'y': y_vec, 
            'z': z_vec,
            'dt': dt,
            'Nx': Nx,
            'Ny': Ny,
            'Nz': Nz,
            'Ns': Ns,
            'metadata': {
                'format': 'mat',
                'original_shape': q.shape,
                'file_path': file_path
            }
        }
    
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
        return file_path.lower().endswith(('.h5', '.hdf5')) or 'jet' in file_path.lower()
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load data from HDF5 file with JetLES format."""
        print(f"📂 Loading HDF5/JetLES data from {file_path}")
        
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
        
        return {
            'q': q_reshaped,
            'x': x,
            'y': y,
            'z': None,
            'dt': dt,
            'Nx': Nx,
            'Ny': Ny, 
            'Nz': 1,
            'Ns': Ns,
            'metadata': {
                'format': 'hdf5_jetles',
                'original_shape': (Nx, Ny, Ns),
                'file_path': file_path
            }
        }


class CGNSDataLoader(DataLoader):
    """Loader for CGNS files - placeholder for future implementation."""
    
    def supports_format(self, file_path: str) -> bool:
        """Check if file is a CGNS file."""
        return file_path.lower().endswith('.cgns')
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load data from CGNS file."""
        print(f"📂 Loading CGNS data from {file_path}")
        
        # TODO: Implement CGNS loading using python-cgns or similar
        # This is where you'll add CGNS support later
        
        # Placeholder implementation
        raise NotImplementedError(
            "CGNS support not yet implemented. "
            "To add CGNS support, implement the loading logic here and "
            "ensure output follows the standard format."
        )
        
        # Template for what the implementation should return:
        return {
            'q': None,  # np.ndarray (Ns, Nspace)
            'x': None,  # np.ndarray (Nx,)
            'y': None,  # np.ndarray (Ny,)
            'z': None,  # np.ndarray (Nz,) or None
            'dt': None,  # float
            'Nx': None,  # int
            'Ny': None,  # int
            'Nz': None,  # int
            'Ns': None,  # int
            'metadata': {
                'format': 'cgns',
                'file_path': file_path,
                # Add CGNS-specific metadata here
            }
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
            # Add new loaders here in the future
        ]
    
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
            loader_map = {
                'mat': MATDataLoader,
                'hdf5': HDF5DataLoader,
                'cgns': CGNSDataLoader
            }
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
        supported_formats = ['.mat', '.h5', '.hdf5', '.cgns']
        raise ValueError(
            f"No loader found for file extension '{file_ext}'. "
            f"Supported formats: {supported_formats}"
        )
    
    def get_weight_type(self, data: Dict[str, Any], file_path: str) -> str:
        """
        Determine appropriate spatial weight type based on data characteristics.
        
        Parameters:
        -----------
        data : dict
            Loaded data dictionary
        file_path : str
            Original file path for heuristics
            
        Returns:
        --------
        str
            Weight type ('uniform', 'polar', 'custom')
        """
        # Use existing heuristics but make them more robust
        if "cavity" in file_path.lower():
            return "uniform"
        elif "jet" in file_path.lower() or data['metadata'].get('format') == 'hdf5_jetles':
            return "polar"
        else:
            # Could add more sophisticated detection based on coordinate structure
            # For now, default to uniform for unknown formats
            return "uniform"
    
    def list_supported_formats(self) -> Dict[str, str]:
        """Return dictionary of supported formats and their descriptions."""
        return {
            '.mat': 'MATLAB data files',
            '.h5/.hdf5': 'HDF5 format (including JetLES)',
            '.cgns': 'CGNS format (future implementation)',
        }


# Global instance for easy access
data_manager = DataInterfaceManager()


def load_data(file_path: str, loader_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for loading data.
    
    This is the main entry point that all analysis methods should use.
    """
    return data_manager.load_data(file_path, loader_type)


def get_weight_type(data: Dict[str, Any], file_path: str) -> str:
    """Convenience function for determining weight type."""
    return data_manager.get_weight_type(data, file_path)


# Legacy compatibility functions - these maintain backward compatibility
# while redirecting to the new modular system

def load_jetles_data(file_path: str) -> Dict[str, Any]:
    """Legacy compatibility function for JetLES data."""
    return load_data(file_path, loader_type='hdf5')


def load_mat_data(file_path: str) -> Dict[str, Any]:
    """Legacy compatibility function for .mat data."""
    return load_data(file_path, loader_type='mat')


def auto_detect_weight_type(file_path: str) -> str:
    """Legacy compatibility function for weight detection."""
    # For backward compatibility, we need to load the data first
    data = load_data(file_path)
    return get_weight_type(data, file_path)

