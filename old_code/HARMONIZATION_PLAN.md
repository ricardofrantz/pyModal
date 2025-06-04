# Modal Decomposition Harmonization Plan

## Overview

This document outlines the harmonization strategy for POD, SPOD, and BSMD implementations to create a modular, consistent codebase that can easily adapt to new data formats (like CGNS) with minimal changes.

## Current State Analysis

### Code Structure Issues
1. **Inconsistent imports and dependencies** across files
2. **Mixed documentation styles** and comment formatting  
3. **Different error handling** and progress reporting approaches
4. **Hardcoded data loading** specific to .mat/.h5 formats
5. **Duplicated utility functions** across files
6. **Inconsistent variable naming** and code organization

### Data Format Dependencies
- **POD**: Uses `BaseAnalyzer` → `load_jetles_data`/`load_mat_data`
- **SPOD**: Uses `BaseAnalyzer` → same data loaders
- **BSMD**: Uses `BaseAnalyzer` → same data loaders

**Problem**: Adding CGNS support requires changes in multiple places.

## Harmonization Strategy

### 1. Modular Data Interface Layer ✅

**Created**: `data_interface.py` with:
- **Abstract `DataLoader` class** for extensible format support
- **Concrete loaders**: `MATDataLoader`, `HDF5DataLoader`, `CGNSDataLoader` (placeholder)
- **Unified output format** for all analysis methods
- **Automatic format detection** and manual override options
- **Legacy compatibility** functions

**Benefits**:
- ✅ Add CGNS support by implementing one class
- ✅ All analysis methods automatically support new formats
- ✅ Consistent data structure across all methods
- ✅ Backward compatibility maintained

### 2. Harmonized Base Classes ✅

**Created**: `harmonized_utils.py` with:
- **`HarmonizedAnalyzer`** abstract base class
- **Consistent initialization** and configuration
- **Standardized progress reporting** with emojis and clear formatting
- **Common utility functions** (weights, plotting, file naming)
- **Unified analysis pipeline** (`run_analysis()` method)

### 3. Implementation Harmonization Plan

#### A. Theory and Algorithm Verification

**POD (Proper Orthogonal Decomposition)**:
- ✅ **Method**: Snapshot POD via eigenvalue decomposition
- ✅ **Theory**: Energy-optimal spatial modes via SVD/eigen decomposition
- ✅ **Implementation**: Standard covariance matrix approach
- 🔄 **Needs**: Enhanced documentation, consistent variable naming

**SPOD (Spectral POD)**:
- ✅ **Method**: Cross-spectral density matrix eigenvalue problem
- ✅ **Theory**: Towne, Schmidt & Colonius (2018) reference implementation
- ✅ **Implementation**: Welch's method + frequency-domain decomposition
- 🔄 **Needs**: Harmonized with new base class, improved error handling

**BSMD (Bispectral Mode Decomposition)**:
- ✅ **Method**: Third-order spectral analysis for triadic interactions
- ✅ **Theory**: Schmidt (2020) bispectral correlation approach
- ✅ **Implementation**: FFT blocks + bispectral correlation matrix
- 🔄 **Needs**: Enhanced triad selection, better progress reporting

#### B. Code Structure Harmonization

**Common Elements**:
```python
class PODAnalyzer(HarmonizedAnalyzer):
    def __init__(self, file_path, **kwargs):
        super().__init__(file_path, **kwargs)
        # Method-specific initialization
    
    def compute_decomposition(self):
        # Core POD algorithm
        
    def save_results(self, filename=None):
        # Standardized HDF5 saving
        
    def create_visualizations(self):
        # Method-specific plots
        
    def _print_analysis_summary(self):
        # Method-specific summary
```

**Harmonization Changes**:

1. **Consistent Imports**:
```python
#!/usr/bin/env python3
"""
[Method] Analysis for Modal Decomposition

This module implements [METHOD] for extracting coherent structures
from spatio-temporal data.

Theory: [Reference and brief description]
Author: Modal Decomposition Team
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from configs import *
from harmonized_utils import HarmonizedAnalyzer, setup_matplotlib_style
```

2. **Standardized Documentation**:
```python
def method_name(parameter: type, parameter2: type = default) -> return_type:
    """
    Brief method description.
    
    Detailed explanation of the method, theory, and implementation.
    
    Parameters:
    -----------
    parameter : type
        Description of parameter
    parameter2 : type, optional
        Description with default value
        
    Returns:
    --------
    return_type
        Description of return value
        
    References:
    -----------
    Author (Year). "Title." Journal.
    """
```

3. **Harmonized Progress Reporting**:
```python
print(f"🔬 Computing [METHOD] decomposition...")
print(f"   📐 Processing {n_modes} modes")
print(f"   ⏱️  Estimated time: {est_time:.1f}s")
# ... computation ...
print(f"   ✅ [METHOD] completed in {actual_time:.2f}s")
```

4. **Consistent Variable Naming**:
```python
# Spatial coordinates
x_coords, y_coords, z_coords = data['x'], data['y'], data['z']

# Data arrays  
data_matrix = data['q']  # Shape: (Ns, Nspace)
spatial_weights = self.W  # Shape: (Nspace, 1)

# Results
spatial_modes = phi  # Shape: (Nspace, Nmodes)
eigenvalues = lambda_values  # Shape: (Nmodes,)
time_coefficients = temporal_coeffs  # Shape: (Ns, Nmodes)
```

### 4. CGNS Integration Strategy

**When adding CGNS support**:

1. **Implement `CGNSDataLoader`** in `data_interface.py`:
```python
class CGNSDataLoader(DataLoader):
    def load(self, file_path: str) -> Dict[str, Any]:
        # Use python-cgns or similar library
        # Return standardized format
        return {
            'q': data_reshaped,  # (Ns, Nspace)
            'x': x_coords, 'y': y_coords, 'z': z_coords,
            'dt': time_step,
            'Nx': nx, 'Ny': ny, 'Nz': nz, 'Ns': ns,
            'metadata': {'format': 'cgns', ...}
        }
```

2. **All analysis methods automatically work** - no changes needed!

3. **Test with**:
```python
# Automatic detection
analyzer = PODAnalyzer('data.cgns')
results = analyzer.run_analysis()

# Manual specification  
analyzer = SPODAnalyzer('data.cgns', loader_type='cgns')
```

### 5. Enhanced Features

#### A. Improved Error Handling
```python
try:
    self.compute_decomposition()
except MemoryError:
    print("❌ Insufficient memory. Try reducing data size or number of modes.")
    raise
except np.linalg.LinAlgError:
    print("❌ Numerical instability detected. Check data conditioning.")
    raise
```

#### B. Progress Tracking
```python
from tqdm import tqdm

for freq_idx in tqdm(range(n_frequencies), desc="🔬 Computing SPOD modes"):
    # Computation per frequency
```

#### C. Metadata and Provenance
```python
# Enhanced result files with full provenance
metadata = {
    'method': 'SPOD',
    'version': '2.0',
    'parameters': self.get_parameters(),
    'data_source': self.file_path,
    'computation_time': self.computation_time,
    'timestamp': datetime.now().isoformat(),
    'git_commit': get_git_commit(),  # For reproducibility
}
```

### 6. Implementation Timeline

#### Phase 1: Core Harmonization ✅
- ✅ Create modular data interface (`data_interface.py`)
- ✅ Create harmonized base classes (`harmonized_utils.py`)
- ✅ Plan refactoring strategy

#### Phase 2: Method Harmonization 🔄
- 🔄 Refactor POD to use `HarmonizedAnalyzer`
- 🔄 Refactor SPOD to use new interface
- 🔄 Refactor BSMD to use new interface
- 🔄 Update documentation and comments

#### Phase 3: Enhanced Features
- 🔲 Add progress tracking with `tqdm`
- 🔲 Implement advanced error handling
- 🔲 Add metadata and provenance tracking
- 🔲 Create comprehensive test suite

#### Phase 4: CGNS Integration
- 🔲 Implement `CGNSDataLoader` class
- 🔲 Test with real CGNS data
- 🔲 Validate all methods work with CGNS
- 🔲 Update documentation

### 7. Testing Strategy

#### Unit Tests
```python
def test_data_interface():
    # Test each loader with sample data
    
def test_pod_harmonized():
    # Test POD with new interface
    
def test_cgns_compatibility():
    # Test CGNS data loading and analysis
```

#### Integration Tests
```python
def test_all_methods_with_format(format_type):
    # Test POD, SPOD, BSMD with same dataset
    # Verify consistent results
```

### 8. Benefits of Harmonization

#### For Users
- ✅ **Consistent API** across all methods
- ✅ **Automatic format support** - just change file extension
- ✅ **Better error messages** and progress tracking
- ✅ **Standardized outputs** and documentation

#### For Developers  
- ✅ **Single place** to add new data formats
- ✅ **Consistent code style** and documentation
- ✅ **Shared utilities** reduce duplication
- ✅ **Easier testing** and maintenance

#### For Research
- ✅ **Reproducible results** with metadata tracking
- ✅ **Easy method comparison** with same interface
- ✅ **Extensible framework** for new decomposition methods
- ✅ **Publication-ready outputs** with consistent formatting

## Next Steps

1. **Complete Phase 2**: Refactor existing methods to use harmonized structure
2. **Implement CGNS support**: Add `CGNSDataLoader` when ready
3. **Create test suite**: Ensure robustness across formats and methods
4. **Documentation update**: Create unified user guide

The modular design means **adding CGNS support is now a single-class implementation** that automatically enables all analysis methods! 🎯