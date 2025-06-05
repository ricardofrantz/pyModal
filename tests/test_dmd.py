import numpy as np

from dmd import DMDAnalyzer


def test_perform_dmd_simple():
    data = {
        "q": np.array([[1, 2], [2, 4], [4, 8]], dtype=float),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 3,
    }
    analyzer = DMDAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_dmd()
    assert analyzer.modes.shape == (2, 2)
    assert analyzer.time_coefficients.shape == (3, 2)
    assert np.isclose(analyzer.eigenvalues[0], 2.0, atol=1e-6)
