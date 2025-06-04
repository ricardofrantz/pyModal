import numpy as np
from pod import PODAnalyzer


def test_perform_pod_simple():
    data = {
        'q': np.array([[1, 2], [3, 4], [5, 6]], dtype=float),
        'x': np.array([0.0, 1.0]),
        'y': np.array([0.0]),
        'dt': 1.0,
        'Nx': 2,
        'Ny': 1,
        'Ns': 3,
    }
    analyzer = PODAnalyzer(
        file_path='dummy',
        data_loader=lambda _: data,
        spatial_weight_type='uniform',
        n_modes_save=2,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod()
    assert analyzer.modes.shape == (2, 2)
    assert analyzer.time_coefficients.shape == (3, 2)
    assert np.isclose(analyzer.eigenvalues[0], 5.333333333333333, atol=1e-6)

