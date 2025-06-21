import numpy as np
import matplotlib

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


def test_plot_eigenspectra_stem_compat(monkeypatch, tmp_path):
    data = {
        'q': np.random.randn(8, 4),
        'x': np.linspace(0, 1, 2),
        'y': np.linspace(0, 1, 2),
        'dt': 1.0,
        'Nx': 2,
        'Ny': 2,
        'Ns': 8,
    }
    analyzer = DMDAnalyzer(
        file_path='dummy.h5',
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type='uniform',
    )
    analyzer.load_and_preprocess()
    analyzer.perform_dmd()

    calls = []

    def stem_no_use(self, x, y, linefmt=None, markerfmt=None, basefmt=None):
        calls.append('no_use')
        return None

    monkeypatch.setattr(matplotlib.axes.Axes, 'stem', stem_no_use)
    analyzer.plot_eigenspectra()
    assert 'no_use' in calls

    calls.clear()

    def stem_use(self, x, y, linefmt=None, markerfmt=None, basefmt=None, use_line_collection=True):
        calls.append('use_line_collection' if use_line_collection else 'use')
        return None

    monkeypatch.setattr(matplotlib.axes.Axes, 'stem', stem_use)
    analyzer.plot_eigenspectra()
    assert 'use_line_collection' in calls

    expected = tmp_path / 'dummy_dmd_eigenspectra.png'
    assert expected.exists()
