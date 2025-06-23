import numpy as np
import matplotlib.pyplot as plt
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


def test_periodogram_strouhal_axis(monkeypatch, tmp_path):
    data = {
        'q': np.random.randn(4, 2),
        'x': np.array([0.0, 1.0]),
        'y': np.array([0.0]),
        'dt': 1.0,
        'Nx': 2,
        'Ny': 1,
        'Ns': 4,
    }
    analyzer = PODAnalyzer(
        file_path='dummy',
        data_loader=lambda _: data,
        figures_dir=tmp_path,
        spatial_weight_type='uniform',
        n_modes_save=1,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod()

    saved = {}
    monkeypatch.setattr(plt, 'savefig', lambda fname, dpi=None: saved.setdefault('fig', plt.gcf()))
    monkeypatch.setattr(plt, 'close', lambda fig=None: None)

    analyzer.plot_time_coefficients(n_coeffs_to_plot=1, L=1.0, U=2.0)
    fig = saved['fig']
    ax = fig.axes[1]
    assert ax.get_xlabel() == 'Strouhal Number (St)'

