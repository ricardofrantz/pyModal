import numpy as np
from scipy import signal
from pod import PODAnalyzer


def test_perform_pod_simple():
    data = {
        "q": np.array([[1, 2], [3, 4], [5, 6]], dtype=float),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 3,
    }
    analyzer = PODAnalyzer(
        file_path="dummy",
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,
    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod()
    assert analyzer.modes.shape == (2, 2)
    assert analyzer.time_coefficients.shape == (3, 2)
    assert np.isclose(analyzer.eigenvalues[0], 5.333333333333333, atol=1e-6)


def test_plot_time_coefficients_strouhal(monkeypatch, tmp_path):
    data = {
        "q": np.array([[1, 2], [3, 4], [5, 6]], dtype=float),
        "x": np.array([0.0, 1.0]),
        "y": np.array([0.0]),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 1,
        "Ns": 3,
    }
    analyzer = PODAnalyzer(
        file_path="dummy",
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        n_modes_save=2,

    )
    analyzer.load_and_preprocess()
    analyzer.perform_pod()

    labels = []
    x_data = []

    def xlabel_mock(text):
        labels.append(text)

    def semilogy_mock(x, y, **kwargs):
        x_data.append(np.array(x))
        return None

    monkeypatch.setattr("matplotlib.pyplot.xlabel", xlabel_mock)
    monkeypatch.setattr("matplotlib.pyplot.semilogy", semilogy_mock)

    analyzer.plot_time_coefficients(n_coeffs_to_plot=1, L=2.0, U=4.0)

    assert "Strouhal Number (St)" in labels
    coeff = analyzer.time_coefficients[:3, 0]
    freqs, _ = signal.periodogram(coeff, analyzer.fs, scaling="density")
    expected = freqs * 2.0 / 4.0
    assert np.allclose(x_data[0], expected)

