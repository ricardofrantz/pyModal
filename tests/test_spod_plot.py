
import numpy as np

from spod import SPODAnalyzer


def test_plot_eigenvalues_v2(tmp_path):
    data = {
        "q": np.random.randn(8, 4),
        "x": np.linspace(0, 1, 2),
        "y": np.linspace(0, 1, 2),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 2,
        "Ns": 8,
    }
    analyzer = SPODAnalyzer(
        file_path="dummy.h5",
        nfft=4,
        overlap=0.0,
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_spod()
    analyzer.plot_eigenvalues_v2()
    expected = tmp_path / "dummy_SPOD_eigenvalues_v2_nfft4_noverlap0.0.png"
    assert expected.exists()


def test_plot_modes_and_timecoeffs(tmp_path):
    data = {
        "q": np.random.randn(8, 4),
        "x": np.linspace(0, 1, 2),
        "y": np.linspace(0, 1, 2),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 2,
        "Ns": 8,
    }
    analyzer = SPODAnalyzer(
        file_path="dummy.h5",
        nfft=4,
        overlap=0.0,
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_spod()
    analyzer.plot_modes(plot_n_modes=1)
    st = analyzer.St[np.argmax(analyzer.eigenvalues[:, 0])]
    expected_modes = tmp_path / f"dummy_SPOD_mode1_St{st:.4f}_q.png"
    assert expected_modes.exists()
    analyzer.plot_time_coeffs()
    expected_time = tmp_path / f"dummy_SPOD_timecoeffs_St{st:.4f}_nfft4_noverlap0.0.png"
    assert expected_time.exists()
