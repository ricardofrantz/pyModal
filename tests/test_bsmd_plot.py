import numpy as np

from bmsd import BSMDAnalyzer


def _make_analyzer(tmp_path):
    data = {
        "q": np.random.randn(10, 4),
        "x": np.linspace(0, 1, 2),
        "y": np.linspace(0, 1, 2),
        "dt": 1.0,
        "Nx": 2,
        "Ny": 2,
        "Ns": 10,
    }
    analyzer = BSMDAnalyzer(
        file_path="dummy.h5",
        nfft=4,
        overlap=0.0,
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type="uniform",
        use_static_triads=True,
        static_triads=[(0, 0, 0)],
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer._perform_static_bsmd_core()
    return analyzer


def test_plot_eigenvalues(tmp_path):
    analyzer = _make_analyzer(tmp_path)
    analyzer.plot_eigenvalues()
    expected = tmp_path / "dummy_BSMD_eigenvalues.png"
    assert expected.exists()


def test_plot_eig_complex_plane(tmp_path):
    analyzer = _make_analyzer(tmp_path)
    analyzer.plot_eig_complex_plane()
    expected = tmp_path / "dummy_BSMD_eig_complex_plane.png"
    assert expected.exists()


def test_plot_energy_plane(tmp_path):
    analyzer = _make_analyzer(tmp_path)
    analyzer.plot_energy_plane()
    expected = tmp_path / "dummy_BSMD_eig_St1St2_plane.png"
    assert expected.exists()
