import os
import numpy as np
from spod import SPODAnalyzer


def test_plot_eigenvalues_v2(tmp_path):
    data = {
        'q': np.random.randn(8, 4),
        'x': np.linspace(0, 1, 2),
        'y': np.linspace(0, 1, 2),
        'dt': 1.0,
        'Nx': 2,
        'Ny': 2,
        'Ns': 8,
    }
    analyzer = SPODAnalyzer(
        file_path='dummy.h5',
        nfft=4,
        overlap=0.0,
        results_dir=tmp_path,
        figures_dir=tmp_path,
        data_loader=lambda _: data,
        spatial_weight_type='uniform',
    )
    analyzer.load_and_preprocess()
    analyzer.compute_fft_blocks()
    analyzer.perform_spod()
    analyzer.plot_eigenvalues_v2()
    expected = tmp_path / 'dummy_SPOD_eigenvalues_v2_nfft4_noverlap0.0.png'
    assert expected.exists()
