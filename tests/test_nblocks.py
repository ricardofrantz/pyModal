import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import BaseAnalyzer


def dummy_loader(path):
    Nx, Ny = 4, 4
    Ns = 30
    dt = 0.1
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    q = np.random.rand(Ns, Nx * Ny)
    return {"q": q, "x": x, "y": y, "dt": dt, "Nx": Nx, "Ny": Ny, "Ns": Ns}


def compute_block_starts(nfft, novlap, nblocks, Ns):
    step = nfft - novlap
    starts = []
    for k in range(nblocks):
        ts = min(k * step, Ns - nfft)
        starts.append(ts)
    return starts


def test_nblocks_and_block_starts():
    nfft = 8
    overlap = 0.5
    analyzer = BaseAnalyzer("dummy", nfft=nfft, overlap=overlap, data_loader=dummy_loader)
    analyzer.load_and_preprocess()
    expected_nblocks = 1 + int(np.floor((analyzer.data["Ns"] - nfft) / (nfft - analyzer.novlap)))
    assert analyzer.nblocks == expected_nblocks

    starts = compute_block_starts(nfft, analyzer.novlap, analyzer.nblocks, analyzer.data["Ns"])
    step = nfft - analyzer.novlap
    diffs = np.diff(starts)
    assert np.all(diffs == step)

    analyzer.compute_fft_blocks()
    assert analyzer.qhat.shape[2] == analyzer.nblocks
