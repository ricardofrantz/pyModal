import numpy as np
from data_interface import DNamiXNPZLoader


def test_parallel_loading_identical(tmp_path, monkeypatch):
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 1.0])
    dt = 1.0
    for i in range(3):
        arr = np.full((2, 2, 2), i + 1.0)
        times = np.array([0.0, 1.0]) + 2 * i
        np.savez(tmp_path / f'file_{i}.npz', x=x, y=y, dt=dt, times=times, u=arr)

    loader = DNamiXNPZLoader()

    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    data_seq = loader.load(str(tmp_path / 'file_0.npz'))

    monkeypatch.setenv('OMP_NUM_THREADS', '4')
    data_par = loader.load(str(tmp_path / 'file_0.npz'))

    assert np.array_equal(data_seq['q'], data_par['q'])
    assert np.array_equal(data_seq['x'], data_par['x'])
    assert np.array_equal(data_seq['y'], data_par['y'])
    assert data_seq['dt'] == data_par['dt']
    assert data_seq['metadata']['loaded_files'] == data_par['metadata']['loaded_files']
