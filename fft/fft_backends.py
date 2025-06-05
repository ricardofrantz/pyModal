"""
Shared FFT backend selection and wrapper utilities for modal decomposition and benchmarking.
"""

from configs import FFT_BACKEND


def accelerate_fft(x, axis=0):
    """FFT using Apple's Accelerate framework via PyObjC or ctypes."""
    import sys
    import numpy as np
    import ctypes
    import ctypes.util

    if sys.platform != 'darwin':
        raise NotImplementedError('Accelerate FFT is only available on macOS.')

    lib_path = ctypes.util.find_library('Accelerate')
    if lib_path is None:
        raise RuntimeError('Accelerate framework not found.')
    accel = ctypes.cdll.LoadLibrary(lib_path)

    class DSPDoubleSplitComplex(ctypes.Structure):
        _fields_ = [
            ('realp', ctypes.POINTER(ctypes.c_double)),
            ('imagp', ctypes.POINTER(ctypes.c_double)),
        ]

    x_arr = np.asarray(x, dtype=np.complex128)
    n = x_arr.shape[axis]
    log2n = int(np.log2(n))
    if 2**log2n != n:
        raise ValueError('vDSP FFT requires power-of-two length.')

    real = np.ascontiguousarray(np.real(x_arr), dtype=np.float64)
    imag = np.ascontiguousarray(np.imag(x_arr), dtype=np.float64)
    split = DSPDoubleSplitComplex(real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                  imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    accel.vDSP_create_fftsetupD.restype = ctypes.c_void_p
    setup = accel.vDSP_create_fftsetupD(ctypes.c_uint(log2n), 2)
    if not setup:
        raise RuntimeError('Failed to create FFT setup.')
    accel.vDSP_fft_zipD(ctypes.c_void_p(setup), ctypes.byref(split), 1, ctypes.c_uint(log2n), 1)
    accel.vDSP_destroy_fftsetupD(ctypes.c_void_p(setup))
    return real + 1j * imag


def scipy_fft(x, axis=0):
    from scipy.fft import fft

    return fft(x, axis=axis)


def numpy_fft(x, axis=0):
    from numpy.fft import fft

    return fft(x, axis=axis)


def tensorflow_fft(x, axis=0):
    import tensorflow as tf

    x_tf = tf.convert_to_tensor(x)
    x_tf_complex = tf.cast(x_tf, tf.complex64)
    return tf.signal.fft(x_tf_complex).numpy()


def torch_fft(x, axis=0):
    import torch

    x_torch = torch.from_numpy(x)
    x_torch_complex = x_torch.type(torch.complex64)
    return torch.fft.fft(x_torch_complex, dim=axis).numpy()


def _pyfftw_fft_impl(x, axis=0):
    import numpy as np
    import pyfftw

    # Match dtype: if float64 or complex128, use complex128; else use complex64
    if np.issubdtype(x.dtype, np.floating):
        dtype = np.complex128 if x.dtype == np.float64 else np.complex64
        x = x.astype(dtype)
    elif x.dtype == np.complex128 or x.dtype == np.complex64:
        dtype = x.dtype
    else:
        dtype = np.complex64
        x = x.astype(dtype)
    a = pyfftw.empty_aligned(x.shape, dtype=dtype)
    a[:] = x
    fft_object = pyfftw.builders.fft(a, axis=axis)
    return fft_object()


def mkl_fft(x, axis=0):
    """FFT via the Intel MKL :mod:`mkl_fft` library."""
    from mkl_fft import fft as mkl_fft_func

    return mkl_fft_func(x, axis=axis)

FFT_BACKENDS = {
    'scipy': scipy_fft,
    'numpy': numpy_fft,
    'tensorflow': tensorflow_fft,
    'torch': torch_fft,
    'mkl': mkl_fft,
    'accelerate': accelerate_fft,
    # Additional backends are appended below when available
}

try:
    import pyfftw  # Attempt to import to check availability

    FFT_BACKENDS["pyfftw"] = _pyfftw_fft_impl
    # print("PyFFTW backend enabled.")
except ImportError:
    print("PyFFTW not installed or found. PyFFTW backend will be unavailable.")
except Exception as e:
    # Catch any other error during PyFFTW probing/loading
    print(f"Error loading PyFFTW backend: {e}. PyFFTW backend will be unavailable.")


def get_fft_func(backend=None):
    backend = backend or FFT_BACKEND
    if backend not in FFT_BACKENDS:
        raise ValueError(f"Unknown FFT_BACKEND: {backend}")
    return FFT_BACKENDS[backend]


def get_fft_backend_names():
    return list(FFT_BACKENDS.keys())
