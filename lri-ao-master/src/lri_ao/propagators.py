import numpy as np


def fourier_transform_propagator(field, shift=False):
    """Wrapping of the numpy fourier transform in order to propagate
    Coordinate_system style planes"""
    if shift:
        fft = np.fft.fft2(field, norm="ortho")
        return [fft, np.fft.fftshift(fft, axes=(1, 2))]
    else:
        return np.fft.fft2(field, norm="ortho")
    # if shift:
    #     return np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft2(field, norm="ortho"), axes=(1, 2))
    # else:
    #     return pyfftw.interfaces.numpy_fft.fft2(field, norm="ortho")


def fourier_auto_correlator(field, shift=True):
    """Given a field, calculate it's auto correlation in order to get
    non coherent imaging"""
    if shift:
        return np.fft.fftshift(
            np.fft.ifft2(np.abs(np.fft.fft2(field, norm="ortho")) ** 2, norm="ortho"),
            axes=(1, 2),
        )
    else:
        return np.fft.ifft2(np.abs(np.fft.fft2(field, norm="ortho")) ** 2, norm="ortho")
