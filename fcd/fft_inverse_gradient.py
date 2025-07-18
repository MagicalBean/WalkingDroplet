import numpy as np
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2

from kspace import kvec

# Based on work from Sander Wildeman, which based it on the following paper:
#   Huhn, et al. Exp Fluids (2016), 57, 151, https://doi.org/10.1007/s00348-016-2236-3

def fftinvgrad(fx, fy):
    r, c = fx.shape

    # add impulse to boundaries to compensate for non-periodicity
    imp_i_x = -0.5 * np.sum(fx[:, 1:-1], axis=1)
    imp_i_y = -0.5 * np.sum(fy[1:-1, :], axis=0)
    fx_edge = fx[:, [0, -1]].copy()
    fy_edge = fy[[0, -1], :].copy()
    fx[:, 0] = imp_i_x
    fx[:, -1] = imp_i_x
    fy[0,:] = imp_i_y
    fy[-1,:] = imp_i_y

    # the fourier method will implicitly subtract mean from gradient to satisfy
    # the periodicity assumption, we will tag it back on later
    mx = np.mean(fx)
    my = np.mean(fy)

    kx, ky = np.meshgrid(kvec(c), kvec(r))

    # pre-compute k^2
    k2 = kx**2 + ky**2

    # Remove Nyquist degenracy if dimension is even
    if c % 2 == 0:
        kx[:, c//2] = 0
    if r % 2 == 0:
        ky[r//2, :] = 0

    # compute fft of gradients
    fx_hat = fft2(fx)
    fy_hat = fft2(fy)

    # integrate in fourier domain
    k2[k2 < np.finfo(float).eps] = 1 # shortcut to prevent division by zero (this effectively subtracts a linear plane)
    f_hat = (-1.0j * kx * fx_hat + -1.0j * ky * fy_hat) / k2

    # transform back to spatial domain
    f = np.real(ifft2(f_hat))

    #  add mean slope back on
    x, y = np.meshgrid(np.arange(c), np.arange(r))
    f = f + mx*x + my * y

    # fix edges
    f[:, 0] = (4 * f[:, 1] - f[:, 2] - 2 * fx_edge[:, 0]) / 3
    f[:, -1] = (4 * f[:, -2] - f[:, -3] + 2 * fx_edge[:, 1]) / 3
    f[0, :] = (4 * f[1, :] - f[2, :] - 2 * fy_edge[0, :]) / 3
    f[-1, :] = (4 * f[-2, :] - f[-3, :] + 2 * fy_edge[1, :]) / 3

    return f