import functools

import numpy as np
from pyfftw.interfaces.scipy_fftpack import fftshift, fftfreq

def kvec(N):
    if N % 2 == 0: # even
        k = np.concatenate((np.arange(0, N//2), np.arange(N//2, N) - N))
    else: # odd
        k = np.concatenate((np.arange(0, (N-1)//2 + 1), np.arange((N+1)//2, N) - N))
    return k * (2.0 * np.pi) / N 

# TODO: update to use caching
def pixel2kspace(img_shape, location):
    rows, cols = img_shape
    y, x = location  # row, col (float allowed)

    # Create fftshifted k-space vectors
    ky = fftshift(fftfreq(rows, d=1 / (2.0 * np.pi)))
    kx = fftshift(fftfreq(cols, d=1 / (2.0 * np.pi)))

    # Interpolate in y (rows / ky)
    y0 = int(np.floor(y))
    y1 = y0 + 1
    alpha_y = y - y0
    if y1 >= len(ky):
        y1 = y0  # clamp to edge
    ky_interp = (1 - alpha_y) * ky[y0] + alpha_y * ky[y1]

    # Interpolate in x (cols / kx)
    x0 = int(np.floor(x))
    x1 = x0 + 1
    alpha_x = x - x0
    if x1 >= len(kx):
        x1 = x0  # clamp to edge
    kx_interp = (1 - alpha_x) * kx[x0] + alpha_x * kx[x1]

    return np.array([kx_interp, ky_interp])