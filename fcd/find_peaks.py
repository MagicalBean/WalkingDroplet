import numpy as np
from numpy import array
from pyfftw.interfaces.scipy_fftpack import fft2, fftshift, fftfreq
from scipy.ndimage import label

from kspace import pixel2kspace

def peaks(img: array, threshold, n_peaks, subpixel=False):
    # Step 1: Threshold the image
    blob_img = img > threshold
    blob_img[0, :] = blob_img[-1, :] = False # make the borders false
    blob_img[:, 0] = blob_img[:, -1] = False

    # Step 2: Label connected components
    labeled, num = label(blob_img)
    if num == 0:
        return []
    
    # Step 3: For each component, find the brightest pixel
    peak_coords = []
    peak_vals = []

    for region in range(1, num + 1):
        coords = np.argwhere(labeled == region)
        values = np.array([img[tuple(c)] for c in coords])
        max_idx = np.argmax(values)
        peak_coords.append(coords[max_idx])
        peak_vals.append(values[max_idx])

    # Step 4: sort by intensity
    sorted_peaks = sorted(zip(peak_vals, peak_coords), key=lambda x: -x[0])

    # Step 5: Select top N
    selected = sorted_peaks[:n_peaks]
    coords = np.array([c for _, c in selected], dtype=np.float64)  # float64 for subpixel

    if subpixel:
        rows, cols = img.shape
        for i, (y, x) in enumerate(coords):
            y, x = int(round(y)), int(round(x))
            if 1 <= y < rows - 1 and 1 <= x < cols - 1:
                iiv = img[y, x]
                # Gaussian subpixel adjustment (x-direction)
                lv = img[y, x - 1]
                rv = img[y, x + 1]
                denom = np.log(lv) + np.log(rv) - 2 * np.log(iiv)
                if denom != 0:
                    dx = -0.5 * (np.log(rv) - np.log(lv)) / denom
                    coords[i, 1] += dx
                # Gaussian subpixel adjustment (y-direction)
                tv = img[y - 1, x]
                bv = img[y + 1, x]
                denom = np.log(tv) + np.log(bv) - 2 * np.log(iiv)
                if denom != 0:
                    dy = -0.5 * (np.log(bv) - np.log(tv)) / denom
                    coords[i, 0] += dy

    return coords


def find_peaks(img, kmin=None, kmax=None, thresh=0.5, use_hamming=True):
    r, c = img.shape
    img = img.astype(np.float64)
    img -= np.mean(img)

    # Step 1: optional hamming window
    if use_hamming:
        wr = np.hamming(r)
        wc = np.hamming(c)
        window = np.outer(wr, wc)
        img *= window

    # Step 2: FFT and magnitude
    i_fft = np.abs(fftshift(fft2(img)))

    # Step 3: Get k-space coordinates
    kx = fftshift(fftfreq(c, d=1 / (2.0 * np.pi)))
    ky = fftshift(fftfreq(r, d=1 / (2.0 * np.pi)))
    kxgrid, kygrid = np.meshgrid(kx, ky)
    k2 = kxgrid**2 + kygrid**2

    # Step 4: Apply radial bandpass mask
    if kmin is None:
        kmin = 4.0 * np.pi / min(img.shape)
    if kmax is None:
        kmax = 0.5 * np.max(np.abs(kx)) # or just a large number
    
    mask = (k2 > kmin**2) & (k2 < kmax**2)
    i_fft *= mask

    # Step 5: Find local peaks above threshold
    threshold = thresh * np.max(i_fft)
    peak_locations = peaks(i_fft, threshold, 4, True) # use subpixel accuracy

    if len(peak_locations) < 4:
        raise RuntimeError("Could not detect 4 carrier peaks")
    
    # Step 6: Interpolate pixel2kspace for each peak
    k_locs = np.array([pixel2kspace(i_fft.shape, p) for p in peak_locations])

    # Step 7: Select rightward-pointing peak (min angle to x-axis)
    angles = np.arctan2(k_locs[:, 1], k_locs[:, 0])
    kr_idx = np.argmin(np.abs(angles))
    kr = k_locs[kr_idx]

    # Step 8: Select most orthagonal peak (max cross product magnitude)
    norms = np.linalg.norm(k_locs, axis=1)
    cross = (kr[0] * k_locs[:, 1] - kr[1] * k_locs[:, 0]) / norms
    ku_idx = np.argmax(cross)
    ku = k_locs[ku_idx]

    return kr, ku