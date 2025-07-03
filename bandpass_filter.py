import numpy as np
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2, fftshift, ifftshift

# both of these functions are converted from MatLab to python with the help of ChatGPT
def filt_large_scale(h, large_scale):
    """
    Applies a low-pass Butterworth filter to suppress high-frequency content.
    
    Parameters:
        h (np.ndarray): Input 2D image
        large_scale (float): Cutoff frequency for low-pass filtering
        
    Returns:
        np.ndarray: Low-pass filtered version of the input
    """
    n_order = 3  # Order of the Butterworth filter
    h = h.astype(np.float32)
    nx, ny = h.shape

    # Step 1: FFT
    fftH = fftshift(fft2(h))

    # Step 2: Generate low-pass Butterworth filter
    X, Y = np.meshgrid(np.arange(ny), np.arange(nx))
    D = np.sqrt((X - ny // 2) ** 2 + (Y - nx // 2) ** 2)
    butterworth_filter = 1 / (1 + (D / large_scale) ** (2 * n_order))

    # Step 3: Apply filter
    filtered_fft = fftH * butterworth_filter

    # Step 4: Inverse FFT
    filtered_image = np.real(ifft2(ifftshift(filtered_fft)))

    return filtered_image

# apply Butterworth band-pass filter
# h: height map
def filt_band_pass(h, band, large_scale):
    r, c = h.shape
    extension = 100

    # Padding to square the image
    r, c = h.shape
    if r > c:
        diff = r - c
        left_pad = diff // 2 + extension
        right_pad = diff - diff // 2 + extension  # handles even/odd
        P = np.pad(h, ((extension, extension), (left_pad, right_pad)), mode='symmetric')
    elif c > r:
        diff = c - r
        top_pad = diff // 2 + extension
        bottom_pad = diff - diff // 2 + extension
        P = np.pad(h, ((top_pad, bottom_pad), (extension, extension)), mode='symmetric')
    else:
        P = np.pad(h, extension, mode='symmetric')

    rP, cP = P.shape
    if rP != cP:
        raise ValueError("Padded image is not square!")
    
    # Normalize to 0–255
    I = (P - np.min(h)) / (np.max(h) - np.min(h)) * 255
    I = I.astype(np.uint8)

    # Band-pass filter parameters
    lt, ut = band
    n = 3

    nx, ny = I.shape
    fftI = fftshift(fft2(I, shape=(2*nx-1, 2*ny-1)))

    # Create Butterworth band-pass filter
    y, x = np.ogrid[:2*nx-1, :2*ny-1]
    center_x, center_y = nx, ny
    dist = np.sqrt((x - center_y)**2 + (y - center_x)**2)

    filter1 = 1 / (1 + (dist / ut)**(2*n))
    filter2 = 1 / (1 + (dist / lt)**(2*n))
    filter3 = filter1 * (1 - filter2)

    # Apply filter
    filt_fft_I = fftI * filter3

    # Reconstruct filtered image
    I_filt = ifft2(ifftshift(filt_fft_I))
    I_filt = np.real(I_filt[:nx, :ny])

    # Crop to original
    h1 = I_filt
    if r > c:
        pad_c = (r - c) // 2
        if (r - c) % 2 == 0:
            h2 = h1[extension:-extension, pad_c + extension : -(pad_c + extension)]
        else:
            h2 = h1[:, :-1]
            h2 = h2[extension:-extension, pad_c + extension : -(pad_c + extension)]
    elif r < c:
        pad_r = (c - r) // 2
        if (c - r) % 2 == 0:
            h2 = h1[pad_r + extension : -(pad_r + extension), extension:-extension]
        else:
            h2 = h1[:-1, :]
            h2 = h2[pad_r + extension : -(pad_r + extension), extension:-extension]
    else:
        h2 = h1[extension:-extension, extension:-extension]

    if not h2.shape == h.shape:
        raise ValueError("Shape mismatch after cropping")

    # Rescale
    if large_scale > 0:
        F = 2
        h_hp = filt_large_scale(h, large_scale)
        int_h_hp = np.sum(np.abs(h_hp)**F)
        int_h2 = np.sum(np.abs(h2)**F)
        mult_h = int_h2 / int_h_hp
        h_out = h2 / (mult_h**(1/F))
    else:
        mult_h = 1 / (np.max(h) - np.min(h))
        h_out = h2 / 255 / mult_h

    return h_out