import time
from typing import List
import os.path
from pathlib import Path

import numpy as np
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
from skimage.io import imread
from more_itertools import flatten
from joblib import Memory
from vidstab import VidStab

from fft_inverse_gradient import fftinvgrad
from subplane import subplane
from  bandpass_filter import filt_band_pass
from calculate_carriers import Carrier, calculate_carriers
from renderer import render

# cache for storing fcd results
cache = Memory(location=Path("cache"), verbose=0)  # no console spam

# initialize video stabilizer
stabilizer = VidStab()

def run_fcd(ref_img_path, def_folder_path, crop_region, scale, drop_diameter, hstar, render_mode=2, input_length=-1, debug=False, progress_cb=None, status_cb=None):
    """
    Wrapper for the fcd function that handles the image preperation.

    Args:
        ref_img_path (str): Path to the reference image.
        def_folder_path (int): Path to the folder containing the definition images.
        crop_region (tuple): Crop region coords in (x1, x2, y1, y2) format
        render_mode (int): Render mode 1, 2, or 3
        input_length (int): The number of definition images to use (default is all files)
        debug (bool): Whether or not to print to the console (default False)

    Returns:
        animation: The rendered matplotlib animation.
    """
    import glob

    if status_cb: status_cb("Processing reference image...")
    
    # Read reference image
    i_ref = imread(ref_img_path, as_gray=True)

    # Grab all images in definition directory
    image_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff") 
    filenames = [f for ext in image_extensions for f in glob.glob(os.path.join(def_folder_path, f'*{ext}'))]
    if input_length == -1: input_length = len(filenames)+1
    filenames = filenames[:input_length] # use files up to input length (or max if none is provided)
    
    files = list(sorted(flatten((glob.glob(x) if '*' in x else [x]) for x in filenames)))

    # crop reference image to crop region
    x1, x2, y1, y2 = crop_region
    if x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0:
        i_ref = i_ref[y1:y2, x1:x2]

    if debug: print(f'processing reference image...', end='')
    carriers = calculate_carriers(i_ref)
    if debug: print('done')

    height_maps = []
    for i, file in enumerate(files):
        # Read deformed image
        if status_cb: status_cb(f"Processing frame ({i+1}/{len(files)})...")
        if progress_cb: progress_cb(int((i + 1) / len(files) * 100))
        if debug: print(f'processing {file} ... ', end='')

        i_def = imread(file, as_gray=True)

        i_def_stab = stabilizer.stabilize_frame(input_frame=i_def,
                                                   smoothing_window=30) # stabilizer will output black frames until smoothing window is completed
                
        if np.sum(i_def_stab) == 0: continue # skip black frames
                
        # Crop deformed image
        if x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0:
            i_def_stab = i_def_stab[y1:y2, x1:x2]

        t0 = time.time()
        height_field = fcd(i_def_stab, carriers, hstar) / (scale**2) # divide by scale^2 (pixel to mm conversion?)
        # height_field_sub = subplane(height_field) # removed to match surferbot example
        height_field_filtered = filt_band_pass(height_field, [25, 200], 0)

        if debug: print(f'done in {time.time() - t0:.2f}s\n')
 
        height_maps.append(height_field_filtered[20:-20, 20:-20])

    if debug: print(f'height map processing done\n')

    n = 8 # frame rate frequency / drive frequency
    map_bins = [np.mean(height_maps[i::n], axis=0) for i in range(n)]
    # data = np.mean(height_maps, axis=0)[center_y, :]

    height_maps = np.stack(map_bins) 

    if status_cb: status_cb("Creating animation...")
    if debug: print("creating animation...")

    # render the 2d color plot by default
    return (height_maps, drop_diameter, scale)
    # ani = render(height_maps, drop_diameter, scale, render_mode)
    # return ani


@cache.cache
def fcd(i_def, carriers: List[Carrier], hstar):
    i_def_fft = fft2(i_def)

    phis = [-np.angle(ifft2(i_def_fft * c.mask) * c.ccsgn) for c in carriers]

    det_a = carriers[0].k_loc[1] * carriers[1].k_loc[0] - carriers[0].k_loc[0] * carriers[1].k_loc[1]
    u = (carriers[0].k_loc[1] * phis[1] - carriers[1].k_loc[1] * phis[0]) / det_a
    v = (carriers[1].k_loc[0] * phis[0] - carriers[0].k_loc[0] * phis[1]) / det_a

    # remove mean (global translation)
    u = u - np.mean(u)
    v = v - np.mean(v)

    # scaling to account for fluid and container depth 
    u /= hstar
    v /= hstar

    return fftinvgrad(-u, -v)