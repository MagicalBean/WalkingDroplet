import time
from typing import List
import os.path
from pathlib import Path

import numpy as np
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
from skimage.io import imread
from more_itertools import flatten
from joblib import Memory

from fft_inverse_gradient import fftinvgrad
from subplane import subplane
from  bandpass_filter import filt_band_pass
from select_roi import select_region
from calculate_carriers import Carrier, calculate_carriers
from renderer import render

# cache for storing fcd results
cache = Memory(location=Path("cache"), verbose=0)  # no console spam

@cache.cache
def fcd(i_def, carriers: List[Carrier]):
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

# constants
scale = 1 / 0.055; # pix/mm, lengthscale
drop_diameter = 0.78

fluid_depth = 5 # mm (h_l)
acrylic_thickness = 6.35 # mm (h_c)
# hstar formula: (1 - n_a/ n_l) * (h_l + (n_l / n_c)* h_c)
hstar = (1 - (1 / 1.4009)) * (fluid_depth + (1.4009 / 1.4906) * acrylic_thickness); # 5 mm depth, 0.25 is for air/water

if __name__ == "__main__":
    import argparse
    import glob

    start_time = time.time()

    # command line arguments
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('output_folder', type=Path)
    argparser.add_argument('reference_image', type=str)
    argparser.add_argument('definition_folder', type=Path)
    argparser.add_argument('-n', '--output_name', type=str, help='Name of output file')
    argparser.add_argument('-l', '--input_length', type=int, default=-1, help='number of files to process')
    argparser.add_argument('-1d', '--one_d', action='store_true')
    argparser.add_argument('-3d', '--three_d', action='store_true')

    args = argparser.parse_args()

    args.output_folder.mkdir(exist_ok=True)

    # Read reference image
    i_ref = imread(args.reference_image, as_gray=True)

    # Grab all image files in definition directory
    filenames = [f for ext in ('tif', 'tiff', 'png', 'jpg', 'jpeg') for f in glob.glob(os.path.join(args.definition_folder, f'*.{ext}'))]
    if args.input_length == -1: args.input_length = len(filenames)+1
    filenames = filenames[:args.input_length] # use files up to input length (or max if none is provided)
    
    files = list(sorted(flatten((glob.glob(x) if '*' in x else [x]) for x in filenames)))

    # User crop region
    example_img = imread(files[0], as_gray=True)
    x1, x2, y1, y2 = select_region(example_img, args.definition_folder)

    # Crop the reference image
    if x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0:
        i_ref = i_ref[y1:y2, x1:x2]
        
    print(f'processing reference image...', end='')
    carriers = calculate_carriers(i_ref)
    print('done')
    
    height_maps = []
    for file in files:
        # Read deformed image
        print(f'processing {file} ... ', end='')
        i_def = imread(file, as_gray=True)
        
        # Crop deformed image
        if x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0:
            i_def = i_def[y1:y2, x1:x2]

        t0 = time.time()
        height_field = fcd(i_def, carriers) / (scale**2) # divide by scale^2 (pixel to mm conversion?)
        # height_field_sub = subplane(height_field) # removed to match surferbot example
        height_field_filtered = filt_band_pass(height_field, [25, 200], 0)

        print(f'done in {time.time() - t0:.2f}s\n')
 
        height_maps.append(height_field_filtered[20:-20, 20:-20])

    print(f'height map processing done in {time.time() - start_time:.2f}s\n')

    n = 8 # frame rate frequency / drive frequency
    map_bins = [np.mean(height_maps[i::n], axis=0) for i in range(n)]
    # data = np.mean(height_maps, axis=0)[center_y, :]

    height_maps = np.stack(map_bins) 

    print("creating animation...", end='')
    ani_time = time.time()

    ani_name = args.output_name if args.output_name is not None else Path(files[0]).stem
    output_path = args.output_folder.joinpath(f'{ani_name}.mp4')

    # render the 2d color plot by default
    dim = 1 if args.one_d else 3 if args.three_d else 2
    render(height_maps, drop_diameter, scale, output_path, dim)

    print(f' done in {time.time() - ani_time:.2f}s\n')
    print(f'Total runtime {time.time() - start_time:.2f}s\n')