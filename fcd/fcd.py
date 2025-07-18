import time
from dataclasses import dataclass
from typing import List
import os.path

import numpy as np
from numpy import array
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
from skimage.io import imread
from more_itertools import flatten

from fft_inverse_gradient import fftinvgrad
from find_peaks import find_peaks
from kspace import kvec
from subplane import subplane
from  bandpass_filter import filt_band_pass

import matplotlib.pyplot as plt
from matplotlib import animation, cm
from matplotlib.widgets import RectangleSelector

# complex conjugate of the inverse fft of the masked i_ref_fft
def ccsgn(i_ref_fft, mask):
    return np.conj(ifft2(i_ref_fft * mask))

@dataclass
class Carrier:
    k_loc: array
    krad: float
    mask: array
    ccsgn: array

def calculate_carriers(i_ref):
    kr, ku = find_peaks(i_ref)
    
    peak_radius = np.sqrt(np.sum((kr - ku)**2)) / 2
    i_ref_fft = fft2(i_ref)

    def create_mask(shape, kc, krad):
        r, c = shape
        kx, ky = np.meshgrid(kvec(c), kvec(r))
        # Build the circular mask in k-space centered on kc = [kx, ky]
        return ((kx - kc[0])**2 + (ky - kc[1])**2) < krad**2

    carriers = []
    for k_loc in [kr, ku]:
        mask = create_mask(i_ref.shape, k_loc, peak_radius)
        carrier = Carrier(k_loc=np.array(k_loc),
                          krad=peak_radius,
                          mask=mask,
                          ccsgn=ccsgn(i_ref_fft, mask))
        carriers.append(carrier) 
    
    return carriers

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

# previews deformed image and allows user to select a square region of it for analysis (if left blank it selects the whole image)
def select_region(img):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title("Drag to select a region for analysis")

    # close window and submit if enter key is pressed
    def on_press(event):
        if event.key == 'enter':
            plt.close()

    rect_selector = RectangleSelector(ax, useblit=True,
                                      button=[1], # left mouse button
                                      minspanx=5, minspany=5, spancoords='pixels', interactive=True, state_modifier_keys=dict(square=''))
    
    # require region to be square
    rect_selector.add_state('square')
    fig.canvas.mpl_connect('key_press_event', on_press)
    plt.show()

    # x1, x2, y1, y2 = rect_selector.extents
    return tuple(map(round, rect_selector.extents))

# constants
scale = 1 / 0.105; # pix/mm, lengthscale
# hstar formula: (1 - n_a/ n_l) * (h_l + (n_l / n_c)* h_c)
fluid_depth = 5 # mm
acrylic_thickness = 6.35 # mm
hstar = (1 - (1 / 1.4009)) * (fluid_depth + (1.4009 / 1.4906) * acrylic_thickness); # 5 mm depth, 0.25 is for air/water

if __name__ == "__main__":
    import argparse
    import glob
    from pathlib import Path

    start_time = time.time()

    # command line arguments
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('output_folder', type=Path)
    argparser.add_argument('reference_image', type=str)
    argparser.add_argument('definition_folder', type=Path)
    argparser.add_argument('-n', '--output_name', type=str, help='Name of output file')
    argparser.add_argument('-l', '--input_length', type=int, default=-1, help='number of files to process')
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
    x1, x2, y1, y2 = select_region(example_img)

    # Crop the reference image
    if x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0:
        i_ref = i_ref[y1:y2, x1:x2]
        
    print(f'processing reference image...', end='')
    carriers = calculate_carriers(i_ref)
    print('done')
    
    height_maps = []
    for file in files:
        # Read deformed image
        i_def = imread(file, as_gray=True)
        
        # Crop deformed image
        if x1 != 0 and x2 != 0 and y1 != 0 and y2 != 0:
            i_def = i_def[y1:y2, x1:x2]

        t0 = time.time()
        height_field = fcd(i_def, carriers) / (scale**2) # divide by scale^2 (pixel to mm conversion?)
        # height_field_sub = subplane(height_field) # removed to match surferbot example
        height_field_filtered = filt_band_pass(height_field, [25, 200], 0)

        print(f'done in {time.time() - t0:.2f}s\n')
 
        height_maps.append(height_field_filtered[5:-5, 5:-5])

    print(f'height map processing done in {time.time() - start_time:.2f}s\n')
    print("creating animation...", end='')
    ani_time = time.time()

    if args.three_d: # 3d plot with matplotlib 
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        r, c = height_maps[0].shape
        X, Y = np.meshgrid(np.arange(c), np.arange(r))
        ax.set_zlim(-10, 10)
        ax.view_init(elev=25, azim=45, roll=0)

        quality = 10 # 10 is default, 1 is highest

        surf = ax.plot_surface(X, Y, height_maps[0], cmap=cm.ocean, rstride=quality, cstride=quality)
            
        def update(i):
            ax.clear()
            surf = ax.plot_surface(X, Y, height_maps[i], cmap=cm.ocean, rstride=quality, cstride=quality) # update data
            ax.set_zlim(-10, 10)
            return surf
    else: # 2d plot
        fig = plt.figure(figsize=(6, 4))
        fig.set_dpi(100)
        im = plt.imshow(height_maps[0], cmap='ocean')
        plt.colorbar()

        def update(i):
            im.set_array(height_maps[i])
            return im
    
    ani = animation.FuncAnimation(fig, update, frames=len(height_maps), interval=1, blit=False)
    ani_name = args.output_name if args.output_name is not None else Path(files[0]).stem
    ani.save(args.output_folder.joinpath(f'{ani_name}.mp4'), writer='ffmpeg', fps=30)
    print(f' done in {time.time() - ani_time:.2f}s\n')
    print(f'Total runtime {time.time() - start_time:.2f}s\n')
    plt.show()
