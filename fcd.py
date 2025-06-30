import time
from dataclasses import dataclass
from typing import List
import os.path

import numpy as np
from numpy import array
from scipy.fft import fft2, ifft2, ifftshift
from skimage.draw import disk
from skimage.io import imread
from more_itertools import flatten

from fft_inverse_gradient import fftinvgrad
from find_peaks import find_peaks
from kspace import pixel2kspace
from subplane import subplane

import matplotlib.pyplot as plt
from matplotlib import animation, cm
from matplotlib.widgets import RectangleSelector

def normalize_image(img):
    return (img - img.min()) / (img.max()-img.min())

def peak_mask(shape, pos, r):
    result = np.zeros(shape, dtype=bool)
    result[disk(pos, r, shape=shape)] = True
    return result


def ccsgn(i_ref_fft, mask):
    return np.conj(ifft2(i_ref_fft * mask))


@dataclass
class Carrier:
    pixel_loc: array
    k_loc: array
    krad: float
    mask: array
    ccsgn: array


def calculate_carriers(i_ref):
    peaks = find_peaks(i_ref)
    peak_radius = np.linalg.norm(peaks[0] - peaks[1]) / 2
    i_ref_fft = fft2(i_ref)

    carriers = [Carrier(peak, pixel2kspace(i_ref.shape, peak), peak_radius, mask, ccsgn(i_ref_fft, mask)) for mask, peak
                in
                [(ifftshift(peak_mask(i_ref.shape, peak, peak_radius)), peak) for peak in peaks]]
    return carriers


def fcd(i_def, carriers: List[Carrier]):
    global i_def_fft
    i_def_fft = fft2(i_def)

    phis = [-np.angle(ifft2(i_def_fft * c.mask) * c.ccsgn) for c in carriers]

    # scale by 0.25 * depth (H * 1 - n_a / n_w)

    det_a = carriers[0].k_loc[1] * carriers[1].k_loc[0] - carriers[0].k_loc[0] * carriers[1].k_loc[1]
    u = (carriers[1].k_loc[0] * phis[0] - carriers[0].k_loc[0] * phis[1]) / det_a
    v = (carriers[0].k_loc[1] * phis[1] - carriers[1].k_loc[1] * phis[0]) / det_a

    # scaling to account for fluid depth (recommended by Jack)
    u /= hstar
    v /= hstar

    return fftinvgrad(-u, -v)

def select_region(img):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Drag to select a rectangular region")

    def on_press(event):
        if event.key == 'enter':
            plt.close()

    rect_selector = RectangleSelector(ax, useblit=True,
                                      button=[1], # left mouse button
                                      minspanx=5, minspany=5, spancoords='pixels', interactive=True, state_modifier_keys=dict(square=''))
    
    rect_selector.add_state('square')
    fig.canvas.mpl_connect('key_press_event', on_press)
    plt.show()

    # x1, x2, y1, y2 = rect_selector.extents
    return tuple(map(round, rect_selector.extents))

# constants
scale= 17.8 / 3; # pix/mm, lengthscale
hstar= 0.25 * 5; # 5 mm depth, 0.25 is for air/water

if __name__ == "__main__":
    import argparse
    import glob
    from pathlib import Path

    start_time = time.time()

    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('output_folder', type=Path)
    argparser.add_argument('reference_image', type=str)
    argparser.add_argument('definition_image', nargs='+', help='May contain wildcards')
    argparser.add_argument('-l', '--input_length', type=int, default=-1, help='number of files to process')
    argparser.add_argument('-n', '--output_name', type=str, help='Name of output file')
    argparser.add_argument('--output-format', default='tiff', choices=['tiff', 'bmp', 'png', 'jpg', 'jpeg'], help='The output format')
    argparser.add_argument('--skip-existing', action='store_true', help='Skip processing an image if the output file already exists')

    args = argparser.parse_args()

    args.output_folder.mkdir(exist_ok=True)

    i_ref = imread(args.reference_image, as_gray=True)

    if args.input_length == -1:
        filenames = [args.definition_image[0] + x for x in os.listdir(args.definition_image[0]) if x.endswith(('.tif', '.tiff', '.png'))]
    else:
        filenames = [args.definition_image[0] + x for x in os.listdir(args.definition_image[0])[:args.input_length] if x.endswith(('.tif', '.tiff'))]
    
    files = list(sorted(flatten((glob.glob(x) if '*' in x else [x]) for x in filenames)))

    example_img = imread(files[0], as_gray=True)
    x1, x2, y1, y2 = select_region(example_img)
    
    i_ref = i_ref[y1:y2, x1:x2]
    
    print(f'processing reference image...', end='')
    carriers = calculate_carriers(i_ref)
    print('done')
    
    grids = []

    for file in files:
        output_file_path = args.output_folder.joinpath(f'{Path(file).stem}.{args.output_format}')

        if os.path.abspath(file).lower() == os.path.abspath(output_file_path).lower():
            print(f'Warning: Skipping converting {file} because it would overwrite a input file')
            continue

        if args.skip_existing and output_file_path.exists():
            continue

        print(f'processing {file} -> {output_file_path} ... ', end='')
        i_def = imread(file, as_gray=True)
        
        i_def = i_def[y1:y2, x1:x2]

        t0 = time.time()
        height_field = fcd(i_def, carriers)
        height_field_sub = subplane(height_field) / (scale * scale) # divide by scale^2 (pixel to mm conversion?)
        # band pass filter

        print(f'done in {time.time() - t0:.2f}s\n')
 
        grids.append(height_field_sub[5:-5, 5:-5])

    print(f'height map processing done in {time.time() - start_time:.2f}s\n')
    print("creating animation...", end='')
    ani_time = time.time()
    
    # 3d plot with matplotlib 
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    x = range(0, len(grids[0]), 1)
    X, Y = np.meshgrid(x, x)
    ax.set_zlim(-100, 100)
    ax.view_init(elev=25, azim=45, roll=0)

    quality = 10 # 10 is default, 1 is highest

    surf = ax.plot_surface(X, Y, grids[0], cmap=cm.ocean, rstride=quality, cstride=quality)
        
    def update(i):
        ax.clear()
        surf = ax.plot_surface(X, Y, grids[i], cmap=cm.ocean, rstride=quality, cstride=quality) # update data
        ax.set_zlim(-100, 100)
        return surf
    
    ani = animation.FuncAnimation(fig, update, frames=len(grids), interval=1, blit=False)
    ani.save(args.output_folder.joinpath(f'{args.output_name}.mp4'), writer='ffmpeg', fps=30)
    print(f' done in {time.time() - ani_time:.2f}s\n')
    print(f'Total runtime {time.time() - start_time:.2f}s\n')
