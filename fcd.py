import time
from dataclasses import dataclass
from typing import List
import os.path

import imageio.v2 as imageio
import numpy as np
from numpy import array
from scipy.fft import fft2, ifft2, ifftshift
from skimage.draw import disk
from skimage.io import imread
from more_itertools import flatten

from fft_inverse_gradient import fftinvgrad
from find_peaks import find_peaks
from kspace import pixel2kspace

import matplotlib.pyplot as plt
from matplotlib import animation, cm

from subplane import subplane

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

    det_a = carriers[0].k_loc[1] * carriers[1].k_loc[0] - carriers[0].k_loc[0] * carriers[1].k_loc[1]
    u = (carriers[1].k_loc[0] * phis[0] - carriers[0].k_loc[0] * phis[1]) / det_a
    v = (carriers[0].k_loc[1] * phis[1] - carriers[1].k_loc[1] * phis[0]) / det_a

    return fftinvgrad(-u, -v)

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
    
    y_0, x_0 = np.asarray(i_ref.shape) // 2
    
    x_0 += 0
    y_0 -= 0
    im_size = 522//2 # half-extends
    
    i_ref = i_ref[y_0 - im_size : y_0 + im_size, x_0 - im_size : x_0 + im_size]
    
    # plt.imshow(i_ref, cmap='gray')
    # plt.show()
    # print(max(i_ref[0]), min(i_ref[0]))

    print(f'processing reference image...', end='')
    carriers = calculate_carriers(i_ref)
    print('done')
    
    if args.input_length == -1:
        filenames = [args.definition_image[0] + x for x in os.listdir(args.definition_image[0]) if x.endswith(('.tif', '.tiff'))]
    else:
        filenames = [args.definition_image[0] + x for x in os.listdir(args.definition_image[0])[:args.input_length] if x.endswith(('.tif', '.tiff'))]
    

    files = list(sorted(flatten((glob.glob(x) if '*' in x else [x]) for x in filenames)))

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
        
        i_def = i_def[y_0 - im_size : y_0 + im_size, x_0 - im_size : x_0 + im_size]
        # plt.imshow(i_def, cmap='grey')
        # plt.show()
        t0 = time.time()
        height_field = fcd(i_def, carriers)
        height_field_sub = subplane(height_field)
        print(f'done in {time.time() - t0:.2f}s\n')

        # imageio.imwrite(output_file_path, (normalize_image(height_field) * 255.0).astype(np.uint8))

        # fig = plt.figure(figsize=plt.figaspect(0.5))
        # ax = fig.add_subplot(1, 2, 1, projection='3d')

        # x = range(0, len(height_field), 1)
        # X, Y = np.meshgrid(x, x)
        # surf = ax.plot_surface(X, Y, height_field, cmap=cm.ocean, antialiased=True)
        # ax.set_zlim(-2000, 2000)

        # ax = fig.add_subplot(1, 2, 2, projection='3d')
        # surf = ax.plot_surface(X, Y, height_field_2, cmap=cm.ocean, antialiased=True)
        # ax.set_zlim(-2000, 2000)

        # plt.show()

        grids.append(height_field_sub[5:-5, 5:-5])

    print(f'height map processing done in {time.time() - start_time:.2f}s\n')
    print("creating animation...", end='')
    ani_time = time.time()
    
    # 3d plot with matplotlib 
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    x = range(0, len(grids[0]), 1)
    X, Y = np.meshgrid(x, x)
    ax.set_zlim(-2000, 2000)

    surf = ax.plot_surface(X, Y, grids[0], cmap=cm.ocean)
        
    def update(i):
        ax.clear()
        surf = ax.plot_surface(X, Y, grids[i], cmap=cm.ocean) # update data
        ax.set_zlim(-2000, 2000)
        return surf
    
    ani = animation.FuncAnimation(fig, update, frames=len(grids), interval=1, blit=False)
    ani.save(args.output_folder.joinpath(f'{args.output_name}.mp4'), writer='ffmpeg', fps=20)
    print(f' done in {time.time() - ani_time:.2f}s\n')
    print(f'Total runtime {time.time() - start_time:.2f}s\n')
    # plt.show()
