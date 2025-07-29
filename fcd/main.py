import time
import os
from pathlib import Path

from select_roi import select_region
from fcd import run_fcd

import matplotlib.pyplot as plt

if __name__ == "__main__":
    import argparse
    
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

    # user crop region
    crop_region = select_region(args.definition_folder)

    # render mode
    render_mode = 1 if args.one_d else 3 if args.three_d else 2

    # run fcd
    ani = run_fcd(args.reference_image, args.definition_folder, crop_region, render_mode, debug=True)

    ani_name = args.output_name if args.output_name is not None else Path(img_path).stem
    output_path = args.output_folder.joinpath(f'{ani_name}.mp4')

    ani.save(output_path, writer='ffmpeg', fps=10)
    print(f'Total runtime {time.time() - start_time:.2f}s\n')
    
    plt.show()
