# WalkingDroplet

This repository contains all the project files for our Summer Collaborative Research Program (SCRP) project focused on Fast Checkerboard Demodulation (FCD) analysis of walking droplets.

## Contents

- [`fcd/`](./fcd): Fast Checkerboard Demodulation implementation adapted from Wildeman's original [FCD](https://github.com/swildeman/fcd) and Kaspervn's [PyFCD](https://github.com/kaspervn/pyfcd) Python port.

- [`scripts/`](./scripts): Assorted utilities and analysis tools developed during the experiment setup and testing phase. Includes edge detection and droplet diameter measurement using OpenCV.

We are using this code alongside the droplet generator from the Harris Lab at Brown University: https://github.com/harrislab-brown/DropGen

## Usage

### Command line syntax:

```bash
python  fcd.py  <output_dir> <reference_image_path> <definition_images_dir [options]
```

### Required Arguments

- `<output_folder>` – Path to the folder where output results will be saved
- `<reference_image_path>` – Path to the reference (undeformed) checkerboard image
- `<definition_images_folder>` – Path to the folder containing definition (distorted) images

### Optional Flags

- `-n <output_name>` – name for the output file
- `-l <#>` – Limit the number of frames to process
- `-3d` – Enable 3D surface wave reconstruction rendering

## Installation

Clone the repo and install dependencies:
```bash
git  clone  https://github.com/MagicalBean/WalkingDroplet.git
cd  WalkingDroplet
pip install -r requirements.txt
```