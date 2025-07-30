# WalkingDroplet

This repository contains all the project files for our Summer Collaborative Research Program (SCRP) project focused on Fast Checkerboard Demodulation (FCD) analysis of walking droplets.

## Contents

- [`fcd/`](./fcd): Fast Checkerboard Demodulation implementation adapted from Wildeman's original [FCD](https://github.com/swildeman/fcd) and Kaspervn's [PyFCD](https://github.com/kaspervn/pyfcd) Python port.

- [`scripts/`](./scripts): Assorted utilities and analysis tools developed during the experiment setup and testing phase. Includes edge detection and droplet diameter measurement using OpenCV.

We are using this code alongside the droplet generator from the Harris Lab at Brown University: https://github.com/harrislab-brown/DropGen

## Usage

You can run the Fast Checkerboard Demodulation (FCD) tool in two ways: through the command line or the graphical interface.

### Command-Line Interface (CLI):
Run the FCD pipeline directly from the terminal:
```bash
python  fcd\fcd.py  <output_dir> <reference_image_path> <definition_images_dir> [options]
```

#### Required Arguments

- `<output_folder>` – Path to the folder where output results will be saved
- `<reference_image_path>` – Path to the reference (undeformed) checkerboard image
- `<definition_images_folder>` – Path to the folder containing definition (distorted) images

#### Optional Flags

- `-n <output_name>` – name for the output file
- `-l <#>` – Limit the number of frames to process
- `-1d` – Enable wave profile rendering
- `-3d` – Enable 3D surface wave reconstruction rendering

### Graphical User Interface (GUI):
You can also use a GUI to run the FCD pipeline.
#### To launch the GUI:
```bash
python fcd\gui_app.py
```
or double click the `gui_app.py` file

#### How to use:
1.  **Launch** the GUI
2.  **Select** the reference image and definition folder
3.  **Crop** the region of interest or load a previous ROI
4.  **Choose** render mode (1D wave profile / 2D color map / 3D height map)
5.  _(Optional)_ Click “Advanced” to tune scale and fluid parameters
6.  Click **Run** to start processing
7.  After completion, click **Save** to export the result

## Installation
Clone the repo and install dependencies:
```bash
git  clone  https://github.com/MagicalBean/WalkingDroplet.git
cd  WalkingDroplet
pip install -r requirements.txt
```

You might also need to install [ffmpeg](https://ffmpeg.org/download.html).