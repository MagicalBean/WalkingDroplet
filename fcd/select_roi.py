import json
import hashlib
from pathlib import Path
import os

from skimage.io import imread

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle

def roi_filename_for(folder_path, cache_root=Path("cache")):
    folder_path = folder_path.resolve()
    folder_hash = hashlib.sha1(str(folder_path).encode()).hexdigest()[:10]
    return cache_root / f"roi_{folder_hash}.json"

def show_saved_roi(image, roi_path):
    with open(roi_path) as f:
        roi = json.load(f)

    x1, x2, y1, y2 = roi

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', facecolor='none', lw=2)
    ax.add_patch(rect)
    ax.set_title(f"Use existing ROI at {roi_path.name}? (Y/n): ")

    answer = True
    def on_press(event):
        nonlocal answer
        if event.key == 'n':
            answer = False
        plt.close()

    fig.canvas.mpl_connect('key_press_event', on_press)
    
    plt.show()

    return answer

def get_first_image(i_def_path):
    image_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    img_path = next((os.path.join(i_def_path, f) for f in os.listdir(i_def_path) if f.lower().endswith(image_extensions)), None)
    return imread(img_path, as_gray=True)

# previews deformed image and allows user to select a square region of it for analysis (if left blank it selects the whole image)

# if user_override is False and preview is True, this works as normal
# if user_override is True and preview is False, this will for a manual region selection

def select_region(i_def_path, user_override=False, preview=True):
    roi_path = roi_filename_for(Path(i_def_path))
    use_existing = roi_path.exists() and not user_override

    # select preview image
    img = get_first_image(i_def_path)

    if use_existing and preview:
        use_existing = show_saved_roi(img, roi_path)

    if use_existing:
        with open(roi_path) as f:
            roi = json.load(f)
    else:
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')

        # close window and submit if enter key is pressed
        def on_press(event):
            if event.key == 'enter':
                plt.close()

        rect_selector = RectangleSelector(ax, useblit=True,
                                        button=[1], # left mouse button,
                                        minspanx=5, minspany=5, spancoords='pixels', interactive=True, state_modifier_keys=dict(square=''))
        
        # require region to be square
        rect_selector.add_state('square')
        fig.canvas.mpl_connect('key_press_event', on_press)
        ax.set_title("Drag to select a region for analysis")
        plt.show()

        roi = tuple(map(round, rect_selector.extents))

        with open(roi_path, "w") as f:
            json.dump(roi, f)

    # x1, x2, y1, y2 = rect_selector.extents
    return roi
