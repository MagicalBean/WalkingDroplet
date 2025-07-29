import json
import hashlib
from pathlib import Path

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

# previews deformed image and allows user to select a square region of it for analysis (if left blank it selects the whole image)
def select_region(img_path, i_def_path):
    roi_path = roi_filename_for(i_def_path)
    use_existing = roi_path.exists()

    img = imread(img_path, as_gray=True)


    if use_existing:
        use_existing = show_saved_roi(img, roi_path)

    if use_existing:
        with open(roi_path) as f:
            roi = json.load(f)
    else:
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

        roi = tuple(map(round, rect_selector.extents))

        with open(roi_path, "w") as f:
            json.dump(roi, f)

    # x1, x2, y1, y2 = rect_selector.extents
    return roi
