import numpy as np
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from matplotlib import animation, cm
from matplotlib.patches import Rectangle, Circle
from matplotlib.ticker import FuncFormatter

# Estimate the (row, col) center of concentric waves in `height`.
def droplet_center(height, sigma=2.0, grad_thresh=0.2):
    h_blur = gaussian_filter(height, sigma=sigma)

    dy, dx = np.gradient(h_blur.astype(np.float32))
    mag = np.hypot(dx, dy)

    mask = mag > grad_thresh * mag.max()
    ys, xs = np.nonzero(mask)         # pixel coordinates

    # 4) Build and solve A·c = d   (2×N, N≈few‑thousand)
    a = dy[mask]                      # ∂h/∂y
    b = -dx[mask]                     # -∂h/∂x
    d = a*xs + b*ys                   # right‑hand side

    w = np.sqrt(mag[mask])
    Aw = np.stack([a, b], 1) * w[:, None]
    dw = d * w

    cx, cy = np.linalg.lstsq(Aw, dw, rcond=None)[0]

    return cy, cx

# render height_map in one, two, or three dimensions
def render(height_maps, drop_diameter, scale, render_mode):
    center_y, center_x = droplet_center(height_maps[0])

    if render_mode == 1:
        fig = plt.figure(figsize=(6, 4))
        fig.set_dpi(100)
        ax = plt.gca()
        r, c = height_maps[0].shape

        # draw gray box around droplet center
        bot, top = ax.get_ylim()
        ax.add_patch(Rectangle((center_x - ((drop_diameter / 2) * scale) - .5, bot - .5), drop_diameter * scale, top - bot, facecolor="grey", alpha=0.5))

        global_max = np.max(height_maps[:, int(center_y), :])
        global_min = np.min(height_maps[:, int(center_y), :])

        plt.ylim(global_min * 1.1, global_max * 1.1)
        line, = plt.plot(height_maps[0][int(center_y), :])
        plt.xlabel("Distance (mm)")
        plt.ylabel("Amplitude")

        def pixel_to_mm_formatter(x, pos):
            mm_value = x / scale
            return f"{mm_value:.1f} mm" # format to two decimal places

        formatter = FuncFormatter(pixel_to_mm_formatter)
        ax.xaxis.set_major_formatter(formatter)

        def update(i):
            line.set_ydata(height_maps[i][int(center_y), :])
            return line
        
    if render_mode == 2: # 2d plot
        fig = plt.figure(figsize=(6, 4))
        fig.set_dpi(100)
        im = plt.imshow(height_maps[0], cmap='ocean')
        plt.colorbar()

        # draw gray circle around droplet center
        ax = plt.gca()
        ax.add_patch(Circle((center_x, center_y), (drop_diameter / 2) * scale, facecolor="grey", alpha=0.5))

        def update(i):
            im.set_array(height_maps[i])
            return im


    if render_mode == 3: # 3d plot with matplotlib 
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        r, c = height_maps[0].shape
        X, Y = np.meshgrid(np.arange(c), np.arange(r))
        # ax.set_zlim(-10, 10)
        ax.view_init(elev=25, azim=45, roll=0)

        quality = 10 # 10 is default, 1 is highest

        surf = ax.plot_surface(X, Y, height_maps[0], cmap=cm.ocean, rstride=quality, cstride=quality)
            
        def update(i):
            ax.clear()
            surf = ax.plot_surface(X, Y, height_maps[i], cmap=cm.ocean, rstride=quality, cstride=quality) # update data
            # ax.set_zlim(-10, 10)
            return surf
            
    ani = animation.FuncAnimation(fig, update, frames=len(height_maps), interval=1, blit=False)
    return ani