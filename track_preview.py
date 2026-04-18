

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from config import cfg
from env.track import Track


ROWS, COLS = 4, 4
TOTAL = ROWS * COLS
TRACK_ALPHA = 0.25
SAVE_PATH = "checkpoints/track_preview.png"


cfg.random_track = True


fig, axes = plt.subplots(
    ROWS, COLS,
    figsize=(16, 11),
    facecolor="#0f0f14",
)
fig.suptitle(
    "Procedural Random Track Preview  ·  16 unique layouts",
    color="#e8e8f0", fontsize=16, fontweight="bold", y=0.98,
)

rng = np.random.default_rng(seed=42)

for idx, ax in enumerate(axes.flat):
    ax.set_facecolor("#141420")
    ax.set_aspect("equal")
    ax.axis("off")


    t = Track(
        screen_width=cfg.screen_width,
        screen_height=cfg.screen_height,
        track_width=cfg.track_width,
        random_track=True,
        track_min_radius=cfg.track_min_radius,
        track_max_radius=cfg.track_max_radius,
        track_num_waypoints=cfg.track_num_waypoints,
    )
    t.randomize(seed=int(rng.integers(0, 9999)))

    cl  = t.centerline
    inn = t.inner_boundary
    out = t.outer_boundary


    outer_poly = Polygon(out, closed=True, facecolor="#3a3a50", edgecolor="none", alpha=0.9, zorder=1)
    inner_poly = Polygon(inn, closed=True, facecolor="#141420", edgecolor="none", zorder=2)
    ax.add_patch(outer_poly)
    ax.add_patch(inner_poly)


    ax.plot(np.append(inn[:, 0], inn[0, 0]),
            np.append(inn[:, 1], inn[0, 1]),
            color="#5555aa", lw=0.8, zorder=3)
    ax.plot(np.append(out[:, 0], out[0, 0]),
            np.append(out[:, 1], out[0, 1]),
            color="#5555aa", lw=0.8, zorder=3)


    step = max(1, len(cl) // 60)
    for i in range(0, len(cl) - step, step * 2):
        ax.plot(cl[i:i+step, 0], cl[i:i+step, 1],
                color="#c8c820", lw=0.6, alpha=0.7, zorder=4)


    sx, sy = cl[0]
    ax.plot(sx, sy, "o", color="#40e090", markersize=4, zorder=5)


    ax.set_title(
        f"#{idx+1}  ·  {t.total_length:.0f} px",
        color="#a0a0c0", fontsize=7.5, pad=2,
    )


    margin = 40
    ax.set_xlim(0, cfg.screen_width)
    ax.set_ylim(0, cfg.screen_height)
    ax.invert_yaxis()

plt.tight_layout(rect=[0, 0, 1, 0.97])

import os
os.makedirs("checkpoints", exist_ok=True)
plt.savefig(SAVE_PATH, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved -> {SAVE_PATH}")
