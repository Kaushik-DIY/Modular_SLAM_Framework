# viz/live_view.py
import matplotlib.pyplot as plt
import numpy as np
import os


class LiveView:
    def __init__(self, out_dir="outputs"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.frame = 0

        self.fig, self.ax = plt.subplots()
        self.img = None
        self.traj_line, = self.ax.plot([], [], linewidth=1)

    def update(self, gridmap, trajectory):
        p = gridmap.prob()

        if self.img is None:
            self.img = self.ax.imshow(p, origin="lower", cmap="gray")
        else:
            self.img.set_data(p)

        if len(trajectory) > 1:
            traj = np.array(trajectory)
            gxy = gridmap.world_to_grid(traj[:, :2])
            self.traj_line.set_data(gxy[:, 0], gxy[:, 1])

        self.ax.set_title("Hector SLAM (map + trajectory)")
        self.fig.tight_layout()

        # Save every update
        out_path = os.path.join(self.out_dir, f"frame_{self.frame:04d}.png")
        self.fig.savefig(out_path, dpi=150)
        self.frame += 1
