import numpy as np
import matplotlib.pyplot as plt

def plot_map_and_traj(gridmap, trajectory, out_path="outputs/final_map_traj_fr079.png", title=None):
    p = gridmap.prob()
    traj = np.array(trajectory)

    gxy = gridmap.world_to_grid(traj[:, :2])

    plt.figure()
    plt.imshow(p, origin="lower", cmap="gray")
    plt.plot(gxy[:, 0], gxy[:, 1], linewidth=1)
    H, W = p.shape
    plt.xlim(0, W - 1)
    plt.ylim(0, H - 1)
    if title is None:
        title = "Hector SLAM: final map + trajectory"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    print("saved", out_path)