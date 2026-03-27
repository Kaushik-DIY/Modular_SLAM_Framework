import numpy as np
import matplotlib.pyplot as plt

# ---------- Load data ----------
traj = np.loadtxt("outputs/trajectory_fr079.txt")
ts   = np.loadtxt("outputs/timestamps_fr079.txt")

# If you saved odometry separately
odom = np.loadtxt("outputs/odom_fr079.txt")

# ---------- Plot SLAM vs Odom ----------
plt.figure(figsize=(6,6))
plt.plot(odom[:,0], odom[:,1], 'k--', label="Odometry")
plt.plot(traj[:,0], traj[:,1], 'r', label="Hector SLAM")
plt.axis("equal")
plt.legend()
plt.title("FR079: SLAM vs Odometry")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.tight_layout()
plt.savefig("outputs/slam_vs_odom_fr079.png", dpi=200)
plt.close()
