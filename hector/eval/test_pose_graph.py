import numpy as np
from slam.pose_graph import PoseGraph2D, wrap_angle

def info(sig_xy=0.05, sig_th=np.deg2rad(2.0)):
    return np.diag([1.0/sig_xy**2, 1.0/sig_xy**2, 1.0/sig_th**2])

g = PoseGraph2D()
g.add_node(np.array([0.0, 0.0, 0.0]))
g.add_node(np.array([1.0, 0.0, 0.0]))
g.add_node(np.array([1.0, 1.0, np.pi/2]))
g.add_node(np.array([0.0, 1.0, np.pi]))

# chain edges
g.add_edge(0, 1, np.array([1,0,0]), info())
g.add_edge(1, 2, np.array([0,1,np.pi/2]), info())
g.add_edge(2, 3, np.array([1,0,np.pi/2]), info())  # intentionally a bit inconsistent
# loop closure edge (should force square)
g.add_edge(3, 0, np.array([0,1,np.pi]), info(sig_xy=0.02, sig_th=np.deg2rad(1)))

print("Before:", [n.copy() for n in g.nodes])
g.optimize(iters=15)
print("After :", [n.copy() for n in g.nodes])
