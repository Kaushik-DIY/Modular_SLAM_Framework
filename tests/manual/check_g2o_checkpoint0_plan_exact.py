import faulthandler
faulthandler.enable()

import g2o
import numpy as np


def make_iso(x=0.0, y=0.0, z=0.0):
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return g2o.Isometry3d(T)


# Create a trivial SE3 pose graph: 3 nodes in a line, 1 loop closure
optimizer = g2o.SparseOptimizer()

# Use Cholmod first because this is what your plan specifies.
solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
optimizer.set_algorithm(algorithm)

# Add 3 vertices
for i in range(3):
    v = g2o.VertexSE3()
    v.set_id(i)
    v.set_estimate(make_iso(float(i), 0.0, 0.0))
    if i == 0:
        v.set_fixed(True)
    optimizer.add_vertex(v)

# Add odometry edges: 0 -> 1 and 1 -> 2
for i in range(2):
    e = g2o.EdgeSE3()
    e.set_vertex(0, optimizer.vertex(i))
    e.set_vertex(1, optimizer.vertex(i + 1))
    e.set_measurement(make_iso(1.0, 0.0, 0.0))
    e.set_information(np.eye(6, dtype=np.float64) * 100.0)
    optimizer.add_edge(e)

# Add loop closure edge: 2 -> 0 with slight error
e_loop = g2o.EdgeSE3()
e_loop.set_vertex(0, optimizer.vertex(2))
e_loop.set_vertex(1, optimizer.vertex(0))
e_loop.set_measurement(make_iso(-2.05, 0.0, 0.0))
e_loop.set_information(np.eye(6, dtype=np.float64) * 50.0)
optimizer.add_edge(e_loop)

optimizer.initialize_optimization()
optimizer.optimize(20)

pose2 = optimizer.vertex(2).estimate().matrix()

print("Node 2 optimized pose:")
print(pose2)

print(
    f"Node 2 position: "
    f"x={pose2[0, 3]:.4f}, "
    f"y={pose2[1, 3]:.4f}, "
    f"z={pose2[2, 3]:.4f}"
)

assert abs(pose2[0, 3] - 2.0) < 0.05, (
    "g2o SE3 optimization not working correctly"
)

print("CHECKPOINT 0 PASSED: g2o SE3 PGO working correctly")

