import faulthandler
faulthandler.enable()

import numpy as np
import g2o


def make_se3(x, y, z):
    R = np.eye(3, dtype=np.float64)
    t = np.array([[x], [y], [z]], dtype=np.float64)
    return g2o.SE3Quat(R, t)


optimizer = g2o.SparseOptimizer()
solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
optimizer.set_algorithm(algorithm)

# Vertices: 0, 1, 2 along x-axis.
for i in range(3):
    v = g2o.VertexSE3Expmap()
    v.set_id(i)
    v.set_estimate(make_se3(float(i), 0.0, 0.0))
    if i == 0:
        v.set_fixed(True)
    optimizer.add_vertex(v)

# Edge 0 -> 1: +1m x
e01 = g2o.EdgeSE3Expmap()
e01.set_vertex(0, optimizer.vertex(0))
e01.set_vertex(1, optimizer.vertex(1))
e01.set_measurement(make_se3(1.0, 0.0, 0.0))
e01.set_information(np.eye(6, dtype=np.float64) * 100.0)
optimizer.add_edge(e01)

# Edge 1 -> 2: +1m x
e12 = g2o.EdgeSE3Expmap()
e12.set_vertex(0, optimizer.vertex(1))
e12.set_vertex(1, optimizer.vertex(2))
e12.set_measurement(make_se3(1.0, 0.0, 0.0))
e12.set_information(np.eye(6, dtype=np.float64) * 100.0)
optimizer.add_edge(e12)

optimizer.initialize_optimization()
optimizer.optimize(20)

est = optimizer.vertex(2).estimate()
T = est.to_homogeneous_matrix()

print("Vertex 2 estimate:")
print(T)
print("x:", T[0, 3])

assert abs(T[0, 3] - 2.0) < 1e-6

print("CHECKPOINT 0B PASSED: SE3Expmap PGO graph optimization works.")
