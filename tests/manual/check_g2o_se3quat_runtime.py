import sys
import faulthandler
faulthandler.enable()

import numpy as np
import g2o


def stage(name):
    print(f"\n--- {name} ---", flush=True)


stage("0 import")
print("Python:", sys.executable, flush=True)
print("g2o:", g2o.__file__, flush=True)

stage("1 inspect SE3Quat")
print("Has SE3Quat:", hasattr(g2o, "SE3Quat"), flush=True)
print("SE3Quat dir sample:", [x for x in dir(g2o.SE3Quat) if not x.startswith("_")][:60], flush=True)

stage("2 create default SE3Quat")
se3_default = g2o.SE3Quat()
print("Default SE3Quat OK:", se3_default, flush=True)

stage("3 create SE3Quat from R,t")
R = np.eye(3, dtype=np.float64)
t = np.array([1.0, 0.0, 0.0], dtype=np.float64)
se3_rt = g2o.SE3Quat(R, t)
print("SE3Quat(R,t) OK:", se3_rt, flush=True)

stage("4 create optimizer")
optimizer = g2o.SparseOptimizer()
linear_solver = g2o.LinearSolverDenseSE3()
block_solver = g2o.BlockSolverSE3(linear_solver)
algorithm = g2o.OptimizationAlgorithmLevenberg(block_solver)
optimizer.set_algorithm(algorithm)
print("Optimizer OK", flush=True)

stage("5 create VertexSE3Expmap")
v = g2o.VertexSE3Expmap()
v.set_id(0)
v.set_estimate(se3_default)
v.set_fixed(True)
optimizer.add_vertex(v)
print("VertexSE3Expmap add OK", flush=True)

stage("6 create EdgeSE3Expmap")
e = g2o.EdgeSE3Expmap()
print("EdgeSE3Expmap OK:", e, flush=True)

print("\nCHECKPOINT 0C PASSED: SE3Quat / VertexSE3Expmap runtime path works.", flush=True)

