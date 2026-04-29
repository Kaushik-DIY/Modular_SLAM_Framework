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

stage("1 create SparseOptimizer")
optimizer = g2o.SparseOptimizer()
print("SparseOptimizer OK", flush=True)

stage("2 create LinearSolverDenseSE3")
linear_solver = g2o.LinearSolverDenseSE3()
print("LinearSolverDenseSE3 OK", flush=True)

stage("3 create BlockSolverSE3")
block_solver = g2o.BlockSolverSE3(linear_solver)
print("BlockSolverSE3 OK", flush=True)

stage("4 create OptimizationAlgorithmLevenberg")
algorithm = g2o.OptimizationAlgorithmLevenberg(block_solver)
print("OptimizationAlgorithmLevenberg OK", flush=True)

stage("5 set optimizer algorithm")
optimizer.set_algorithm(algorithm)
print("set_algorithm OK", flush=True)

stage("6 create Isometry3d")
T = np.eye(4)
T[0, 3] = 1.0
iso = g2o.Isometry3d(T)
print("Isometry3d OK", flush=True)

stage("7 create VertexSE3")
v = g2o.VertexSE3()
v.set_id(0)
v.set_estimate(iso)
v.set_fixed(True)
optimizer.add_vertex(v)
print("VertexSE3 add OK", flush=True)

stage("8 create EdgeSE3")
e = g2o.EdgeSE3()
print("EdgeSE3 OK", flush=True)

print("\nCHECKPOINT 0B-DIAG PASSED: core runtime objects work.", flush=True)
