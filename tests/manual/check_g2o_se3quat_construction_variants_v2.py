import subprocess
import sys
import textwrap


CASES = {
    "default_only": r"""
import g2o
se3 = g2o.SE3Quat()
print("OK default", se3)
print("matrix:", se3.to_homogeneous_matrix())
""",

    "default_set_translation_col": r"""
import numpy as np
import g2o
se3 = g2o.SE3Quat()
t = np.array([[1.0], [0.0], [0.0]], dtype=np.float64)
se3.set_translation(t)
print("OK set_translation column")
print("translation:", se3.translation())
print("matrix:", se3.to_homogeneous_matrix())
""",

    "constructor_R_t_col": r"""
import numpy as np
import g2o
R = np.eye(3, dtype=np.float64)
t = np.array([[1.0], [0.0], [0.0]], dtype=np.float64)
se3 = g2o.SE3Quat(R, t)
print("OK SE3Quat(R,t column)")
print("matrix:", se3.to_homogeneous_matrix())
""",

    "constructor_v6_col": r"""
import numpy as np
import g2o
# Minimal vector: usually [omega_x, omega_y, omega_z, tx, ty, tz] or internal convention.
v = np.zeros((6, 1), dtype=np.float64)
v[3, 0] = 1.0
se3 = g2o.SE3Quat(v)
print("OK SE3Quat(v6 column)")
print("matrix:", se3.to_homogeneous_matrix())
""",

    "constructor_v7_col_guess": r"""
import numpy as np
import g2o
# g2o SE3Quat vector usually stores translation + quaternion.
# We test a likely order: tx, ty, tz, qx, qy, qz, qw.
v = np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0]], dtype=np.float64)
se3 = g2o.SE3Quat(v)
print("OK SE3Quat(v7 column)")
print("matrix:", se3.to_homogeneous_matrix())
""",

    "instance_from_vector_v7": r"""
import numpy as np
import g2o
v = np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0]], dtype=np.float64)
se3 = g2o.SE3Quat()
se3.from_vector(v)
print("OK instance from_vector(v7)")
print("matrix:", se3.to_homogeneous_matrix())
""",

    "instance_from_minimal_vector_v6": r"""
import numpy as np
import g2o
v = np.zeros((6, 1), dtype=np.float64)
v[3, 0] = 1.0
se3 = g2o.SE3Quat()
se3.from_minimal_vector(v)
print("OK instance from_minimal_vector(v6)")
print("matrix:", se3.to_homogeneous_matrix())
""",

    "quaternion_wxyz": r"""
import g2o
q = g2o.Quaternion(1.0, 0.0, 0.0, 0.0)
print("OK Quaternion(w,x,y,z)", q)
""",

    "constructor_q_t_col": r"""
import numpy as np
import g2o
q = g2o.Quaternion(1.0, 0.0, 0.0, 0.0)
t = np.array([[1.0], [0.0], [0.0]], dtype=np.float64)
se3 = g2o.SE3Quat(q, t)
print("OK SE3Quat(q,t column)")
print("matrix:", se3.to_homogeneous_matrix())
""",
}


for name, code in CASES.items():
    print("\n" + "=" * 80)
    print("CASE:", name)
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        text=True,
        capture_output=True,
    )

    print("returncode:", result.returncode)

    if result.stdout:
        print("--- stdout ---")
        print(result.stdout)

    if result.stderr:
        print("--- stderr ---")
        print(result.stderr)

    if result.returncode == -11:
        print("RESULT: SEGFAULT")
    elif result.returncode == 0:
        print("RESULT: OK")
    else:
        print("RESULT: ERROR")
