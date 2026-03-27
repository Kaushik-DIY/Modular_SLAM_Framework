from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


ResidualJacobianFn = Callable[[np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]


@dataclass
class GNLMConfig:
    iters: int = 10
    damping: float = 1e-3          # LM lambda
    eps_stop: float = 1e-6
    step_clip: Optional[np.ndarray] = None  # shape (n,) or None
    verbose: bool = False


class GaussNewtonLM:
    """
    Generic Gauss-Newton / Levenberg-Marquardt solver.
    Only needs r(x), J(x).
    """

    def __init__(self, cfg: GNLMConfig):
        self.cfg = cfg

    def solve(self, x0: np.ndarray, compute_r_J: ResidualJacobianFn) -> np.ndarray:
        x = np.array(x0, dtype=float).reshape(-1)
        n = x.shape[0]
        lam = float(self.cfg.damping)

        for k in range(int(self.cfg.iters)):
            r, J = compute_r_J(x)
            if r is None or J is None:
                if self.cfg.verbose:
                    print(f"[GNLM] stop: r/J None at iter {k}")
                break

            r = np.asarray(r, dtype=float).reshape(-1, 1)
            J = np.asarray(J, dtype=float)
            if J.ndim != 2 or J.shape[1] != n:
                raise ValueError(f"Jacobian must be (m,{n}), got {J.shape}")

            # Normal equations with LM damping
            H = (J.T @ J) + lam * np.eye(n)
            g = (J.T @ r).reshape(n)

            try:
                dx = -np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                if self.cfg.verbose:
                    print(f"[GNLM] stop: singular H at iter {k}")
                break

            if self.cfg.step_clip is not None:
                clip = np.asarray(self.cfg.step_clip, dtype=float).reshape(n)
                dx = np.clip(dx, -clip, clip)

            x = x + dx

            if self.cfg.verbose:
                print(f"[GNLM] iter {k}: |dx|={float(np.linalg.norm(dx)):.3e}")

            if float(np.linalg.norm(dx)) < float(self.cfg.eps_stop):
                break

        return x