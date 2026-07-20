"""Semantic mapping core for fusion v2 (offline v1, live-ready API).

Per-keyframe pipeline: segment RGB (SegFormer-B0 ADE20K via onnxruntime CPU)
-> back-project labeled pixels through aligned depth (same math as
extract_orb_rgbd) -> camera->base via BASE_T_CAM (+ one fixed tilt correction
fitted from floor-labeled points) -> base->map via the keyframe's ANCHORED
SE(2) pose -> accumulate per-class votes in a grid that shares map_meta.json
georeferencing exactly.

The class LABEL decides floor vs obstacle; base-frame z-bands are only
backstops against gross misprojections (camera mount tilt is a known gotcha).

Offline consumer: tools/semantic_map_video.py. Future live hook (v2):
run_realtime._extract_visual / IngestEngine.ingest, where rgb/depth/K are in
scope -- backend.segment(rgb) + grid.add_keyframe(...) is the whole call.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from slam_core.fusion2.Dependencies.visual_features import BASE_T_CAM

# ---------------------------------------------------------------------------
# Class groups: ADE20K's 150 labels collapse to a small thesis-relevant set.
# gid = index into GROUPS; 255 marks "ignore" pixels (never voted -- ceiling
# and other above-plane fixtures would project as phantom obstacles).
# ---------------------------------------------------------------------------
GROUPS = ["floor", "wall", "door", "chair", "table",
          "person", "screen", "storage", "other"]
FLOOR_GID = 0
PERSON_GID = GROUPS.index("person")
OTHER_GID = GROUPS.index("other")
IGNORE_GID = 255
# wall-like classes must coincide with LiDAR structure (they live in the scan
# plane); furniture may legitimately extend into LiDAR-free space (tabletops,
# chair seats) so it is not gated.
WALL_LIKE_GIDS = (GROUPS.index("wall"), GROUPS.index("storage"),
                  GROUPS.index("door"))

GROUP_MEMBERS = {
    # ADE20K labels indoor concrete/industrial floor as road/sidewalk/earth
    # (verified on the lab entrance corridor: "road" 100%) -- treat all
    # walkable ground classes as floor.
    "floor":   ["floor", "rug", "mat", "carpet", "road", "sidewalk", "path",
                "earth", "land", "dirt track"],
    "wall":    ["wall", "windowpane", "curtain", "mirror", "column", "railing"],
    "door":    ["door", "screen door"],
    "chair":   ["chair", "armchair", "swivel chair", "seat", "sofa", "stool", "ottoman"],
    "table":   ["table", "desk", "coffee table", "counter", "countertop", "kitchen island"],
    "person":  ["person"],
    "screen":  ["computer", "monitor", "crt screen", "screen", "television receiver"],
    # wheeled lab carts/transport read as truck/van and container objects as
    # tank/ashcan/box on ADE20K -- group them with storage so lab equipment
    # renders as a distinct obstacle class, not "other".
    "storage": ["cabinet", "shelf", "wardrobe", "bookcase", "chest of drawers",
                "refrigerator", "case", "box", "truck", "van", "hovel",
                "tank", "ashcan", "barrel", "basket"],
    # everything unmatched falls into "other" via the LUT default
}
IGNORE_LABELS = {"ceiling", "sky", "light", "lamp", "chandelier",
                 "painting", "signboard", "poster", "sconce", "fan",
                 "traffic light", "awning", "flag"}

PALETTE_BGR = {                       # BGR for cv2 drawing
    "floor":   (168, 211, 141),      # light green
    "wall":    (140, 107, 91),       # slate blue
    "door":    (110, 200, 230),      # amber
    "chair":   (95, 95, 217),        # red
    "table":   (198, 124, 176),      # purple
    "person":  (85, 153, 255),       # orange
    "screen":  (196, 184, 77),       # cyan
    "storage": (92, 122, 138),       # brown
    "other":   (158, 158, 158),      # gray
}

# z-band backstops in the (tilt-corrected) base frame, used when no floor
# height has been fitted. With a fit, bands re-center on the fitted height.
FLOOR_Z_BAND = (-0.4, 0.3)
OBST_Z_BAND = (-0.2, 2.0)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class SegmenterBackend:
    """Lazy onnxruntime wrapper around SegFormer-B0 ADE20K (150 classes).

    segment() returns an (H,W) uint8 group-id map in the ORIGINAL image size
    (nearest-neighbour upsample of the 128x128 argmax), 255 = ignore.
    """

    def __init__(self, model_dir: Path, input_size: int = 512):
        self.model_dir = Path(model_dir)
        self.input_size = int(input_size)
        self._sess = None
        self._lut = None

    def _ensure(self):
        if self._sess is not None:
            return
        import onnxruntime as ort
        self._sess = ort.InferenceSession(str(self.model_dir / "model.onnx"),
                                          providers=["CPUExecutionProvider"])
        id2label = json.load(open(self.model_dir / "config.json"))["id2label"]
        n = len(id2label)
        lut = np.full(n, OTHER_GID, dtype=np.uint8)
        name_to_id = {v.strip().lower(): int(k) for k, v in id2label.items()}
        for gid, group in enumerate(GROUPS):
            for name in GROUP_MEMBERS.get(group, []):
                cid = name_to_id.get(name)
                if cid is None:
                    print(f"[semantics] warn: label {name!r} not in id2label")
                else:
                    lut[cid] = gid
        for name in IGNORE_LABELS:
            cid = name_to_id.get(name)
            if cid is not None:
                lut[cid] = IGNORE_GID
        self._lut = lut

    @property
    def lut(self) -> np.ndarray:
        """150-entry raw-ADE20K-id -> group-id table (255 = ignore)."""
        self._ensure()
        return self._lut

    def segment_raw(self, rgb_bgr: np.ndarray) -> np.ndarray:
        """(H,W) uint8 RAW ADE20K class ids (cache-friendly: group remapping
        can be re-applied later without re-running the model)."""
        self._ensure()
        h, w = rgb_bgr.shape[:2]
        s = self.input_size
        img = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
        x = (img.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])       # 1,3,S,S
        logits = self._sess.run(None, {"pixel_values": x})[0][0]   # 150,S/4,S/4
        cls = np.argmax(logits, axis=0).astype(np.uint8)
        return cv2.resize(cls, (w, h), interpolation=cv2.INTER_NEAREST)

    def segment(self, rgb_bgr: np.ndarray) -> np.ndarray:
        return self.lut[self.segment_raw(rgb_bgr)]

    def colorize(self, gmap: np.ndarray) -> np.ndarray:
        """Group map -> BGR color image (ignore = black)."""
        out = np.zeros((*gmap.shape, 3), np.uint8)
        for gid, group in enumerate(GROUPS):
            out[gmap == gid] = PALETTE_BGR[group]
        return out


def backproject_labeled(gmap: np.ndarray, depth: np.ndarray, K: np.ndarray,
                        stride: int = 4, max_depth: float = 4.0,
                        depth_factor: float = 1000.0
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Strided labeled pixels -> (pts_cam (N,3) optical frame, gids (N,)).

    Same back-projection as extract_orb_rgbd; drops ignore pixels and
    invalid/far depth.
    """
    h, w = depth.shape[:2]
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    us, vs = us.ravel(), vs.ravel()
    g = gmap[vs, us]
    z = depth[vs, us].astype(np.float32) / float(depth_factor)
    keep = (g != IGNORE_GID) & (z > 0.05) & (z <= max_depth)
    us, vs, z, g = us[keep], vs[keep], z[keep], g[keep]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts = np.empty((len(z), 3), np.float32)
    pts[:, 0] = (us - cx) / fx * z
    pts[:, 1] = (vs - cy) / fy * z
    pts[:, 2] = z
    return pts, g


def cam_to_base(pts_cam: np.ndarray, tilt_R: Optional[np.ndarray] = None
                ) -> np.ndarray:
    """Optical -> base (REP-103) via BASE_T_CAM; optional fixed tilt fix.

    Rotation-only, matching how the rest of the pipeline treats the camera
    extrinsic (the camera/base lever arm is ~1-2 map cells and ignored).
    """
    R = BASE_T_CAM[:3, :3]
    if tilt_R is not None:
        R = tilt_R @ R
    return pts_cam @ R.T.astype(np.float32)


def estimate_tilt_from_floor(points_base: np.ndarray, n_iters: int = 200,
                             thresh: float = 0.03, seed: int = 0):
    """RANSAC-fit a plane to floor-LABELED base-frame points; return
    (R_tilt 3x3, z_floor, inlier_ratio). R_tilt levels the fitted plane so
    its normal becomes +z. Falls back to (I, None, 0.0) on too few points."""
    P = np.asarray(points_base, np.float64)
    if len(P) < 500:
        return np.eye(3), None, 0.0
    rng = np.random.default_rng(seed)
    best_inl, best_n, best_d = 0, None, 0.0
    for _ in range(n_iters):
        i = rng.choice(len(P), 3, replace=False)
        a, b, c = P[i]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        if n[2] < 0:
            n = -n
        if n[2] < 0.7:          # reject walls masquerading as floor planes
            continue
        d = -float(n @ a)
        inl = int(np.sum(np.abs(P @ n + d) < thresh))
        if inl > best_inl:
            best_inl, best_n, best_d = inl, n, d
    if best_n is None:
        return np.eye(3), None, 0.0
    # refine on inliers via least squares (SVD of centered inliers)
    m = np.abs(P @ best_n + best_d) < thresh
    Q = P[m]
    ctr = Q.mean(axis=0)
    _, _, vt = np.linalg.svd(Q - ctr, full_matrices=False)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    # rotation taking fitted normal -> +z (Rodrigues about n x z)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s, c = np.linalg.norm(v), float(n @ z)
    if s < 1e-9:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    z_floor = float((R @ ctr)[2])
    return R, z_floor, best_inl / len(P)


class SemanticVoteGrid:
    """Per-class vote accumulator sharing map_meta.json georeferencing.

    votes[gid, iy, ix] counts how often map cell (ix,iy) was observed as
    class gid across keyframes. Final raster: obstacle-priority argmax.
    """

    def __init__(self, meta: dict, n_groups: int = len(GROUPS)):
        self.meta = dict(meta)
        self.h, self.w = int(meta["height"]), int(meta["width"])
        self.ox, self.oy = float(meta["origin_x"]), float(meta["origin_y"])
        self.res = float(meta["resolution"])
        self.votes = np.zeros((n_groups, self.h, self.w), np.uint16)
        self.z_floor: Optional[float] = None   # fitted floor height (base z)

    def _z_bands(self):
        if self.z_floor is None:
            return FLOOR_Z_BAND, OBST_Z_BAND
        zf = self.z_floor
        return (zf - 0.25, zf + 0.15), (zf + 0.05, zf + 2.0)

    def add_points(self, pts_base: np.ndarray, gids: np.ndarray,
                   pose_xyt, obstacle_max_depth: float = 3.0) -> dict:
        """Splat tilt-corrected base-frame labeled points at an SE(2) pose.

        Reliability weighting (dominant fix for far-range wall smear): votes
        count 3x under 1.5 m, 2x under 2.5 m, 1x beyond; obstacle classes are
        additionally CAPPED at ``obstacle_max_depth`` (floor may use the full
        depth range -- it is validated against free space anyway). Person is
        a dynamic object and never votes into the (static) map.
        """
        (f_lo, f_hi), (o_lo, o_hi) = self._z_bands()
        z = pts_base[:, 2]
        rng = np.linalg.norm(pts_base[:, :2], axis=1)
        is_floor = gids == FLOOR_GID
        keep = np.where(is_floor, (z >= f_lo) & (z <= f_hi),
                        (z >= o_lo) & (z <= o_hi)
                        & (rng <= obstacle_max_depth))
        keep &= gids != PERSON_GID
        pts, g, rng = pts_base[keep], gids[keep], rng[keep]
        x, y, th = float(pose_xyt[0]), float(pose_xyt[1]), float(pose_xyt[2])
        c, s = math.cos(th), math.sin(th)
        wx = c * pts[:, 0] - s * pts[:, 1] + x
        wy = s * pts[:, 0] + c * pts[:, 1] + y
        ix = np.floor((wx - self.ox) / self.res).astype(np.int64)
        iy = np.floor((wy - self.oy) / self.res).astype(np.int64)
        inb = (ix >= 0) & (ix < self.w) & (iy >= 0) & (iy < self.h)
        ix, iy, g, rng = ix[inb], iy[inb], g[inb], rng[inb]
        w = np.where(rng < 1.5, 3, np.where(rng < 2.5, 2, 1)).astype(np.uint16)
        np.add.at(self.votes, (g.astype(np.int64), iy, ix), w)
        # uint16 saturation guard (np.add.at wraps): clamp cells near the top
        hot = self.votes >= 60000
        if hot.any():
            self.votes[hot] = 60000
        return dict(n_samples=len(gids), n_kept=int(keep.sum()),
                    n_in_map=int(inb.sum()))

    def add_keyframe(self, gmap, depth, K, pose_xyt, *, stride=4,
                     max_depth=4.0, depth_factor=1000.0,
                     tilt_R: Optional[np.ndarray] = None,
                     obstacle_max_depth: float = 3.0) -> dict:
        """Convenience: backproject + cam_to_base + add_points."""
        pts_cam, gids = backproject_labeled(gmap, depth, K, stride=stride,
                                            max_depth=max_depth,
                                            depth_factor=depth_factor)
        return self.add_points(cam_to_base(pts_cam, tilt_R), gids, pose_xyt,
                               obstacle_max_depth=obstacle_max_depth)

    def argmax_raster(self, min_votes: int = 8, floor_min_votes: int = 3,
                      dominance: float = 0.35,
                      occ_prob: Optional[np.ndarray] = None) -> np.ndarray:
        """(H,W) int16 raster of gids, -1 = unknown.

        Floor is placed wherever it has >= floor_min_votes (a LOW bar: floor
        is separately validated against LiDAR free space, so false floor is
        cheap to accept and holes are expensive). The dominant obstacle class
        overrides a cell when it has >= min_votes AND >= dominance * floor
        votes there. Rationale for the fraction: grazing rays passing UNDER
        furniture give floor votes to the same cell the furniture occupies,
        so requiring obstacle >= floor (dominance 1.0) erased every chair;
        requiring only min_votes (no dominance) let far-range wall smear
        flood the corridor. 0.35 keeps both failure modes out.

        ``occ_prob`` enables the CROSS-MODAL check at the EVIDENCE level:
        wall-like classes (wall/door/storage) physically live in the LiDAR
        scan plane, so their votes are zeroed at cells farther than ~0.15 m
        from LiDAR-occupied structure BEFORE the decision. The cell is then
        resolved from the remaining genuine evidence (usually floor, often a
        chair that had been buried under wall smear) instead of being deleted
        outright -- gating decisions (label deletion) left unlabeled holes;
        gating evidence keeps the map complete AND geometrically plausible.
        Floor is deliberately NOT gated: a 2-D LiDAR marks the object ABOVE
        the floor, not the floor itself, so the camera is the authority on
        the floor surface (e.g. under tables)."""
        votes = self.votes
        if occ_prob is not None:
            near = cv2.dilate(occupied_mask(occ_prob).astype(np.uint8),
                              np.ones((7, 7), np.uint8)) > 0      # ~0.15 m
            votes = votes.copy()
            for g in WALL_LIKE_GIDS:
                votes[g][~near] = 0
        floor_v = votes[FLOOR_GID]
        obst = votes[1:]                            # all non-floor groups
        best = np.argmax(obst, axis=0).astype(np.int16) + 1
        best_v = np.take_along_axis(obst, (best - 1)[None].astype(np.int64),
                                    axis=0)[0]
        out = np.full((self.h, self.w), -1, np.int16)
        out[floor_v >= floor_min_votes] = FLOOR_GID
        obst_ok = (best_v >= min_votes) & (best_v >= dominance * floor_v)
        out[obst_ok] = best[obst_ok]
        return out

    def save(self, npz_path):
        np.savez_compressed(npz_path, votes=self.votes,
                            groups=np.array(GROUPS),
                            meta=json.dumps(self.meta),
                            z_floor=-1e9 if self.z_floor is None else self.z_floor)

    @classmethod
    def load(cls, npz_path) -> "SemanticVoteGrid":
        d = np.load(npz_path, allow_pickle=False)
        grid = cls(json.loads(str(d["meta"])))
        grid.votes = d["votes"]
        zf = float(d["z_floor"])
        grid.z_floor = None if zf < -1e8 else zf
        return grid


def clean_raster(raster: np.ndarray, min_region: int = 6) -> np.ndarray:
    """Publication-grade cleanup of the argmax raster (int16, -1 unknown).

    (The cross-modal LiDAR check happens at the EVIDENCE level inside
    ``SemanticVoteGrid.argmax_raster(occ_prob=...)`` -- gating raster labels
    here just left unlabeled holes where the runner-up class had genuine
    votes.)

    1. 5x5 majority smoothing over labeled cells (fills pinholes, kills
       salt-and-pepper single-cell noise without moving class borders).
    2. Small-component removal: connected regions below ``min_region`` cells
       are unlabeled (the median raw fragment is 2 cells -- pure speckle).
    """
    out = raster.copy()

    # majority filter: per-class box counts, argmax where enough support
    n = len(GROUPS)
    counts = np.stack([cv2.boxFilter((out == g).astype(np.float32), -1, (5, 5),
                                     normalize=False) for g in range(n)])
    best = np.argmax(counts, axis=0).astype(np.int16)
    best_c = np.take_along_axis(counts, best[None].astype(np.int64), 0)[0]
    labeled = out >= 0
    smoothed = np.where(best_c >= 5, best, out)   # >=5 of 25 neighbours agree
    out = np.where(labeled, smoothed, -1).astype(np.int16)

    for g in range(n):
        m = (out == g).astype(np.uint8)
        if not m.any():
            continue
        nc, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        small = np.flatnonzero(stats[1:, 4] < min_region) + 1
        if len(small):
            out[np.isin(lab, small)] = -1
    return out


def interp_pose(traj: np.ndarray, t: float) -> Tuple[float, float, float]:
    """Linear xy + shortest-arc theta interpolation of an anchored keyframe
    trajectory ``traj`` = (N,4) [t,x,y,theta] at dataset time ``t``. Lets the
    dense (sub-keyframe) segmentation cadence project at accurate poses."""
    ts = traj[:, 0]
    i = int(np.searchsorted(ts, t))
    if i <= 0:
        r = traj[0]
        return float(r[1]), float(r[2]), float(r[3])
    if i >= len(traj):
        r = traj[-1]
        return float(r[1]), float(r[2]), float(r[3])
    a, b = traj[i - 1], traj[i]
    f = (t - a[0]) / max(b[0] - a[0], 1e-9)
    dth = math.atan2(math.sin(b[3] - a[3]), math.cos(b[3] - a[3]))
    return (float(a[1] + f * (b[1] - a[1])),
            float(a[2] + f * (b[2] - a[2])),
            float(a[3] + f * dth))


def raster_to_bgr(raster: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Semantic raster -> (BGR image, bool mask of known cells)."""
    known = raster >= 0
    img = np.zeros((*raster.shape, 3), np.uint8)
    for gid, group in enumerate(GROUPS):
        img[raster == gid] = PALETTE_BGR[group]
    return img, known


def occupied_mask(occ_prob: np.ndarray, min_region: int = 8) -> np.ndarray:
    """Despeckled occupied-cell mask (drops isolated LiDAR ray/noise cells)."""
    occ = (occ_prob > 0.65).astype(np.uint8)
    nc, lab, stats, _ = cv2.connectedComponentsWithStats(occ, 8)
    small = np.flatnonzero(stats[1:, 4] < min_region) + 1
    if len(small):
        occ[np.isin(lab, small)] = 0
    return occ > 0


def content_bbox(raster: np.ndarray, occ: np.ndarray, margin_cells: int = 14):
    """(row0, row1, col0, col1) crop around semantic + occupied content."""
    m = (raster >= 0) | occ
    rows = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    if not len(rows):
        return 0, raster.shape[0], 0, raster.shape[1]
    return (max(0, rows[0] - margin_cells),
            min(raster.shape[0], rows[-1] + 1 + margin_cells),
            max(0, cols[0] - margin_cells),
            min(raster.shape[1], cols[-1] + 1 + margin_cells))


def render_semantic_png(raster: np.ndarray, occupancy_prob, meta: dict,
                        out_png, title: str, traj_xyt=None):
    """Publication-style semantic floor plan: white background, despeckled
    black LiDAR walls, solid class colors, trajectory, legend; cropped to
    the mapped content (no ray fans, no gray void)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    occ = (occupied_mask(occupancy_prob) if occupancy_prob is not None
           else np.zeros(raster.shape, bool))
    r0, r1, c0, c1 = content_bbox(raster, occ)
    res, ox, oy = float(meta["resolution"]), float(meta["origin_x"]), float(meta["origin_y"])
    extent = [ox + c0 * res, ox + c1 * res, oy + r0 * res, oy + r1 * res]

    img = np.ones((r1 - r0, c1 - c0, 3), np.float32)          # white base
    sub = raster[r0:r1, c0:c1]
    for gid, group in enumerate(GROUPS):
        b, g, r = PALETTE_BGR[group]
        img[sub == gid] = (r / 255.0, g / 255.0, b / 255.0)
    img[occ[r0:r1, c0:c1]] = (0.12, 0.12, 0.12)               # walls on top

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.imshow(img, origin="lower", extent=extent, interpolation="nearest")
    if traj_xyt is not None and len(traj_xyt):
        t = np.asarray(traj_xyt, float)
        ax.plot(t[:, 0], t[:, 1], "-", lw=1.3, color="tab:blue", alpha=0.95)
        ax.scatter(t[0, 0], t[0, 1], c="g", s=55, zorder=5, label="start")
        ax.scatter(t[-1, 0], t[-1, 1], c="r", s=55, zorder=5, label="end")
    handles = [Patch(facecolor=(r / 255, g / 255, b / 255), edgecolor="0.6",
                     label=name)
               for name, (b, g, r) in ((n, PALETTE_BGR[n]) for n in GROUPS
                                       if n != "person")]
    handles.append(Patch(facecolor=(0.12, 0.12, 0.12), label="LiDAR wall"))
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5),
              fontsize=9, frameon=False)
    ax.set_title(title)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
