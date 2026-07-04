#!/usr/bin/env python3
"""
thesis_switch_video.py
======================
Dual-panel presentation video of the fusion layer's LIVE module switching.

  Left  : robot POV (the RGB camera frame nearest each keyframe)
  Right : occupancy map being built + trajectory (coloured per active config),
          live loop-closure arcs, and numbered switch markers.  The map
          visibly DEFORMS (snaps tighter) at each loop closure, replayed from
          the per-loop graph snapshots dumped by run_realtime.py.

  Bottom: a persistent HUD strip  [ Front-end | Proposer | Verifier ]  that
          tracks the active config; at a switch the changed cell flashes and a
          full-width banner is shown while the frame is held for a moment.

Consumes ONE switching-demo run dir produced by:

    .venv/bin/python -m slam_core.fusion2.run_realtime ... --switch-schedule "..."

Inputs in the run dir: frames.npz (per-keyframe scans + per-loop pose snapshots),
timeline.csv, switches.csv, verifications.csv, map_meta.json.

    .venv/bin/python tools/thesis_switch_video.py \
        --run thesis_outputs/switching_demo/presentation/realtime_orb_YYYYMMDD_HHMMSS \
        --out thesis_outputs/switching_demo/presentation/switching_demo.mp4 \
        --speed 6            # 1 = actual real-time, N = N times faster

The video is driven by the dataset clock (uniform motion), NOT the keyframe index.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from slam_core.matching.scan_to_map import GridMap, _transform_points
from slam_core.fusion2.Dependencies.dataset import LabHybridStream
# small drawing helpers reused verbatim from the Hector map-build video
from hector.eval.render_map_build_video import (
    _put_text, _draw_scale_bar,
    BG_DARK, DIVIDER_COLOR, ROBOT_COLOR, START_COLOR,
    TITLE_COLOR, SUBTITLE_COLOR, LABEL_COLOR,
)

# per-active-config segment colours (same palette as tools/thesis_switch_demo.py)
SEG_COLORS_RGB = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
_FE_SHORT = {"native_s2s": "s2s", "native_s2m": "s2m", "visual_vo": "VO"}
LOOP_COLOR = (60, 240, 255)        # bright yellow (BGR): loop-closure arc
BANNER_BG = (30, 30, 40)
FLASH_A = np.array([40, 220, 255], dtype=np.float32)   # yellow  (BGR)
FLASH_B = np.array([80, 220, 90], dtype=np.float32)    # green   (BGR)


def _hex_bgr(h: str):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


SEG_COLORS = [_hex_bgr(h) for h in SEG_COLORS_RGB]


# ------------------------------------------------------------------ I/O
def _read_csv(path: Path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


class _Data:
    """Everything the renderer needs, resolved to keyframe order."""

    def __init__(self, run_dir: Path, dataset: Path):
        rd = Path(run_dir)
        npz_path = rd / "frames.npz"
        if not npz_path.exists():
            raise SystemExit(f"{npz_path} missing — run_realtime must be run with "
                             f"--switch-schedule (and the frames.npz instrumentation).")
        z = np.load(npz_path, allow_pickle=True)
        self.kf_ids = z["kf_ids"].astype(int)
        self.stamps = z["stamps"].astype(float)
        self.scans = list(z["scans"])
        self.snap_counts = z["snap_kfcounts"].astype(int)
        self.snap_poses = list(z["snap_poses"])
        self.final_poses = z["final_poses"]            # (M,4): nid,x,y,th
        self.extent = z["extent"].astype(float)        # [xmin,xmax,ymin,ymax]
        self.res = float(z["resolution"])
        self.N = len(self.kf_ids)

        self.timeline = _read_csv(rd / "timeline.csv")
        self.switches = _read_csv(rd / "switches.csv")
        self.verifs = _read_csv(rd / "verifications.csv")

        # live (pre-correction) poses per keyframe, keyframe order
        self.live = {int(r["kf"]): (float(r["x"]), float(r["y"]), float(r["theta"]))
                     for r in self.timeline}
        # active modules per keyframe row (parallel to kf_ids)
        self.fe = [r["fe"] for r in self.timeline]
        self.proposer = [r["proposer"] for r in self.timeline]
        self.verifier = [r["verifier"] for r in self.timeline]
        self.sensor = [r["sensor"] for r in self.timeline]

        # switch keyframes -> nearest keyframe-row index (for markers/banners)
        self.sw_kf = [int(s["kf"]) for s in self.switches]
        self.sw_rowidx = [int(np.argmin(np.abs(self.kf_ids - k))) for k in self.sw_kf]

        # accepted loops keyed by the query keyframe id
        self.loops_by_query: dict[int, list[int]] = {}
        for v in self.verifs:
            if str(v.get("accepted", "")).lower() in ("true", "1"):
                q, c = int(v["query"]), int(v["cand"])
                self.loops_by_query.setdefault(q, []).append(c)

        # final pose per node id (for placing switch/loop markers stably)
        self.final = {int(nid): (float(x), float(y), float(th))
                      for nid, x, y, th in self.final_poses}

        # The dumped extent/final_poses are in the START-anchored frame
        # (_anchor_poses: first keyframe at origin facing +x), but the live
        # timeline poses and per-loop snapshots are RAW graph poses. Re-frame
        # the raw poses with the SAME rigid transform (gauge = node 0, which is
        # graph-fixed, so one constant transform) so map + trajectory register
        # with the anchored crop bbox.
        ax, ay, ath = self.live[int(self.kf_ids[0])]
        c, s = np.cos(ath), np.sin(ath)

        def _anchor(x, y, th):
            dx, dy = x - ax, y - ay
            return (c * dx + s * dy, -s * dx + c * dy,
                    float(np.arctan2(np.sin(th - ath), np.cos(th - ath))))

        self.live = {kf: _anchor(*p) for kf, p in self.live.items()}
        anchored = []
        for sp in self.snap_poses:
            a = np.asarray(sp, float).copy()
            for r in range(len(a)):
                a[r, 1], a[r, 2], a[r, 3] = _anchor(a[r, 1], a[r, 2], a[r, 3])
            anchored.append(a)
        self.snap_poses = anchored

        # RGB frames (POV): nearest by stamp, no sync tolerance so POV is never blank
        stream = LabHybridStream(Path(dataset))
        self._rgb = stream.rgbd_entries()               # [(t, rgb_path, depth_path)]
        self._rgb_t = np.array([e[0] for e in self._rgb])

    # -- helpers ------------------------------------------------------
    def seg_index(self, j: int) -> int:
        """Active-config segment number for keyframe row j (0-based)."""
        return int(sum(1 for k in self.sw_kf if self.kf_ids[j] >= k))

    def snap_index(self, upto_count: int) -> int:
        """Index of the latest loop snapshot valid after `upto_count` keyframes."""
        idx = -1
        for si, c in enumerate(self.snap_counts):
            if c <= upto_count:
                idx = si
            else:
                break
        return idx

    def pose_dict(self, snap_idx: int) -> dict:
        """nid -> (x,y,th): snapshot poses where available, live otherwise."""
        if snap_idx < 0:
            return dict(self.live)
        d = dict(self.live)
        for nid, x, y, th in self.snap_poses[snap_idx]:
            d[int(nid)] = (float(x), float(y), float(th))
        return d

    def rgb_frame(self, stamp: float):
        i = int(np.argmin(np.abs(self._rgb_t - stamp)))
        img = cv2.imread(str(self._rgb[i][1]), cv2.IMREAD_COLOR)
        return img


# ------------------------------------------------------------------ map canvas
class _MapCanvas:
    """Fixed world->grid->panel mapping + a vectorized wall-hit occupancy grid
    that is rebuilt (cheaply) whenever a loop snapshot deforms the graph."""

    HIT_SAT = 3.0                      # hits at which a wall cell reaches full white

    def __init__(self, D: _Data, panel_w: int, panel_h: int,
                 header_h: int = 70, footer_h: int = 40):
        self.D = D
        self.pw, self.ph = panel_w, panel_h
        self.header_h, self.footer_h = header_h, footer_h

        xmin, xmax, ymin, ymax = D.extent
        margin = 1.5
        reach = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax)) + margin
        size_m = 2.0 * reach
        self.grid = GridMap(res=D.res, size_m=size_m, l_min=-5.0, l_max=5.0)

        # fixed crop (grid-index box) around the padded trajectory bbox
        pad_px = int(margin / D.res)
        corners = np.array([[xmin, ymin], [xmax, ymax]])
        g = self.grid.world_to_grid(corners)
        gx_lo, gx_hi = int(g[:, 0].min()) - pad_px, int(g[:, 0].max()) + pad_px
        gy_lo, gy_hi = int(g[:, 1].min()) - pad_px, int(g[:, 1].max()) + pad_px
        sz = self.grid.size
        self.ix_lo = max(0, gx_lo); self.ix_hi = min(sz, gx_hi + 1)
        # after flipud, iy = (size-1) - gy
        self.iy_lo = max(0, (sz - 1) - gy_hi); self.iy_hi = min(sz, (sz - 1) - gy_lo + 1)

        self.crop_w = self.ix_hi - self.ix_lo
        self.crop_h = self.iy_hi - self.iy_lo
        area_w = panel_w - 48
        area_h = panel_h - header_h - footer_h
        self.scale = min(area_w / self.crop_w, area_h / self.crop_h)
        self.dw = int(self.crop_w * self.scale); self.dh = int(self.crop_h * self.scale)
        self.off_x = 24 + (area_w - self.dw) // 2
        self.off_y = header_h + (area_h - self.dh) // 2

        # wall-hit accumulator over the (flipped-y) crop window
        self.hits = np.zeros((self.crop_h, self.crop_w), dtype=np.float32)
        self._cur_snap = -2          # force first rebuild
        self._built_upto = -1

    def _integrate(self, j: int, pd: dict):
        kf = int(self.D.kf_ids[j])
        pts = self.D.scans[j]
        if pts is None or len(pts) == 0 or kf not in pd:
            return
        x, y, th = pd[kf]
        pw = _transform_points((x, y, th), np.asarray(pts, float))
        g = self.grid.world_to_grid(pw)
        lx = np.round(g[:, 0]).astype(int) - self.ix_lo
        ly = np.round((self.grid.size - 1) - g[:, 1]).astype(int) - self.iy_lo
        m = (lx >= 0) & (lx < self.crop_w) & (ly >= 0) & (ly < self.crop_h)
        np.add.at(self.hits, (ly[m], lx[m]), 1.0)

    def update(self, i: int):
        """Advance the map to include keyframes 0..i, deforming at snapshots."""
        snap = self.D.snap_index(i + 1)
        pd = self.D.pose_dict(snap)
        if snap != self._cur_snap:
            # graph deformed (or first frame): rebuild from scratch at new poses
            self.hits[:] = 0.0
            for j in range(i + 1):
                self._integrate(j, pd)
            self._cur_snap = snap
        else:
            for j in range(self._built_upto + 1, i + 1):
                self._integrate(j, pd)
        self._built_upto = i
        return pd

    # world (x,y) -> panel pixel
    def w2p(self, x: float, y: float):
        g = self.grid.world_to_grid(np.array([[x, y]]))[0]
        ix, iy = g[0], (self.grid.size - 1) - g[1]      # flipud
        px = int((ix - self.ix_lo) * self.scale) + self.off_x
        py = int((iy - self.iy_lo) * self.scale) + self.off_y
        return px, py

    def render(self, i: int, pd: dict, loop_flashes: list, n_acc: int) -> np.ndarray:
        D = self.D
        panel = np.full((self.ph, self.pw, 3), BG_DARK, dtype=np.uint8)

        # occupancy background: wall hits -> white walls on black (cropped + scaled)
        gray = (np.clip(self.hits / self.HIT_SAT, 0.0, 1.0) * 255).astype(np.uint8)
        if gray.shape[0] >= 2 and gray.shape[1] >= 2:
            scaled = cv2.resize(gray, (self.dw, self.dh), interpolation=cv2.INTER_NEAREST)
            panel[self.off_y:self.off_y + self.dh, self.off_x:self.off_x + self.dw] = \
                cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)

        # trajectory up to i, coloured per active-config segment
        pts_by_seg: dict[int, list] = {}
        for j in range(i + 1):
            kf = int(D.kf_ids[j])
            if kf not in pd:
                continue
            x, y, _ = pd[kf]
            pts_by_seg.setdefault(D.seg_index(j), []).append(self.w2p(x, y))
        for seg, pxy in pts_by_seg.items():
            col = SEG_COLORS[seg % len(SEG_COLORS)]
            for a, b in zip(pxy[:-1], pxy[1:]):
                cv2.line(panel, a, b, col, 2, cv2.LINE_AA)

        # loop-closure arcs (flashing)
        for fl in loop_flashes:
            q, c, life = fl
            if q not in pd or c not in pd:
                continue
            a = self.w2p(pd[q][0], pd[q][1]); b = self.w2p(pd[c][0], pd[c][1])
            alpha = max(0.25, life / 6.0)
            ov = panel.copy()
            cv2.line(ov, a, b, LOOP_COLOR, 2, cv2.LINE_AA)
            cv2.circle(ov, b, 5, LOOP_COLOR, -1, cv2.LINE_AA)
            cv2.addWeighted(ov, alpha, panel, 1 - alpha, 0, panel)

        # numbered switch diamonds (only those already reached)
        for n, (sw, ridx) in enumerate(zip(D.switches, D.sw_rowidx), 1):
            if ridx > i:
                continue
            kf = int(D.kf_ids[ridx])
            x, y, _ = pd.get(kf, D.final.get(kf, (0, 0, 0)))
            px, py = self.w2p(x, y)
            cv2.circle(panel, (px, py), 11, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(panel, (px, py), 11, (0, 0, 0), 2, cv2.LINE_AA)
            _put_text(panel, str(n), px - 5, py + 5, (0, 0, 0), 0.55, 2)

        # start + current robot
        kf0 = int(D.kf_ids[0])
        if kf0 in pd:
            sp = self.w2p(pd[kf0][0], pd[kf0][1])
            cv2.circle(panel, sp, 7, START_COLOR, -1, cv2.LINE_AA)
        kfi = int(D.kf_ids[i])
        if kfi in pd:
            x, y, _ = pd[kfi]
            cp = self.w2p(x, y)
            cv2.circle(panel, cp, 9, ROBOT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(panel, cp, 9, (255, 255, 255), 1, cv2.LINE_AA)

        # header + counters
        _put_text(panel, "Occupancy map + live loop closure", 24, 40, TITLE_COLOR, 0.8, 2)
        _put_text(panel, f"loops accepted: {n_acc}", 24, 62, SUBTITLE_COLOR, 0.55, 1)
        _draw_scale_bar(panel, 24, self.ph - 16, 2.0, self.grid.res / self.scale)
        return panel


# ------------------------------------------------------------------ POV panel
def _pov_panel(img, sensor: str, fe: str, kf: int, panel_w: int, panel_h: int,
               header_h: int = 70, footer_h: int = 40) -> np.ndarray:
    panel = np.full((panel_h, panel_w, 3), BG_DARK, dtype=np.uint8)
    area_w, area_h = panel_w - 48, panel_h - header_h - footer_h
    if img is not None:
        h, w = img.shape[:2]
        s = min(area_w / w, area_h / h)
        dw, dh = int(w * s), int(h * s)
        vis = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
        ox = 24 + (area_w - dw) // 2
        oy = header_h + (area_h - dh) // 2
        panel[oy:oy + dh, ox:ox + dw] = vis
    else:
        _put_text(panel, "(no camera frame)", panel_w // 2 - 90, panel_h // 2,
                  LABEL_COLOR, 0.7, 1)
    _put_text(panel, "Robot camera (POV)", 24, 40, TITLE_COLOR, 0.8, 2)
    _put_text(panel, f"keyframe {kf}   sensor={sensor}   front-end={_FE_SHORT.get(fe, fe)}",
              24, 62, SUBTITLE_COLOR, 0.55, 1)
    return panel


# ------------------------------------------------------------------ HUD + banner
def _draw_hud(frame, D: _Data, j: int, changed_axis: str | None, flash_t: float,
              hud_y: int, hud_h: int, width: int):
    """Persistent [Front-end | Proposer | Verifier] strip; flashes the changed cell."""
    cv2.rectangle(frame, (0, hud_y), (width, hud_y + hud_h), (22, 22, 30), -1)
    cv2.line(frame, (0, hud_y), (width, hud_y), (70, 80, 96), 1, cv2.LINE_AA)
    cells = [("Front-end", _FE_SHORT.get(D.fe[j], D.fe[j]), "fe"),
             ("Proposer", D.proposer[j], "proposer"),
             ("Verifier", D.verifier[j], "verifier")]
    cw = width // 3
    for ci, (label, value, axis) in enumerate(cells):
        x0 = ci * cw
        cx = x0 + cw // 2
        col = LABEL_COLOR
        if axis == changed_axis:
            c = ((1 - flash_t) * FLASH_A + flash_t * FLASH_B).astype(int)
            col = (int(c[0]), int(c[1]), int(c[2]))
            cv2.rectangle(frame, (x0 + 6, hud_y + 6), (x0 + cw - 6, hud_y + hud_h - 6),
                          col, 2, cv2.LINE_AA)
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        _put_text(frame, label, cx - lw // 2, hud_y + 34, SUBTITLE_COLOR, 0.6, 1)
        (vw, _), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        _put_text(frame, value, cx - vw // 2, hud_y + hud_h - 20, col, 1.0, 2)
        if ci < 2:
            cv2.line(frame, ((ci + 1) * cw, hud_y + 8), ((ci + 1) * cw, hud_y + hud_h - 8),
                     (60, 60, 74), 1, cv2.LINE_AA)


def _draw_banner(frame, n: int, sw, width: int, y: int):
    frm = _FE_SHORT.get(sw["frm"], sw["frm"]); to = _FE_SHORT.get(sw["to"], sw["to"])
    axis = {"fe": "Front-end", "proposer": "Proposer", "verifier": "Verifier"}.get(
        sw["kind"], sw["kind"])
    text = f"SWITCH {n}    {axis}:  {frm}  ->  {to}"
    if str(sw.get("autofallback", "")).lower() in ("true", "1"):
        text += "   [auto-fallback]"
    bh = 64
    cv2.rectangle(frame, (0, y), (width, y + bh), BANNER_BG, -1)   # opaque: hides panel titles
    cv2.rectangle(frame, (0, y), (width, y + bh), (60, 200, 255), 2)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    _put_text(frame, text, width // 2 - tw // 2, y + 42, (90, 235, 255), 1.0, 2)


def _draw_speed_badge(frame, text: str, width: int):
    """Small, notable badge in the top-right corner: playback speed vs real time."""
    scale, th = 0.72, 2
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    pad = 12
    x1, y0 = width - 20, 18
    x0, y1 = x1 - tw - 2 * pad, y0 + tht + 2 * pad
    box = frame.copy()
    cv2.rectangle(box, (x0, y0), (x1, y1), (25, 25, 32), -1)
    cv2.addWeighted(box, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 180, 235), 1, cv2.LINE_AA)
    _put_text(frame, text, x0 + pad, y1 - pad, (120, 225, 255), scale, th)


# ------------------------------------------------------------------ main render
VIDEO_FPS = 30.0        # fixed, smooth output cadence (not user-facing)


def render(run_dir: Path, out_path: Path, dataset: Path, speed: float,
           hold_s: float, width: int, height: int):
    D = _Data(run_dir, dataset)
    print(f"[video] {D.N} keyframes, {len(D.switches)} switches, "
          f"{len(D.snap_counts)} loop snapshots")

    hud_h = 110
    panel_h = height - hud_h
    panel_w = width // 2
    cvs = _MapCanvas(D, panel_w, panel_h)

    # Drive the video by the DATASET CLOCK, not the keyframe index: keyframes are
    # not uniform in time (the runner emits them on motion), so stepping one per
    # frame makes motion look jerky. Advancing a fixed slice of dataset time per
    # output frame keeps the apparent motion uniform. speed = real-time multiple.
    stamps = D.stamps
    t0, t_end = float(stamps[0]), float(stamps[-1])
    dt = speed / VIDEO_FPS                 # dataset seconds advanced per video frame
    speed_txt = "real-time" if abs(speed - 1.0) < 1e-6 else f"{speed:g}x real-time"
    print(f"[video] speed x{speed:g}  ->  {(t_end - t0) / speed:.1f}s video ({speed_txt})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, VIDEO_FPS, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {out_path}")
    hold_frames = max(1, int(round(hold_s * VIDEO_FPS)))

    # switch keyframe-row index -> (switch number, switch row)
    sw_at = {ridx: (n, D.switches[n - 1]) for n, ridx in enumerate(D.sw_rowidx, 1)}
    flashes: list[list] = []      # [query, cand, life]

    def _compose(base, i, changed_axis=None, flash_t=0.0, banner=None, banner_n=0):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:panel_h] = base
        if banner is not None:
            _draw_banner(frame, banner_n, banner, width, y=10)
        _draw_hud(frame, D, i, changed_axis, flash_t, panel_h, hud_h, width)
        _draw_speed_badge(frame, speed_txt, width)
        return frame

    t, prev_i, fnum = t0, -1, 0
    while t <= t_end + 1e-9:
        i = max(0, int(np.searchsorted(stamps, t, side="right")) - 1)
        pd = cvs.update(i)

        # loops that became active since the last frame; switches crossed since then
        for j in range(prev_i + 1, i + 1):
            for c in D.loops_by_query.get(int(D.kf_ids[j]), []):
                flashes.append([int(D.kf_ids[j]), c, 8])
        crossed = [(n, sw) for ridx, (n, sw) in sorted(sw_at.items())
                   if prev_i < ridx <= i]
        n_acc = sum(len(D.loops_by_query.get(int(k), [])) for k in D.kf_ids[:i + 1])

        map_panel = cvs.render(i, pd, flashes, n_acc)
        pov_panel = _pov_panel(D.rgb_frame(t), D.sensor[i], D.fe[i],
                               int(D.kf_ids[i]), panel_w, panel_h)
        base = np.hstack([pov_panel, map_panel])
        cv2.line(base, (panel_w - 1, 0), (panel_w - 1, panel_h), DIVIDER_COLOR, 2)

        writer.write(_compose(base, i)); fnum += 1
        # hold (freeze the clock) at each switch as it is crossed
        for n, sw in crossed:
            for r in range(hold_frames):
                ft = 0.5 + 0.5 * np.sin(r / hold_frames * np.pi)
                writer.write(_compose(base, i, sw["kind"], ft, sw, n)); fnum += 1

        for fl in flashes:
            fl[2] -= 1
        flashes = [fl for fl in flashes if fl[2] > 0]

        prev_i, t = i, t + dt
        if fnum % 150 == 0:
            print(f"[video]  t+{t - t0:5.0f}s  kf {i + 1}/{D.N}  loops={n_acc}")

    writer.release()
    print(f"[video] saved: {out_path}  ({fnum} frames, {fnum / VIDEO_FPS:.1f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="switching-demo run dir")
    ap.add_argument("--out", type=Path, required=True, help="output MP4 path")
    ap.add_argument("--dataset", type=Path, default=Path("datasets/lab_hybrid"),
                    help="dataset dir for POV RGB frames (default: datasets/lab_hybrid)")
    ap.add_argument("--speed", type=float, default=6.0,
                    help="playback speed vs real time: 1 = actual real-time, "
                         "N = N times faster (default 6)")
    ap.add_argument("--hold-s", type=float, default=0.8, dest="hold_s",
                    help="seconds to hold the frame at each switch (default 0.8)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    a = ap.parse_args()
    render(a.run, a.out, a.dataset, a.speed, a.hold_s, a.width, a.height)


if __name__ == "__main__":
    main()
