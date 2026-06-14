#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ROS Melodic (Python 2) -> fusion SLAM (Python 3.8) bridge.

Runs under ROS Melodic's Python 2.7. Subscribes to the live RGB-D + 2D-LiDAR +
IMU topics, converts each message to the compact wire format
(`slam_core/fusion2/ros_source.py`), and forwards them over a Unix-domain socket
to the py3.8 SLAM process (`run_realtime.py --source ros`). This keeps ROS on its
native py2 and the SLAM stack on py3.8 with no ROS-py3 hacking.

  * RGB + depth are paired with message_filters.ApproximateTimeSynchronizer
    (they must co-register); LaserScan and Imu are forwarded as they arrive
    (the fusion layer soft-syncs via nearest_scan/nearest_rgbd).
  * The LaserScan is converted to (N,2) xy here (so the SLAM side needs no laser
    geometry); IMU yaw is taken from the orientation quaternion (about +Z), wz
    from angular_velocity.z.

Start the SLAM side first (it creates/binds the socket), then this node:

    python tools/ros/fusion_ros_bridge.py \
        --rgb /camera/color/image_raw --depth /camera/aligned_depth_to_color/image_raw \
        --scan /scan --imu /imu --socket /tmp/fusion_ros.sock
"""
import argparse
import math
import socket
import struct
import sys

import numpy as np
import rospy
import message_filters
from sensor_msgs.msg import Image, LaserScan, Imu

# ---- wire protocol (must match slam_core/fusion2/ros_source.py) -------------
_HDR = struct.Struct(">cdI")
TYPE_LIDAR = b"L"
TYPE_RGBD = b"R"
TYPE_IMU = b"I"


def _encode(etype, t, payload):
    return _HDR.pack(etype, float(t), len(payload)) + payload


def _quat_yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class Bridge(object):
    def __init__(self, args):
        self.args = args
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        rospy.loginfo("connecting to SLAM socket %s ...", args.socket)
        self.sock.connect(args.socket)   # SLAM side must be listening first
        rospy.loginfo("connected.")

        # OpenCV (cv_bridge can be ABI-fussy across py2/py3; use raw numpy instead)
        import cv2
        self.cv2 = cv2

        # paired rgb+depth via approximate time sync
        rgb_sub = message_filters.Subscriber(args.rgb, Image)
        dep_sub = message_filters.Subscriber(args.depth, Image)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, dep_sub], queue_size=args.queue, slop=args.slop)
        self.ts.registerCallback(self._rgbd_cb)
        rospy.Subscriber(args.scan, LaserScan, self._scan_cb, queue_size=args.queue)
        rospy.Subscriber(args.imu, Imu, self._imu_cb, queue_size=200)
        self.n_rgbd = self.n_scan = self.n_imu = 0

    def _send(self, etype, t, payload):
        try:
            self.sock.sendall(_encode(etype, t, payload))
        except socket.error as e:
            rospy.logerr("socket send failed (%s); SLAM process gone? shutting down.", e)
            rospy.signal_shutdown("socket closed")

    # --- callbacks --------------------------------------------------------
    def _imread_msg(self, msg):
        """sensor_msgs/Image -> numpy array (no cv_bridge dependency)."""
        dt = {"rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3), "mono8": (np.uint8, 1),
              "mono16": (np.uint16, 1), "16UC1": (np.uint16, 1), "8UC1": (np.uint8, 1),
              "32FC1": (np.float32, 1)}.get(msg.encoding)
        if dt is None:
            rospy.logwarn_throttle(5.0, "unhandled image encoding %s", msg.encoding)
            return None
        npdtype, ch = dt
        arr = np.frombuffer(msg.data, dtype=npdtype)
        arr = arr.reshape(msg.height, msg.width, ch) if ch > 1 else \
            arr.reshape(msg.height, msg.width)
        if msg.encoding == "rgb8":             # SLAM expects BGR (cv2 convention)
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)

    def _rgbd_cb(self, rgb_msg, dep_msg):
        rgb = self._imread_msg(rgb_msg)
        depth = self._imread_msg(dep_msg)
        if rgb is None or depth is None:
            return
        if depth.dtype == np.float32:          # metres -> uint16 mm (depth_factor=1000)
            depth = np.nan_to_num(depth * 1000.0).astype(np.uint16)
        t = rgb_msg.header.stamp.to_sec()
        rb = self.cv2.imencode(".jpg", rgb)[1].tobytes()
        db = self.cv2.imencode(".png", depth)[1].tobytes()
        payload = struct.pack(">I", len(rb)) + rb + struct.pack(">I", len(db)) + db
        self._send(TYPE_RGBD, t, payload)
        self.n_rgbd += 1

    def _scan_cb(self, msg):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        ang = msg.angle_min + np.arange(len(ranges), dtype=np.float32) * msg.angle_increment
        valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        xy = np.empty((int(valid.sum()), 2), dtype=np.float32)
        xy[:, 0] = ranges[valid] * np.cos(ang[valid])
        xy[:, 1] = ranges[valid] * np.sin(ang[valid])
        self._send(TYPE_LIDAR, msg.header.stamp.to_sec(), xy.tobytes())
        self.n_scan += 1

    def _imu_cb(self, msg):
        wz = msg.angular_velocity.z
        yaw = _quat_yaw(msg.orientation)
        self._send(TYPE_IMU, msg.header.stamp.to_sec(), struct.pack(">dd", wz, yaw))
        self.n_imu += 1

    def spin(self):
        rate = rospy.Rate(1.0)
        while not rospy.is_shutdown():
            rospy.loginfo("fwd rgbd=%d scan=%d imu=%d", self.n_rgbd, self.n_scan, self.n_imu)
            rate.sleep()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", default="/camera/color/image_raw")
    ap.add_argument("--depth", default="/camera/aligned_depth_to_color/image_raw")
    ap.add_argument("--scan", default="/scan")
    ap.add_argument("--imu", default="/imu")
    ap.add_argument("--socket", default="/tmp/fusion_ros.sock")
    ap.add_argument("--slop", type=float, default=0.03, help="rgb<->depth sync tolerance (s)")
    ap.add_argument("--queue", type=int, default=10)
    args, _ = ap.parse_known_args()

    rospy.init_node("fusion_ros_bridge", anonymous=True)
    try:
        Bridge(args).spin()
    except socket.error as e:
        rospy.logfatal("could not connect to %s: %s "
                       "(start the SLAM side first)", args.socket, e)
        sys.exit(1)


if __name__ == "__main__":
    main()
