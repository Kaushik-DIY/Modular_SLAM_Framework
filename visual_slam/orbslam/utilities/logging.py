"""
=============================================================================
visual_slam/orbslam/utilities/logging.py

Small pySLAM-compatible Printer subset.

Reference:
- pySLAM: pyslam/utilities/logging.py

Only Printer methods used by the ORB-SLAM subset are implemented.
=============================================================================
"""

from __future__ import annotations


class Printer:
    @staticmethod
    def red(*args, **kwargs):
        print(*args, **kwargs)

    @staticmethod
    def green(*args, **kwargs):
        print(*args, **kwargs)

    @staticmethod
    def blue(*args, **kwargs):
        print(*args, **kwargs)

    @staticmethod
    def orange(*args, **kwargs):
        print(*args, **kwargs)

    @staticmethod
    def purple(*args, **kwargs):
        print(*args, **kwargs)

    @staticmethod
    def cyan(*args, **kwargs):
        print(*args, **kwargs)

    @staticmethod
    def yellow(*args, **kwargs):
        print(*args, **kwargs)
