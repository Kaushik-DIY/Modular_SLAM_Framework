from __future__ import annotations

from typing import Protocol

from slam_core.matching.scan_to_submap.types import (
    SubmapMatchRequest,
    SubmapMatchResponse,
)


class IScanToSubmapBackend(Protocol):
    """
    Internal backend contract for scan-to-submap matching.

    A backend receives an explicit target submap and returns the best match
    against that target. It must not contain SLAM-specific graph policy.
    """

    def match(self, request: SubmapMatchRequest) -> SubmapMatchResponse:
        ...