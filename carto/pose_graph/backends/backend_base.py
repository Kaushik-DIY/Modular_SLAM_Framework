# carto/pose_graph/backends/backend_base.py
from abc import ABC, abstractmethod
from carto.pose_graph.constraint import PoseGraphNode, PoseGraphSubmap, PoseGraphConstraint

class BackendBase(ABC):

    @abstractmethod
    def add_node(self, node: PoseGraphNode):
        """Register a trajectory node variable + initial estimate."""
        raise NotImplementedError

    @abstractmethod
    def add_submap(self, submap: PoseGraphSubmap):
        """Register a submap variable + initial estimate."""
        raise NotImplementedError

    @abstractmethod
    def add_constraint(self, constraint: PoseGraphConstraint):
        """Add a relative SE2 constraint between two variables."""
        raise NotImplementedError

    @abstractmethod
    def solve(self):
        """Run nonlinear optimization (GN/LM)."""
        raise NotImplementedError

    @abstractmethod
    def get_optimized_poses(self):
        """
        Return optimized poses as:
          {
            ("node", id): Pose2(...),
            ("submap", id): Pose2(...)
          }
        """
        raise NotImplementedError