"""
area_handler.py

Responsible for:
- Step 1: Receiving unordered boundary points
- Step 2: Constructing a closed polygon
- Step 3: Shrinking polygon based on safety margin
- Step 4: Determining optimal movement direction

Standalone Python, ROS1-compatible by design.
"""

from typing import List, Tuple
import math

from shapely.geometry import Polygon
from shapely.ops import unary_union


Point2D = Tuple[float, float]


class AreaHandler:
    """
    Handles geometric preprocessing of the coverage area.
    """

    def __init__(
        self,
        boundary_points: List[Point2D],
        safety_margin: float,
    ):
        if len(boundary_points) != 4:
            raise ValueError(
                f"AreaHandler requires exactly 4 points, got {len(boundary_points)}"
            )
        
        self._raw_points = boundary_points
        self._safety_margin = safety_margin

        self._raw_polygon: Polygon = None
        self._safe_polygon: Polygon = None

        self._movement_axis: str = None  # "horizontal" or "vertical"
        self._coverage_axis: str = None

        self._process()

    # -------------------------
    # Public API
    # -------------------------

    def get_safe_polygon(self) -> Polygon:
        return self._safe_polygon

    def get_movement_axis(self) -> str:
        """
        Returns:
            "horizontal" or "vertical"
        """
        return self._movement_axis

    def get_coverage_axis(self) -> str:
        """
        Returns:
            "horizontal" or "vertical"
        """
        return self._coverage_axis

    # -------------------------
    # Core pipeline
    # -------------------------

    def _process(self):
        self._create_polygon()
        self._shrink_polygon()
        self._determine_movement_direction()

    # -------------------------
    # Step 2: Polygon creation
    # -------------------------

    def _create_polygon(self):
        if len(self._raw_points) != 4:
            raise ValueError("Exactly 4 points are required")

        ordered = self._order_points_ccw(self._raw_points)

        polygon = Polygon(ordered)

        if not polygon.is_valid or polygon.area <= 0:
            raise ValueError("Invalid polygon after ordering")

        self._raw_polygon = polygon
    # -------------------------
    # Step 3: Polygon shrinking
    # -------------------------

    def _shrink_polygon(self):
        """
        Shrinks polygon inward by safety margin.
        """
        if self._raw_polygon is None:
            raise RuntimeError("Raw polygon not initialized")

        shrunk = self._raw_polygon.buffer(-self._safety_margin)

        if shrunk.is_empty:
            raise ValueError("Safety margin too large, polygon vanished")

        # In case buffer creates multiple polygons, keep the largest
        if shrunk.geom_type == "MultiPolygon":
            shrunk = max(shrunk.geoms, key=lambda p: p.area)

        self._safe_polygon = shrunk

    # -------------------------
    # Step 4: Direction decision
    # -------------------------
    def _determine_movement_direction(self):
        """
        Determines movement direction:
        - Arrow originates from polygon centroid
        - Points perpendicular to the longest side of the safe polygon
        """
        if self._safe_polygon is None:
            raise RuntimeError("Safe polygon not initialized")

        coords = list(self._safe_polygon.exterior.coords)
        edges = [(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]

        # Find the longest edge
        longest_edge = max(edges, key=lambda e: math.hypot(e[1][0]-e[0][0], e[1][1]-e[0][1]))
        start, end = longest_edge
        self._longest_edge = (start, end)

        # Compute polygon centroid
        centroid = self._safe_polygon.centroid
        self._centroid = (centroid.x, centroid.y)

        # Compute midpoint of longest edge
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        self._longest_edge_midpoint = (mid_x, mid_y)

        # Compute perpendicular vector (normal) to the longest edge
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        # Normal vector (perpendicular)
        # Two choices of perpendicular: (-dy, dx) or (dy, -dx)
        # We'll pick (-dy, dx) as pointing outward from centroid
        norm_length = math.hypot(dx, dy)
        nx = -dy / norm_length
        ny = dx / norm_length

        # To ensure arrow points from centroid toward the polygon interior correctly,
        # check dot product with vector centroid → edge midpoint
        vx = mid_x - centroid.x
        vy = mid_y - centroid.y
        dot = nx * vx + ny * vy
        if dot < 0:  # reverse normal if pointing the wrong way
            nx *= -1
            ny *= -1

        self._movement_vector = (nx, ny)  # normalized perpendicular vector


    def _order_points_ccw(self, points: List[Point2D]) -> List[Point2D]:
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)

        def angle(p):
            return math.atan2(p[1] - cy, p[0] - cx)

        return sorted(points, key=angle)
    # -------------------------
    # Debug helpers (optional)
    # -------------------------

    def debug_summary(self) -> dict:
        return {
            "raw_area": self._raw_polygon.area if self._raw_polygon else None,
            "safe_area": self._safe_polygon.area if self._safe_polygon else None,
            "movement_axis": self._movement_axis,
            "coverage_axis": self._coverage_axis,
        }

    def get_original_polygon_xy(self):
        if self._raw_polygon is None:
            return None
        return list(self._raw_polygon.exterior.coords)

    def get_safe_polygon_xy(self):
        if self._safe_polygon is None:
            return None
        return list(self._safe_polygon.exterior.coords)

    def get_centroid(self):
        return self._centroid

    def get_movement_vector(self):
        """Returns normalized perpendicular vector pointing toward longest side"""
        return self._movement_vector

    def get_longest_edge(self):
        return self._longest_edge

    def get_longest_edge_midpoint(self):
        return self._longest_edge_midpoint
    
    def get_lane_direction_vector(self):
        """
        Returns a normalized vector along which lanes should be created.
        Lanes are perpendicular to the longest side → vector = perpendicular to longest edge
        """
        return self._movement_vector  # already perpendicular to longest edge
