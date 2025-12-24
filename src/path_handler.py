from typing import List, Tuple, Callable
from shapely.geometry import LineString, Polygon
import numpy as np
import math

Point2D = Tuple[float, float]
Pose2D = Tuple[float, float, float]  # x, y, yaw

class PathHandler:
    def __init__(
        self,
        safe_polygon: Polygon,
        lane_direction: Tuple[float, float],  # normalized vector
        lane_spacing: float = 1.0,
        lane_callback: Callable = None,
        step_callback: Callable = None,  # callback per Pose step for visualizer
    ):
        """
        :param lane_direction: normalized vector perpendicular to longest edge
        """
        self._poly = safe_polygon
        self._lane_dir = lane_direction
        self._lane_spacing = lane_spacing

        self._lanes: List[List[Point2D]] = []
        self._lane_callback = lane_callback
        self._step_callback = step_callback

        # Step 7 arrays
        self.start_heading: List[Pose2D] = []
        self.goal_assembly_point: List[Pose2D] = []
        self.lane_heading: List[Pose2D] = []

        self._generate_lanes()
        # self._generate_point_sequencing()
        self._generate_pose_sequences()

    def get_lanes(self) -> List[List[Point2D]]:
        return self._lanes

    # -------------------------
    # Step 5: Lane generation
    # -------------------------
    def _generate_lanes(self):
        # Compute polygon bounds to estimate number of lanes
        minx, miny, maxx, maxy = self._poly.bounds
        centroid = self._poly.centroid
        cx, cy = centroid.x, centroid.y

        # Lane direction vector
        dx, dy = self._lane_dir
        # Perpendicular vector for spacing between lanes
        perp_dx, perp_dy = -dy, dx

        # Compute maximum distance along perpendicular direction
        corners = list(self._poly.exterior.coords)
        projections = [x * perp_dx + y * perp_dy for x, y in corners]
        min_proj, max_proj = min(projections), max(projections)

        # Generate lane offsets along perpendicular vector
        num_lanes = int((max_proj - min_proj) / self._lane_spacing) + 1
        offsets = np.linspace(min_proj, max_proj, num_lanes)

        for i, off in enumerate(offsets):
            # Create lane line along lane_dir at given offset
            start = (dx * -1000 + perp_dx * off, dy * -1000 + perp_dy * off)
            end   = (dx * 1000  + perp_dx * off, dy * 1000  + perp_dy * off)
            raw_line = LineString([start, end])
            clipped = raw_line.intersection(self._poly)
            self._extract_lane(clipped, reverse=(i % 2 == 1))

    def _generate_pose_sequences(self):
        """
        Compute start_heading[], goal_assembly_point[], lane_heading[].
        Assumes lanes already generated.
        """

        self.start_heading = []
        self.goal_assembly_point = []
        self.lane_heading = []

        # Determine longest side vector of the polygon
        exterior = self._poly.exterior.coords
        max_len = 0
        longest_vec = (0, 0)
        longest_midpoint = (0, 0)
        for i in range(len(exterior)-1):
            x1, y1 = exterior[i]
            x2, y2 = exterior[i+1]
            dx, dy = x2-x1, y2-y1
            l = np.hypot(dx, dy)
            if l > max_len:
                max_len = l
                longest_vec = (dx, dy)
                longest_midpoint = ((x1+x2)/2, (y1+y2)/2)

        # Unit vector along longest side
        lvx, lvy = longest_vec
        length = np.hypot(lvx, lvy)
        lvx /= length
        lvy /= length

        # Lane vector (perpendicular to longest side)
        lane_dx, lane_dy = -lvy, lvx
        lane_yaw = math.atan2(lane_dy, lane_dx)  # heading along lane

        for idx, lane in enumerate(self._lanes):
            x_start, y_start = lane[0]         # start of lane
            x_end, y_end = lane[-1]            # end of lane

            # Determine which end is closer to longest side
            # Use Euclidean distance to midpoint of longest side
            dist_start = np.hypot(x_start-longest_midpoint[0], y_start-longest_midpoint[1])
            dist_end   = np.hypot(x_end-longest_midpoint[0], y_end-longest_midpoint[1])

            if dist_start < dist_end:
                goal = (x_start, y_start)
                start = (x_end, y_end)
            else:
                start = (x_start, y_start)
                goal  = (x_end, y_end)

            # Append start_heading and goal_assembly_point along lane direction
            self.start_heading.append((*start, lane_yaw))
            self.goal_assembly_point.append((*goal, lane_yaw))

            # lane_heading: at start, face next lane start if exists
            if idx+1 < len(self._lanes):
                next_start = self._lanes[idx+1][0]
                yaw_lane = math.atan2(next_start[1]-start[1], next_start[0]-start[0])
            else:
                yaw_lane = lane_yaw  # last lane, keep same as lane

            self.lane_heading.append((*start, yaw_lane))


    def _extract_lane(self, geom, reverse=False):
        if geom.is_empty:
            return

        new_lanes = []

        if geom.geom_type == "LineString":
            lane = list(geom.coords)
            if reverse:
                lane = lane[::-1]
            new_lanes.append(lane)

        elif geom.geom_type == "MultiLineString":
            for g in geom.geoms:
                lane = list(g.coords)
                if reverse:
                    lane = lane[::-1]
                new_lanes.append(lane)

        for lane in new_lanes:
            self._lanes.append(lane)
            if self._lane_callback:
                self._lane_callback(lane)

    # -------------------------
    # Step 7: Pose sequencing
    # -------------------------
    def _generate_point_sequencing(self):
        """
        For each lane:
        1. start_heading -> goal_assembly_point
        2. move backward to start_heading
        3. rotate to lane_heading
        """
        num_lanes = len(self._lanes)
        for i, lane in enumerate(self._lanes):
            if len(lane) < 2:
                continue
            start = lane[0]
            end = lane[-1]

            # Yaw towards the longest side (forward/backward)
            dx, dy = self._lane_dir
            yaw_longest = math.atan2(dy, dx)

            # STEP 1: start_heading (forward) → goal_assembly_point
            self.start_heading.append((start[0], start[1], yaw_longest))
            self.goal_assembly_point.append((end[0], end[1], yaw_longest))

            # STEP 2: backward → same start_heading
            self.start_heading.append((start[0], start[1], yaw_longest))
            self.goal_assembly_point.append((end[0], end[1], yaw_longest))  # optional if needed

            # STEP 3: rotate at start → lane_heading
            if i + 1 < num_lanes:
                next_start = self._lanes[i+1][0]
                yaw_lane = math.atan2(next_start[1]-start[1], next_start[0]-start[0])
            else:
                yaw_lane = yaw_longest
            self.lane_heading.append((start[0], start[1], yaw_lane))

            # optional callback for visualization
            if self._step_callback:
                # forward
                self._step_callback((start[0], start[1], yaw_longest), "start_heading")
                self._step_callback((end[0], end[1], yaw_longest), "goal_assembly_point")
                # backward
                self._step_callback((start[0], start[1], yaw_longest), "start_heading")
                # rotate
                self._step_callback((start[0], start[1], yaw_lane), "lane_heading")