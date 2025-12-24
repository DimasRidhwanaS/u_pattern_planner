#!/usr/bin/env python3
import rospy
import threading
import numpy as np
from typing import List, Tuple
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Trigger, TriggerResponse

from area_handler import AreaHandler
from path_handler import PathHandler

Point2D = Tuple[float, float]


class UPathClearanceNode:
    def __init__(self):
        rospy.init_node("u_path_clearance")

        # Parameters
        self.frame_id       = rospy.get_param("~frame_id", "map")
        self.mode           = rospy.get_param("~mode", "rviz")
        self.robot_radius   = rospy.get_param("~robot_radius", 0.3)
        self.safety_margin  = rospy.get_param("~safety_margin", 0.5)
        self.cost_threshold = rospy.get_param("~cost_threshold", 50)
        self.publish_rate   = rospy.get_param("~publish_rate", 1.0)

        # Topics
        self.published_point_topic  = rospy.get_param("~published_point_topic", "/published_point")
        self.costmap_topic          = rospy.get_param("~costmap_topic", "/move_base/global_costmap/costmap")
        self.goal_topic             = rospy.get_param("~goal_topic", "/move_base_simple/goal")

        # Internal state
        self._boundary_points: List[Point2D] = []
        self._costmap: OccupancyGrid = None
        self._lock = threading.Lock()

        # Subscribers
        rospy.Subscriber(self.published_point_topic,PointStamped,self._point_callback,queue_size=10,)
        rospy.Subscriber(self.costmap_topic,OccupancyGrid,self._costmap_callback,queue_size=1,)
        rospy.Subscriber(self.goal_topic,PoseStamped,self._goal_callback,queue_size=1,)

        # Services
        self._process_srv = rospy.Service("~process_area",Trigger,self._process_service_cb,)

        rospy.loginfo("[UPathClearance] Node initialized")

    # Callbacks
    def _point_callback(self, msg: PointStamped):
        with self._lock:
            self._boundary_points.append((msg.point.x, msg.point.y))
            rospy.loginfo(
                f"[UPathClearance] Point received ({len(self._boundary_points)})"
            )

    def _costmap_callback(self, msg: OccupancyGrid):
        with self._lock:
            self._costmap = msg

    def _goal_callback(self, msg: PoseStamped):
        # Reserved for future use (alignment / direction bias)
        pass

    # Service Logic
    def _process_service_cb(self, req: Trigger.Request) -> TriggerResponse:
        with self._lock:
            num_points = len(self._boundary_points)

            # ---------- Validation ----------
            if self._costmap is None:
                return TriggerResponse(
                    success=False,
                    message="global costmap not received yet"
                )

            try:
                # Computing Area
                area_handler = AreaHandler(
                    boundary_points=self._boundary_points,
                    safety_margin=self.safety_margin,
                )

                original_poly = area_handler.get_original_polygon()

                safe_polygon = self._refine_safe_polygon_with_costmap(
                    original_poly,
                    self._costmap,
                    self.cost_threshold,
                    self.robot_radius
                )

                if safe_polygon.is_empty:
                    return TriggerResponse(
                        success=False,
                        message="safe polygon fully blocked by obstacles"
                    )

                # Computing Path
                path_handler = PathHandler(
                    safe_polygon=safe_polygon,
                    lane_direction=area_handler.get_lane_direction_vector(),
                    lane_spacing=self.lane_spacing
                )

                lanes = path_handler.get_lanes()
                if not lanes:
                    return TriggerResponse(
                        success=False,
                        message="no valid lanes generated"
                    )

                self._publish_polygon(original_poly, ns="original")
                self._publish_polygon(safe_polygon, ns="safe")
                self._publish_lanes(lanes)

                self._publish_poses(path_handler.start_heading, ns="start")
                self._publish_poses(path_handler.goal_assembly_point, ns="goal")
                self._publish_poses(path_handler.lane_heading, ns="lane")
                
            except Exception as e:
                rospy.logerr(f"[UPathClearance] Processing failed: {e}")
                return TriggerResponse(
                    success=False,
                    message=str(e)
                )

            # ---------- Reset after processing ----------
            self._boundary_points.clear()

            rospy.loginfo("[UPathClearance] Area processed successfully")

            return TriggerResponse(
                success=True,
                message="area processed successfully"
            )

    def _refine_safe_polygon_with_costmap(
        self,
        original_polygon: Polygon,
        costmap_msg,
        cost_threshold: int,
        inflation_radius: float
    ) -> Polygon:

        resolution  = costmap_msg.info.resolution
        width       = costmap_msg.info.width
        height      = costmap_msg.info.height
        origin_x    = costmap_msg.info.origin.position.x
        origin_y    = costmap_msg.info.origin.position.y

        data = np.array(costmap_msg.data).reshape((height, width))

        obstacle_cells = []

        for y in range(height):
            for x in range(width):
                if data[y, x] >= cost_threshold:
                    wx = origin_x + (x + 0.5) * resolution
                    wy = origin_y + (y + 0.5) * resolution

                    half = resolution / 2.0
                    cell_poly = box(
                        wx - half,
                        wy - half,
                        wx + half,
                        wy + half
                    )

                    obstacle_cells.append(cell_poly)

        if not obstacle_cells:
            return original_polygon

        # Merge all obstacle cells
        obstacle_union = unary_union(obstacle_cells)

        # Inflate obstacles by robot + safety margin
        inflated_obstacles = obstacle_union.buffer(inflation_radius)

        # Subtract from original polygon
        safe_poly = original_polygon.difference(inflated_obstacles)

        if safe_poly.is_empty:
            return safe_poly

        # If multiple regions, keep largest
        if safe_poly.geom_type == "MultiPolygon":
            safe_poly = max(safe_poly.geoms, key=lambda p: p.area)

        return safe_poly

    def spin(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == "__main__":
    node = UPathClearanceNode()
    node.spin()
