#!/usr/bin/env python3

import numpy as np
import math
import threading
import traceback
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from typing import List, Tuple
import time

import rospy
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import Point, PoseStamped, PointStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

from area_handler import AreaHandler
from path_handler import PathHandler

class VisualizerMode:
    PYTHON = "python"
    RVIZ = "rviz"

class Visualizer:
    def __init__(
        self,
        mode: str = VisualizerMode.PYTHON,
        grid_size: int = 10,
        grid_resolution: float = 1.0,
        safety_margin: float = 0.5
    ):
        self.mode = mode
        self.grid_size = grid_size
        self.grid_resolution = grid_resolution
        self.safety_margin = safety_margin

        self.clicked_points: List[Tuple[float, float]] = []
        self.point_plots = []
        self.lane_plots = []  
        self.step_arrows = []


        # Plot handles
        self.point_plot = None
        self.original_poly_plot = None
        self.safe_poly_plot = None

        if self.mode == VisualizerMode.PYTHON:
            self._init_python_visualizer()
        elif self.mode == VisualizerMode.RVIZ:
            self._init_rviz_visualizer()
        else:
            raise ValueError("Unknown visualizer mode")


class PythonVisualizer(Visualizer):
    def _init_python_visualizer(self):
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title("Coverage Planner Visualizer")

        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(-self.grid_size, self.grid_size)
        self.ax.set_ylim(-self.grid_size, self.grid_size)

        self._draw_grid()

        self.fig.canvas.mpl_connect(
            "button_press_event", self._on_mouse_click
        )

        self._add_compute_button()

        plt.show()

    def _draw_grid(self):
        ticks = [
            i * self.grid_resolution
            for i in range(
                int(-self.grid_size / self.grid_resolution),
                int(self.grid_size / self.grid_resolution) + 1
            )
        ]

        self.ax.set_xticks(ticks)
        self.ax.set_yticks(ticks)

        self.ax.grid(
            True,
            color="gray",
            alpha=0.6,
            linewidth=0.8
        )

    def _on_mouse_click(self, event):
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        self.clicked_points.append((x, y))

        plot, = self.ax.plot(x, y, "bo")
        self.point_plots.append(plot)

        self.fig.canvas.draw_idle()

        print(f"[Visualizer] Published point: ({x:.2f}, {y:.2f})")
        
    def _add_compute_button(self):
        ax_compute = plt.axes([0.8, 0.02, 0.15, 0.06])
        self.btn_compute = Button(ax_compute, "Compute")
        self.btn_compute.on_clicked(self._on_compute_pressed)
        # self.btn_compute.on_clicked(self._visualize_goal_points_only)

        ax_clear = plt.axes([0.6, 0.02, 0.15, 0.06])
        self.btn_clear = Button(ax_clear, "Clear")
        self.btn_clear.on_clicked(self._on_clear_pressed)

    def _on_compute_pressed(self, event):
        print("[Visualizer] Compute pressed")

        # Visualizing Area
        area = AreaHandler(
            boundary_points=self.clicked_points,
            safety_margin=self.safety_margin
        )

        print("[Visualizer] Coverage axis:", area.get_coverage_axis())
        print("[Visualizer] Movement axis:", area.get_movement_axis())

        self._visualize_area(area)  # Give visual for safe and original polygon

        # Visualizing Lanes
        def lane_callback(lane):
            xs, ys = zip(*lane)
            plot, = self.ax.plot(xs, ys, "r--", linewidth=1.5)
            self.lane_plots.append(plot)
            self.fig.canvas.draw_idle()
            plt.pause(0.3)

        # Visualizing Pose Sequencing
        def step_callback(pose, kind):
            x, y, yaw = pose
            arrow_color = {
                "start_heading": "blue",
                "goal_assembly_point": "green",
                "lane_heading": "red"
            }.get(kind, "black")
            
            # Arrow length
            L = 0.5
            dx = L * np.cos(yaw)
            dy = L * np.sin(yaw)
            
            arr = self.ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.3, fc=arrow_color, ec=arrow_color)
            
            if not hasattr(self, "step_arrows"):
                self.step_arrows = []
            self.step_arrows.append(arr)
            
            self.fig.canvas.draw_idle()
            plt.pause(0.2)  # step-by-step pause

        # After visualizing area and movement arrow
        lane_dir = area.get_lane_direction_vector()  # perpendicular to longest side

        path_handler = PathHandler(
            safe_polygon=area.get_safe_polygon(),
            lane_direction=area.get_lane_direction_vector(),
            lane_spacing=1.0,
            lane_callback=lane_callback,      # previous red dashed lane
            step_callback=step_callback       # new arrows for each step
        )


        print(f"[Visualizer] Generated {len(path_handler.get_lanes())} lanes")

        # -----------------------------
        # Step-by-step Pose Sequencing
        # -----------------------------
        num_lanes = len(path_handler.start_heading)
        for i in range(num_lanes):
            # 1. start_heading[i]
            step_callback(path_handler.start_heading[i], "start_heading")

            # 2. goal_assembly_point[i]
            step_callback(path_handler.goal_assembly_point[i], "goal_assembly_point")

            # 3. start_heading[i] again (backward movement)
            step_callback(path_handler.start_heading[i], "start_heading")

            # 4. lane_heading[i]
            step_callback(path_handler.lane_heading[i], "lane_heading")

    def _on_clear_pressed(self, event):
        print("[Visualizer] Clearing all data")

        # Clear clicked points
        self.clicked_points.clear()

        # Remove point plots
        for p in self.point_plots:
            p.remove()
        self.point_plots.clear()

        # Remove polygons
        if self.original_poly_plot:
            self.original_poly_plot.remove()
            self.original_poly_plot = None

        if self.safe_poly_plot:
            self.safe_poly_plot.remove()
            self.safe_poly_plot = None

        # Remove movement arrow
        if hasattr(self, "movement_arrow") and self.movement_arrow:
            self.movement_arrow.remove()
            self.movement_arrow = None

        # Remove lanes
        if hasattr(self, "lane_plots") and self.lane_plots:
            for lp in self.lane_plots:
                lp.remove()
            self.lane_plots.clear()
        
        # Remove step arrows
        if hasattr(self, "step_arrows") and self.step_arrows:
            for arr in self.step_arrows:
                arr.remove()
            self.step_arrows.clear()


        self.ax.legend([], [], frameon=False)
        self.fig.canvas.draw_idle()

    def _visualize_area(self, area: AreaHandler):
        # Clear previous polygons and arrows
        if self.original_poly_plot:
            self.original_poly_plot.remove()
        if self.safe_poly_plot:
            self.safe_poly_plot.remove()
        if hasattr(self, "movement_arrow") and self.movement_arrow:
            self.movement_arrow.remove()
            self.movement_arrow = None

        # Original polygon
        original_xy = area.get_original_polygon_xy()
        if original_xy:
            xs, ys = zip(*original_xy)
            self.original_poly_plot, = self.ax.plot(
                xs, ys, "b-", linewidth=2, label="Original Area"
            )

        # Safe polygon
        safe_xy = area.get_safe_polygon_xy()
        if safe_xy:
            xs, ys = zip(*safe_xy)
            self.safe_poly_plot, = self.ax.plot(
                xs, ys, "g-", linewidth=2, label="Safe Area"
            )

        # Draw movement arrow
        safe_poly = area.get_safe_polygon()
        min_x, min_y, max_x, max_y = safe_poly.bounds
        if area.get_movement_axis() == "horizontal":
            y_mid = (min_y + max_y) / 2
            start = (min_x, y_mid)
            end = (max_x, y_mid)
        else:
            x_mid = (min_x + max_x) / 2
            start = (x_mid, min_y)
            end = (x_mid, max_y)

        centroid = area.get_centroid()
        nx, ny = area.get_movement_vector()

        arrow_length = 2.0  # adjust for visibility
        end = (centroid[0] + nx * arrow_length, centroid[1] + ny * arrow_length)

        self.movement_arrow = self.ax.annotate(
            "",
            xy=end,
            xytext=centroid,
            arrowprops=dict(
                facecolor="red",
                edgecolor="red",
                width=2,
                headwidth=10,
                headlength=15
            )
        )

        self.ax.legend()
        self.fig.canvas.draw_idle()

    # Visualization helpers
    def _visualize_goal_points_only(self, event=None):
        """
        Visualize only the goal_assembly_point[] from the current clicked points.
        Step-by-step with SPACE key.
        """
        # Create AreaHandler
        area = AreaHandler(
            boundary_points=self.clicked_points,
            safety_margin=self.safety_margin
        )

        # Get lane direction perpendicular to longest side
        lane_dir = area.get_lane_direction_vector()

        # Generate lanes with PathHandler (but only to get goal points)
        path_handler = PathHandler(
            safe_polygon=area.get_safe_polygon(),
            lane_direction=lane_dir,
            lane_spacing=1.0
        )

        goal_points = path_handler.goal_assembly_point  # <- your real array

        # Clear previous arrows
        if hasattr(self, "step_arrows"):
            for arr in self.step_arrows:
                arr.remove()
            self.step_arrows = []

        # Step-by-step display
        index = 0
        current_arrow = None

        def show_next_goal(event):
            nonlocal index, current_arrow
            if event.key != " ":
                return
            if current_arrow:
                current_arrow.remove()
                current_arrow = None
            if index >= len(goal_points):
                print("[Visualizer] All goal_assembly_point[] displayed")
                return

            x, y, yaw = goal_points[index]
            L = 0.5
            dx = L * np.cos(yaw)
            dy = L * np.sin(yaw)
            current_arrow = self.ax.arrow(x, y, dx, dy,
                                        head_width=0.2, head_length=0.3,
                                        fc="green", ec="green")
            self.ax.plot(x, y, "bo")
            self.fig.canvas.draw_idle()
            print(f"[Visualizer] Showing goal_assembly_point[{index}]: ({x:.2f},{y:.2f})")
            index += 1

        self.fig.canvas.mpl_connect("key_press_event", show_next_goal)
        print("[Visualizer] Press SPACE to show goal_assembly_point[] step by step")

    def _visualize_goal_points_stepwise(self, path_handler):
        """
        Step-by-step visualization of goal_assembly_point[].
        Each step clears the previous arrow.
        """
        goal_points = path_handler.goal_assembly_point
        self.current_step = 0

        def next_step(event=None):
            # Remove previous arrow
            if hasattr(self, "step_arrow") and self.step_arrow:
                self.step_arrow.remove()
                self.step_arrow = None

            if self.current_step >= len(goal_points):
                print("[Visualizer] Completed all goal points")
                return

            x, y, yaw = goal_points[self.current_step]

            # Draw arrow at goal point
            arrow_length = 0.5
            dx = arrow_length * np.cos(yaw)
            dy = arrow_length * np.sin(yaw)

            self.step_arrow = self.ax.arrow(
                x, y, dx, dy,
                head_width=0.2, head_length=0.3,
                fc="green", ec="green"
            )

            self.ax.figure.canvas.draw_idle()
            print(f"[Visualizer] Showing goal_assembly_point[{self.current_step}]: ({x:.2f},{y:.2f})")
            self.current_step += 1

        # Connect to key press (e.g., space to go next)
        self.fig.canvas.mpl_connect("key_press_event", lambda event: next_step() if event.key == " " else None)
        print("[Visualizer] Press SPACE to show next goal point")


class RVIZVisualizer(Visualizer):
    def __init__(
        self, 
        frame_id="map",
        point_topic="/published_point",
        compute=False
    ):
        self.frame_id = frame_id
        self._lock = threading.Lock()   
        self.clicked_points = []  
        self.compute = compute

        # Publishers
        self.poly_pub = rospy.Publisher("~area_markers", Marker, queue_size=10)
        self.lane_pub = rospy.Publisher("~lane_markers", MarkerArray, queue_size=10)
        self.step_pub = rospy.Publisher("~step_markers", MarkerArray, queue_size=10)
        self.point_pub = rospy.Publisher("~clicked_points",MarkerArray,queue_size=10)

        rospy.Subscriber(point_topic, PointStamped, self._point_callback)

        # Service to trigger computation
        self.compute_srv = rospy.Service(
            "~compute_area", Trigger, self._compute_service_callback
        )

        self.clear_srv = rospy.Service(
            "~clear_rviz", Trigger, self._clear_rviz_service_callback
        )

        rospy.loginfo("[RVizVisualizer] Initialized")

    # Internal Methods
    def _point_callback(self, msg: PointStamped):
        with self._lock:
            self.clicked_points.append((msg.point.x, msg.point.y))
            rospy.loginfo(f"[RVizVisualizer] Point received ({len(self.clicked_points)}) : ({msg.point.x:.1f}, {msg.point.y:.1f})")

        # Publish a blue sphere marker for this point
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "clicked_points"
        marker.id = len(self.clicked_points)  # unique ID
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = msg.point.x
        marker.pose.position.y = msg.point.y
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3  # sphere diameter
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Publish immediately
        self.poly_pub.publish(marker)

    def _compute_service_callback(self, req: Trigger):
        with self._lock:
            if not self.clicked_points:
                rospy.logwarn("[RVizVisualizer] No points received yet, cannot compute")
                return TriggerResponse(success=False, message="No points received")

            points_copy = self.clicked_points.copy()


        if self.compute:
            # --- Compute area & path ---
            area = AreaHandler(boundary_points=points_copy, safety_margin=0.2)
            safe_poly_coords = area.get_safe_polygon_xy()

            path_handler = PathHandler(
                safe_polygon=area.get_safe_polygon(),
                lane_direction=area.get_lane_direction_vector(),
                lane_spacing=0.3
            )

            # --- Publish to RVIZ ---
            self.display_polygon(safe_poly_coords)
            self.display_lanes(path_handler.get_lanes())
            self.display_targets(
                path_handler.start_heading,
                path_handler.goal_assembly_point,
                path_handler.lane_heading
            )

        rospy.loginfo("[RVizVisualizer] Computation and publishing complete")
        return TriggerResponse(success=True, message="Area and path published")

    def _clear_rviz_service_callback(self, req):
        rospy.loginfo("[RVizVisualizer] Clearing RViz markers and points")

        with self._lock:
            self.clicked_points.clear()

        # --- Clear polygon ---
        poly_marker = Marker()
        poly_marker.header.frame_id = self.frame_id
        poly_marker.action = Marker.DELETEALL
        self.poly_pub.publish(poly_marker)

        # --- Clear lanes ---
        lane_array = MarkerArray()
        lane_array.markers.append(self._make_delete_all_marker("lanes"))
        self.lane_pub.publish(lane_array)

        # --- Clear steps (arrows) ---
        step_array = MarkerArray()
        step_array.markers.append(self._make_delete_all_marker("start_heading"))
        step_array.markers.append(self._make_delete_all_marker("goal_assembly_point"))
        step_array.markers.append(self._make_delete_all_marker("lane_heading"))
        self.step_pub.publish(step_array)

        # --- Clear clicked points (blue dots) ---
        point_array = MarkerArray()
        point_array.markers.append(self._make_delete_all_marker("clicked_points"))
        self.point_pub.publish(point_array)

        return TriggerResponse(
            success=True,
            message="RViz cleared"
        )

    # Helper
    def _make_delete_all_marker(self, ns):
        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.action = Marker.DELETEALL
        return m

    # Public API
    def display_polygon(self, polygon_coords, ns="safe_polygon", color=(0,1,0,0.3), scale=0.05):
        """
        polygon_coords: list of (x, y)
        """
        if not polygon_coords:
            return

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color

        # Add points to marker
        for x, y in polygon_coords:
            pt = Point()
            pt.x, pt.y, pt.z = x, y, 0.0
            marker.points.append(pt)

        # Close polygon
        pt = Point()
        pt.x, pt.y, pt.z = polygon_coords[0][0], polygon_coords[0][1], 0.0
        marker.points.append(pt)

        self.poly_pub.publish(marker)

    def display_lanes(self, lanes_coords):
        marker_array = MarkerArray()
        for i, lane in enumerate(lanes_coords):
            if not lane:
                continue
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = "lanes"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.03
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.0, 0.0, 1.0

            for x, y in lane:
                pt = Point()
                pt.x, pt.y, pt.z = x, y, 0.0
                marker.points.append(pt)

            marker_array.markers.append(marker)
        self.lane_pub.publish(marker_array)

    def display_targets(self, start_heading, goal_assembly_point, lane_heading):
        marker_array = MarkerArray()

        def add_arrow(pos_yaw, kind, idx):
            x, y, yaw = pos_yaw
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = rospy.Time.now()
            marker.ns = kind
            marker.id = idx
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = x, y, 0.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = np.sin(yaw/2.0)
            marker.pose.orientation.w = np.cos(yaw/2.0)
            marker.scale.x = 0.2  # length
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            if kind=="start_heading":
                marker.color = ColorRGBA(0,0,1,1)
            elif kind=="goal_assembly_point":
                marker.color = ColorRGBA(0,1,0,1)
            else:
                marker.color = ColorRGBA(1,0,0,1)
            marker_array.markers.append(marker)

        for idx in range(len(start_heading)):
            add_arrow(start_heading[idx], "start_heading", idx*3)
            add_arrow(goal_assembly_point[idx], "goal_assembly_point", idx*3+1)
            add_arrow(lane_heading[idx], "lane_heading", idx*3+2)

        self.step_pub.publish(marker_array)


if __name__ == "__main__":
    rospy.init_node("u_path_clearance")
    mode = VisualizerMode.RVIZ

    if mode == VisualizerMode.PYTHON:
        viz = PythonVisualizer(grid_size=10, grid_resolution=1.0, safety_margin=0.5)
    elif mode == VisualizerMode.RVIZ:
        viz = RVIZVisualizer(frame_id="map", point_topic="/clicked_point", compute=True)

    rospy.spin()