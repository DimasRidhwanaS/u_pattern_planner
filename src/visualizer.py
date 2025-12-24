"""
visualizer.py

Main executable node.
Acts as frontend + orchestrator for area_handler and path_handler.

Currently implemented:
- Python visualization
- Click points to define area
- Compute -> area_handler (Step 1-4)
- Visualize original & safe polygon
"""

import matplotlib.pyplot as plt
import numpy as np
import math
# import rospy
from typing import List, Tuple, Callable
from matplotlib.widgets import Button
# from visualization_msgs.msg import Marker, MarkerArray
# from geometry_msgs.msg import Point
# from std_msgs.msg import ColorRGBA


from area_handler import AreaHandler
from path_handler import PathHandler

class VisualizerMode:
    PYTHON = "python"
    RVIZ = "rviz"

# -----------------------------
# Visualizer Class
# -----------------------------
class Visualizer:
    def __init__(
        self,
        mode: str = VisualizerMode.PYTHON,
        grid_size: int = 10,
        grid_resolution: float = 1.0,
        safety_margin: float = 0.5
    ):
        """
        :param mode: 'python' or 'rviz'
        :param grid_size: half-size of grid
        :param grid_resolution: spacing between grid lines
        :param safety_margin: inward offset for safe polygon
        """
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

    # -----------------------------
    # Python Visualization
    # -----------------------------
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

        if self.mode == VisualizerMode.RVIZ:
            area = AreaHandler(
                boundary_points=self.clicked_points,
                safety_margin=self.safety_margin
            )
            self._publish_area_rviz(area)
            return
        
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

    # -----------------------------
    # RViz (stub)
    # -----------------------------
    # def _init_rviz_visualizer(self):
    #     rospy.init_node("coverage_visualizer", anonymous=True)

    #     self.marker_pub = rospy.Publisher(
    #         "/coverage_visualization",
    #         MarkerArray,
    #         queue_size=1,
    #         latch=True
    #     )

    #     rospy.loginfo("[Visualizer] RViz mode initialized")
    #     rospy.spin()

    def _make_color(self, r, g, b, a=1.0):
        return ColorRGBA(r, g, b, a)

    def _make_point(self, x, y, z=0.0):
        p = Point()
        p.x = x
        p.y = y
        p.z = z
        return p

    def _publish_area_rviz(self, area: AreaHandler):
        markers = MarkerArray()
        mid = 0

        # Original polygon
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.ns = "original_polygon"
        m.id = mid; mid += 1
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.05
        m.color = self._make_color(0, 0, 1)

        for x, y in area.get_original_polygon_xy():
            m.points.append(self._make_point(x, y))

        markers.markers.append(m)

        # Safe polygon
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.ns = "safe_polygon"
        m.id = mid; mid += 1
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.05
        m.color = self._make_color(0, 1, 0)

        for x, y in area.get_safe_polygon_xy():
            m.points.append(self._make_point(x, y))

        markers.markers.append(m)

        # Movement arrow
        cx, cy = area.get_centroid()
        nx, ny = area.get_movement_vector()

        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.ns = "movement_direction"
        m.id = mid; mid += 1
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale.x = 1.0   # length
        m.scale.y = 0.15
        m.scale.z = 0.15
        m.color = self._make_color(1, 0, 0)

        m.points.append(self._make_point(cx, cy))
        m.points.append(self._make_point(cx + nx, cy + ny))

        markers.markers.append(m)

        self.marker_pub.publish(markers)




if __name__ == "__main__":
    Visualizer(
        mode=VisualizerMode.PYTHON,
        grid_size=10,
        grid_resolution=1.0,
        safety_margin=0.5
    )
