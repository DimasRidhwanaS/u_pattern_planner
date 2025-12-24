#!/usr/bin/env python3

import math
import threading
import traceback
from typing import List, Tuple
import rospy

from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from area_handler import AreaHandler
from path_handler import PathHandler


class VisualizerBase:
    def __init__(self, safety_margin: float):
        self.safety_margin = safety_margin
        self.clicked_points = []

    def set_points(self, points):
        self.clicked_points = points

    def compute(self):
        if len(self.clicked_points) < 4:
            raise ValueError("minimum 4 point required")
        if len(self.clicked_points) > 4:
            raise ValueError("too much point")

        area = AreaHandler(
            boundary_points=self.clicked_points,
            safety_margin=self.safety_margin
        )

        path = PathHandler(
            safe_polygon=area.get_safe_polygon(),
            lane_direction=area.get_lane_direction_vector(),
            lane_spacing=1.0
        )

        return area, path


class PythonVisualizer(VisualizerBase):
    def __init__(self, grid_size, grid_resolution, safety_margin):
        super().__init__(safety_margin)

        self.grid_size = grid_size
        self.grid_resolution = grid_resolution

        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-grid_size, grid_size)
        self.ax.set_ylim(-grid_size, grid_size)
        self.ax.grid(True)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._add_buttons()

        plt.show()

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return

        self.clicked_points.append((event.xdata, event.ydata))
        self.ax.plot(event.xdata, event.ydata, "bo")
        self.fig.canvas.draw_idle()

    def _add_buttons(self):
        ax = plt.axes([0.8, 0.02, 0.15, 0.06])
        Button(ax, "Compute").on_clicked(self._on_compute)

    def _on_compute(self, _):
        try:
            area, path = self.compute()
        except Exception as e:
            print(e)
            return

        self._draw_polygon(area.get_original_polygon_xy(), "b-")
        self._draw_polygon(area.get_safe_polygon_xy(), "g-")

    def _draw_polygon(self, xy, style):
        xs, ys = zip(*xy)
        self.ax.plot(xs, ys, style, linewidth=2)
        self.fig.canvas.draw_idle()


class RVizVisualizer(VisualizerBase):
    def __init__(self, safety_margin, frame_id="map"):
        super().__init__(safety_margin)

        self.frame_id = frame_id

        rospy.init_node("coverage_visualizer", anonymous=True)
        self.marker_pub = rospy.Publisher(
            "/coverage_visualization",
            MarkerArray,
            queue_size=1,
            latch=True
        )

    def publish(self):
        try:
            area, path = self.compute()
        except Exception as e:
            rospy.logerr(str(e))
            return

        self._publish_area(area)

    def _publish_area(self, area):
        markers = MarkerArray()
        mid = 0

        def line_strip(ns, points, color):
            nonlocal mid
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = rospy.Time.now()
            m.ns = ns
            m.id = mid; mid += 1
            m.type = Marker.LINE_STRIP
            m.scale.x = 0.05
            m.color = color
            for x, y in points:
                m.points.append(Point(x=x, y=y, z=0))
            markers.markers.append(m)

        line_strip("original", area.get_original_polygon_xy(), ColorRGBA(0,0,1,1))
        line_strip("safe", area.get_safe_polygon_xy(), ColorRGBA(0,1,0,1))

        self.marker_pub.publish(markers)



class VisualizerMode:
    PYTHON = "python"
    RVIZ = "rviz"


def main():
    mode = rospy.get_param("~mode", "python")

    if mode == VisualizerMode.PYTHON:
        PythonVisualizer(
            grid_size=10,
            grid_resolution=1.0,
            safety_margin=0.5
        )

    elif mode == VisualizerMode.RVIZ:
        viz = RVizVisualizer(safety_margin=0.5)
        viz.set_points([...])  # later from topic/service
        viz.publish()


if __name__ == "__main__":
    main()
