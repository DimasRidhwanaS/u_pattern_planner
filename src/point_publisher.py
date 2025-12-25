#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PointStamped
import random
import time

def debug_publisher():
    rospy.init_node("debug_point_publisher")
    pub = rospy.Publisher("/published_point", PointStamped, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    points = [
        (0.0, 3.0),
        (0.0, 5.0),
        (7.0, 0.0),
        (5.0, 5.0)
    ]

    while not rospy.is_shutdown():
        for i, (x, y) in enumerate(points):
            msg = PointStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "map"
            msg.point.x = x
            msg.point.y = y
            msg.point.z = 0.0

            rospy.loginfo(f"[DebugPublisher] Publishing point {i}: ({x}, {y})")
            pub.publish(msg)
            rate.sleep()  # Wait for 1 second before next point

if __name__ == "__main__":
    try:
        debug_publisher()
    except rospy.ROSInterruptException:
        pass
