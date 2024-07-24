#!/usr/bin/env python

import rospy
import moveit_commander
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
import sys

class PickAndPlaceNode:
    def __init__(self):
        rospy.init_node('pick_and_place_node', anonymous=True)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/rgbd_camera/rgb/image_raw', Image, self.image_callback)

        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "manipulator"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)

        self.blue_center = None
        self.red_center = None

    def image_callback(self, data):
        # Convert the ROS Image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        # Detect objects
        self.detect_objects(cv_image)

        # If objects are detected, execute pick-and-place tasks
        if self.blue_center and self.red_center:
            self.execute_tasks()

    def detect_objects(self, image):
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for the objects
        blue_lower = np.array([100, 150, 50])
        blue_upper = np.array([140, 255, 255])

        red_lower = np.array([0, 150, 50])
        red_upper = np.array([10, 255, 255])

        # Create masks for the colors
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        red_mask = cv2.inRange(hsv, red_lower, red_upper)

        # Find contours for the blue object
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            self.blue_center = (x + w / 2, y + h / 2)

        # Find contours for the red object
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            self.red_center = (x + w / 2, y + h / 2)

        # Show the result
        cv2.imshow('Detected Objects', image)
        cv2.waitKey(3)

    def execute_tasks(self):
        # Convert pixel coordinates to world coordinates
        # You need to calibrate the camera to get the transformation matrix
        # For simplicity, we assume the camera is directly above the table with no rotation
        scale = 0.001  # Adjust this scale factor based on your setup
        z_height = 0.1  # Fixed height for picking

        pick_pose_blue = Pose()
        pick_pose_blue.position.x = self.blue_center[0] * scale
        pick_pose_blue.position.y = self.blue_center[1] * scale
        pick_pose_blue.position.z = z_height

        place_pose_blue = Pose()
        place_pose_blue.position.x = 0.5
        place_pose_blue.position.y = 0.2
        place_pose_blue.position.z = z_height

        pick_pose_red = Pose()
        pick_pose_red.position.x = self.red_center[0] * scale
        pick_pose_red.position.y = self.red_center[1] * scale
        pick_pose_red.position.z = z_height

        place_pose_red = Pose()
        place_pose_red.position.x = 0.6
        place_pose_red.position.y = 0.3
        place_pose_red.position.z = z_height

        self.pick_and_place(pick_pose_blue, place_pose_blue)
        self.pick_and_place(pick_pose_red, place_pose_red)

    def pick_and_place(self, pick_pose, place_pose):
        self.move_group.set_pose_target(pick_pose)
        plan = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        # Simulate gripper grasping
        rospy.sleep(1)

        self.move_group.set_pose_target(place_pose)
        plan = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        # Simulate gripper releasing
        rospy.sleep(1)

if __name__ == '__main__':
    try:
        node = PickAndPlaceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        moveit_commander.roscpp_shutdown()
