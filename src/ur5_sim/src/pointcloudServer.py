#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import tf2_ros
import tf2_geometry_msgs
from registration_helper import *
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs import point_cloud2
import std_msgs
from open3d_ros_helper import convertCloudFromOpen3dToRos, convertCloudFromRosToOpen3d
import sys

def mask_blue_points(pcd):
    # Extract the RGB values
    rgb = np.asarray(pcd.colors)
    # Mask out the blue points
    mask = (rgb[:,2] > 0.3) & (rgb[:,1] < 0.3) & (rgb[:,0] < 0.3) # Corrected condition to select blue points
    pcd.colors = o3d.utility.Vector3dVector(rgb[mask])
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[mask])

    # remove outliers
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)

    return pcd

def get_cylinder_pose(pcd, vis=False):
    # Create a cylinder cloud of radius 0.03 and height 0.2
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.03, height=0.2)

    source = cylinder.sample_points_uniformly(number_of_points=10000)
    target = pcd

    voxel_size = 0.005

    # 1. Downsample the point clouds and get the FPFH features
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # Coarse registration
    global_registration_result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    # Refine registration
    refine_registration_result = refine_registration(source, target, voxel_size, global_registration_result.transformation)

    # Transform the source point cloud to the target point cloud's frame
    source.transform(refine_registration_result.transformation)

    if vis:
        o3d.visualization.draw_geometries([source, target])

    # Find the centroid of the source point cloud
    centroid = np.mean(np.asarray(source.points), axis=0)  

    return centroid 

def motion_planner(cylinder_pose):
    # Initialize the moveit_commander
    moveit_commander.roscpp_initialize(sys.argv)

    # Initialize the robot commander
    robot = moveit_commander.RobotCommander()

    # Initialize the planning scene interface
    scene = moveit_commander.PlanningSceneInterface()

    # Initialize the move group commander
    group_name = "manipulator"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    # Set the group for the gripper
    gripper = moveit_commander.MoveGroupCommander("gripper")

    # Set the reference frame
    reference_frame = "base_link"
    move_group.set_pose_reference_frame(reference_frame)

    # Remove all world objects
    scene.remove_world_object()

    # Add collision objects for the cylinder and ground plane
    cylinder = geometry_msgs.msg.PoseStamped()
    cylinder.header.frame_id = reference_frame  
    cylinder.pose.position.x = cylinder_pose[0]
    cylinder.pose.position.y = cylinder_pose[1]
    cylinder.pose.position.z = cylinder_pose[2]
    cylinder.pose.orientation.w = 1.0
    scene.add_cylinder("cylinder", cylinder, height=0.2, radius=0.03)

    ground_pose = geometry_msgs.msg.PoseStamped()
    ground_pose.header.frame_id = reference_frame  
    ground_pose.pose.position.x = 0.0
    ground_pose.pose.position.y = 0.0
    ground_pose.pose.position.z = -0.1
    ground_pose.pose.orientation.w = 1.0
    scene.add_plane("ground", ground_pose, normal=(0, 0, 1), offset=0.0)    
    
    # Plan the motion to the cylinder
    move_group.set_start_state_to_current_state()
    move_group.set_goal_position_tolerance(0.1)
    move_group.set_goal_orientation_tolerance(0.1)
    
    # Pick the cylinder
    pose_target = geometry_msgs.msg.PoseStamped()
    pose_target.header.frame_id = reference_frame
    pose_target.pose.position.x = cylinder_pose[0]
    pose_target.pose.position.y = cylinder_pose[1]
    pose_target.pose.position.z = cylinder_pose[2] + 0.2
    pose_target.pose.orientation.x = 0.0
    pose_target.pose.orientation.y = 1.0
    pose_target.pose.orientation.z = 0.0
    pose_target.pose.orientation.w = 0.0

    # print pose target
    print(pose_target)

    move_group.set_pose_target(pose_target)
    move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()

    # Close the gripper
    gripper.set_named_target("closed")
    gripper.go(wait=True)
    gripper.stop()
    gripper.clear_pose_targets()

    # Move the cylinder to a new location
    pose_target.pose.position.x = 0.2
    pose_target.pose.position.y = 0.2
    pose_target.pose.position.z = 0.2
    move_group.set_pose_target(pose_target)

    move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()

    # Open the gripper
    gripper.set_named_target("open")
    gripper.go(wait=True)
    gripper.stop()
    gripper.clear_pose_targets()

    # Return the robot to the home position
    move_group.set_named_target("up")
    move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()

    # Shutdown the moveit_commander
    moveit_commander.roscpp_shutdown()


def point_cloud_callback(msg):
    # setup so that the callback is only triggered once
    global trigger
    if trigger:
        return
    trigger = True

    # Initialize tf2 buffer and listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    # Wait for the transform to become available
    try:
        transform = tf_buffer.lookup_transform('base_link', msg.header.frame_id, msg.header.stamp, rospy.Duration(1.0))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logerr("Could not get transform: %s", e)
        return

    # Transform the point cloud
    cloud_in_base_link = do_transform_cloud(msg, transform)

    rospy.loginfo("Received point cloud with %d points", len(list(pc2.read_points(cloud_in_base_link, field_names=("x", "y", "z"), skip_nans=True))))

    # Convert the point cloud to open3d format

    cloud = convertCloudFromRosToOpen3d(cloud_in_base_link)
    # o3d.visualization.draw_geometries([cloud])

    # Mask out blue points
    cloud = mask_blue_points(cloud)
    pub.publish(convertCloudFromOpen3dToRos(cloud, frame_id="base_link"))
    # o3d.visualization.draw_geometries([cloud])
    
    # Get the pose of the cylinder
    cylinder_pose = get_cylinder_pose(cloud, vis=False)

    rospy.loginfo("Cylinder pose: {}".format(cylinder_pose))
    # Call the moveit_commander to move the robot to the cylinder
    motion_planner(cylinder_pose)


if __name__ == "__main__":
    # setup a global variable to trigger the callback
    trigger = False
    rospy.init_node('cylinder_pickup')
    rospy.Subscriber('/rgbd_camera/depth/points', PointCloud2, point_cloud_callback)
    pub = rospy.Publisher('/mask', PointCloud2, queue_size=10)
    display_trajectory_publisher = rospy.Publisher("/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory, queue_size=20)

    rospy.spin()
