#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point, Quaternion
import numpy as np
from scipy.spatial.transform import Rotation

class TrajectoryPlanner(Node):
    def __init__(self):
        super().__init__('trajectory_planner')
        
        # Parameters
        self.declare_parameter('standoff_distance', 0.15)  # 15cm from surface
        self.declare_parameter('scan_line_spacing', 0.03)  # 3cm between lines
        self.declare_parameter('high_density_spacing', 0.015)  # 1.5cm for complex features
        self.declare_parameter('overlap_percentage', 20.0)  # 20% overlap
        
        # Subscribers
        self.features_sub = self.create_subscription(
            MarkerArray,
            '/perception/detected_features',
            self.features_callback,
            10
        )
        
        # Publishers
        self.trajectory_pub = self.create_publisher(
            PoseArray,
            '/inspection/trajectory',
            10
        )
        
        self.trajectory_viz_pub = self.create_publisher(
            MarkerArray,
            '/inspection/trajectory_visualization',
            10
        )
        
        self.get_logger().info('Trajectory planner started!')
        
        # Storage
        self.detected_features = {
            'plane_center': None,
            'holes': [],
            'edges': [],
        }
        
    def features_callback(self, msg):
        """Process detected features and generate trajectories"""
        
        # Parse features from markers
        self.parse_features(msg)
        
        # Generate inspection trajectory
        trajectory = self.generate_adaptive_trajectory()
        
        if len(trajectory) > 0:
            self.get_logger().info(f'Generated {len(trajectory)} waypoints')
            
            # Publish trajectory
            self.publish_trajectory(trajectory)
            self.publish_trajectory_visualization(trajectory)
        else:
            self.get_logger().warn('No trajectory generated')
    
    def parse_features(self, marker_array):
        """Extract features from marker array"""
        
        self.detected_features = {
            'plane_center': None,
            'holes': [],
            'edges': [],
        }
        
        for marker in marker_array.markers:
            if marker.ns == "plane":
                self.detected_features['plane_center'] = np.array([
                    marker.pose.position.x,
                    marker.pose.position.y,
                    marker.pose.position.z
                ])
            
            elif marker.ns == "holes":
                self.detected_features['holes'].append({
                    'center': np.array([
                        marker.pose.position.x,
                        marker.pose.position.y,
                        marker.pose.position.z
                    ]),
                    'radius': marker.scale.x / 2.0
                })
            
            elif marker.ns == "edges":
                self.detected_features['edges'].append({
                    'center': np.array([
                        marker.pose.position.x,
                        marker.pose.position.y,
                        marker.pose.position.z
                    ])
                })
        
        self.get_logger().info(
            f'Parsed features: plane={self.detected_features["plane_center"] is not None}, '
            f'holes={len(self.detected_features["holes"])}, '
            f'edges={len(self.detected_features["edges"])}'
        )
    
    def generate_adaptive_trajectory(self):
        """Generate adaptive scanning trajectory based on feature complexity"""
        
        all_waypoints = []
        
        # If we have a plane center, generate baseline coverage
        if self.detected_features['plane_center'] is not None:
            plane_waypoints = self.generate_raster_pattern(
                center=self.detected_features['plane_center'],
                width=0.6,
                height=0.5,
                spacing=self.get_parameter('scan_line_spacing').value
            )
            all_waypoints.extend(plane_waypoints)
            self.get_logger().info(f'Generated {len(plane_waypoints)} baseline waypoints')
        
        # Generate high-density scans around holes
        for hole in self.detected_features['holes']:
            hole_waypoints = self.generate_circular_pattern(
                center=hole['center'],
                radius=hole['radius'] + 0.05,  # 5cm margin around hole
                num_points=12,
                layers=2
            )
            all_waypoints.extend(hole_waypoints)
            self.get_logger().info(f'Generated {len(hole_waypoints)} waypoints for hole')
        
        # Generate focused scans for edges
        for edge in self.detected_features['edges']:
            edge_waypoints = self.generate_grid_pattern(
                center=edge['center'],
                size=0.1,
                spacing=self.get_parameter('high_density_spacing').value
            )
            all_waypoints.extend(edge_waypoints)
            self.get_logger().info(f'Generated {len(edge_waypoints)} waypoints for edge')
        
        # Add standoff distance and orientation
        trajectory = []
        standoff = self.get_parameter('standoff_distance').value
        
        for waypoint in all_waypoints:
            pose = self.create_inspection_pose(waypoint, standoff)
            trajectory.append(pose)
        
        return trajectory
    
    def generate_raster_pattern(self, center, width, height, spacing):
        """Generate raster (back-and-forth) scanning pattern"""
        waypoints = []
        
        # Calculate number of scan lines
        num_lines = int(height / spacing) + 1
        
        for i in range(num_lines):
            y_offset = -height/2 + i * spacing
            
            if i % 2 == 0:  # Left to right
                x_start = -width/2
                x_end = width/2
            else:  # Right to left
                x_start = width/2
                x_end = -width/2
            
            # Add waypoints along scan line
            num_points = int(width / spacing) + 1
            for j in range(num_points):
                t = j / (num_points - 1)
                x_offset = x_start + t * (x_end - x_start)
                
                waypoint = center + np.array([x_offset, y_offset, 0])
                waypoints.append(waypoint)
        
        return waypoints
    
    def generate_circular_pattern(self, center, radius, num_points=12, layers=2):
        """Generate circular scanning pattern around a feature"""
        waypoints = []
        
        for layer in range(layers):
            r = radius * (layer + 1) / layers
            
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                x_offset = r * np.cos(angle)
                y_offset = r * np.sin(angle)
                
                waypoint = center + np.array([0, x_offset, y_offset])
                waypoints.append(waypoint)
        
        return waypoints
    
    def generate_grid_pattern(self, center, size, spacing):
        """Generate dense grid pattern for complex features"""
        waypoints = []
        
        num_points = int(size / spacing) + 1
        
        for i in range(num_points):
            for j in range(num_points):
                x_offset = -size/2 + i * spacing
                y_offset = -size/2 + j * spacing
                
                waypoint = center + np.array([0, x_offset, y_offset])
                waypoints.append(waypoint)
        
        return waypoints
    
    def create_inspection_pose(self, point, standoff_distance):
        """Create pose with standoff distance and camera pointing at surface"""
        
        pose = Pose()
        
        # Position: point with standoff offset (assuming surface normal is in -X direction)
        pose.position.x = point[0] - standoff_distance
        pose.position.y = point[1]
        pose.position.z = point[2]
        
        # Orientation: camera looking at surface (perpendicular)
        # Camera frame: Z forward, X right, Y down
        # We want Z pointing toward surface (+X in world frame)
        
        # Rotation: align camera Z with world +X
        r = Rotation.from_euler('xyz', [0, 90, 0], degrees=True)
        quat = r.as_quat()  # [x, y, z, w]
        
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        
        return pose
    
    def publish_trajectory(self, trajectory):
        """Publish trajectory as PoseArray"""
        
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'world'
        pose_array.poses = trajectory
        
        self.trajectory_pub.publish(pose_array)
    
    def publish_trajectory_visualization(self, trajectory):
        """Publish trajectory visualization markers"""
        
        marker_array = MarkerArray()
        
        # Waypoint markers
        waypoint_marker = Marker()
        waypoint_marker.header.stamp = self.get_clock().now().to_msg()
        waypoint_marker.header.frame_id = 'world'
        waypoint_marker.ns = 'waypoints'
        waypoint_marker.id = 0
        waypoint_marker.type = Marker.SPHERE_LIST
        waypoint_marker.action = Marker.ADD
        
        waypoint_marker.scale.x = 0.02
        waypoint_marker.scale.y = 0.02
        waypoint_marker.scale.z = 0.02
        
        waypoint_marker.color.r = 0.0
        waypoint_marker.color.g = 0.0
        waypoint_marker.color.b = 1.0
        waypoint_marker.color.a = 0.8
        
        for pose in trajectory:
            point = Point()
            point.x = pose.position.x
            point.y = pose.position.y
            point.z = pose.position.z
            waypoint_marker.points.append(point)
        
        marker_array.markers.append(waypoint_marker)
        
        # Path line
        path_marker = Marker()
        path_marker.header.stamp = self.get_clock().now().to_msg()
        path_marker.header.frame_id = 'world'
        path_marker.ns = 'path'
        path_marker.id = 1
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        
        path_marker.scale.x = 0.005  # Line width
        
        path_marker.color.r = 0.0
        path_marker.color.g = 1.0
        path_marker.color.b = 1.0
        path_marker.color.a = 0.5
        
        for pose in trajectory:
            point = Point()
            point.x = pose.position.x
            point.y = pose.position.y
            point.z = pose.position.z
            path_marker.points.append(point)
        
        marker_array.markers.append(path_marker)
        
        # Camera orientation arrows
        for i, pose in enumerate(trajectory[::5]):  # Every 5th pose to avoid clutter
            arrow = Marker()
            arrow.header.stamp = self.get_clock().now().to_msg()
            arrow.header.frame_id = 'world'
            arrow.ns = 'orientations'
            arrow.id = i + 100
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            
            arrow.pose = pose
            
            arrow.scale.x = 0.05  # Arrow length
            arrow.scale.y = 0.005
            arrow.scale.z = 0.005
            
            arrow.color.r = 1.0
            arrow.color.g = 0.5
            arrow.color.b = 0.0
            arrow.color.a = 0.7
            
            marker_array.markers.append(arrow)
        
        self.trajectory_viz_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()