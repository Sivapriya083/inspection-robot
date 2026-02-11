#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from sklearn.cluster import DBSCAN
import open3d as o3d

class PerceptionPipeline(Node):
    def __init__(self):
        super().__init__('perception_pipeline')
        
        # Subscribers
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/camera/depth/points',
            self.pointcloud_callback,
            10
        )
        
        # Publishers
        self.feature_pub = self.create_publisher(MarkerArray, '/detected_features', 10)
        self.inspection_points_pub = self.create_publisher(MarkerArray, '/inspection_points', 10)
        
        # Parameters - FIXED VALUES
        self.voxel_size = 0.01  # 1cm voxel grid
        self.min_points_cluster = 50
        self.standoff_distance = 0.10  # REDUCED from 0.30 to 0.10 (10cm)
        
        # Reduced density parameters
        self.hole_circle_points = 4
        self.plane_spacing = 0.15
        self.max_inspection_points = 50
        
        # Processing throttle
        self.last_process_time = self.get_clock().now()
        self.process_interval = 3.0
        
        self.get_logger().info('Perception pipeline initialized!')
        self.get_logger().info(f'Standoff distance: {self.standoff_distance}m')

    def auto_set_workspace(self, points):
        if len(points) < 10:
            return
        
        # Get point cloud bounds
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        
        # Add margin (10cm on each side)
        margin = 0.1
        self.workspace_min = min_bounds - margin
        self.workspace_max = max_bounds + margin
        
        # Update reach distances
        distances = np.linalg.norm(points, axis=1)
        self.min_reach_distance = max(0.1, np.min(distances) - margin)
        self.max_reach_distance = min(2.0, np.max(distances) + margin)
        
        self.get_logger().info(f'Auto workspace: X[{self.workspace_min[0]:.2f}, {self.workspace_max[0]:.2f}]')
        self.get_logger().info(f'                Y[{self.workspace_min[1]:.2f}, {self.workspace_max[1]:.2f}]')
        self.get_logger().info(f'                Z[{self.workspace_min[2]:.2f}, {self.workspace_max[2]:.2f}]')
        self.get_logger().info(f'Reach: {self.min_reach_distance:.2f}m to {self.max_reach_distance:.2f}m')
    
    def pointcloud_callback(self, msg):
        """Process incoming point cloud"""
        
        # Throttle processing
        current_time = self.get_clock().now()
        elapsed = (current_time - self.last_process_time).nanoseconds / 1e9
        
        if elapsed < self.process_interval:
            return
        
        self.last_process_time = current_time
        
        self.get_logger().info('Processing point cloud...')
        
        # Convert ROS PointCloud2 to numpy array
        points = self.pointcloud2_to_array(msg)
        
        if len(points) < 100:
            self.get_logger().warn(f'Too few valid points: {len(points)}')
            return
        
        # Step 1: Voxel grid filtering
        filtered_points = self.voxel_grid_filter(points)
        
        if len(filtered_points) < 50:
            self.get_logger().warn(f'Too few points after filtering: {len(filtered_points)}')
            return
        
        # Auto-set workspace
        self.auto_set_workspace(filtered_points)
        
        self.get_logger().info(f'Points after filtering: {len(filtered_points)}')
        
        # Step 2: Detect surface features
        features = self.detect_features(filtered_points)
        
        # Step 3: Generate inspection points
        inspection_points = self.generate_inspection_points(features)

        if len(inspection_points) > 0:
            positions = np.array([p['position'] for p in inspection_points])
            self.get_logger().info('=== INSPECTION POINTS DEBUG ===')
            self.get_logger().info(f'Total points: {len(positions)}')
            self.get_logger().info(f'X range: [{positions[:,0].min():.3f}, {positions[:,0].max():.3f}]')
            self.get_logger().info(f'Y range: [{positions[:,1].min():.3f}, {positions[:,1].max():.3f}]')
            self.get_logger().info(f'Z range: [{positions[:,2].min():.3f}, {positions[:,2].max():.3f}]')
            self.get_logger().info(f'Sample point 1: {positions[0]}')
            if len(positions) > 5:
                self.get_logger().info(f'Sample point 5: {positions[4]}')
            
        # Step 4: Publish visualization
        self.publish_features(features)
        self.publish_inspection_points(inspection_points)
        
        self.get_logger().info(f'Generated {len(inspection_points)} inspection points')
    
    def pointcloud2_to_array(self, cloud_msg):
        """Convert PointCloud2 message to numpy array"""
        points = []
        
        for point in pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z")):
            if (np.isfinite(point[0]) and np.isfinite(point[1]) and np.isfinite(point[2])):
                points.append([point[0], point[1], point[2]])
        
        if len(points) == 0:
            self.get_logger().warn('No valid points after filtering inf/nan')
            return np.array([]).reshape(0, 3)
        
        points_array = np.array(points)
        
        # Filter by distance
        distances = np.linalg.norm(points_array, axis=1)
        valid_mask = (distances > 0.05) & (distances < 1.5)
        filtered_points = points_array[valid_mask]
        
        self.get_logger().info(f'Valid points: {len(filtered_points)} / {len(points_array)}')
        return filtered_points
    
    def voxel_grid_filter(self, points):
        """Downsample point cloud"""
        if len(points) < 10:
            return points
        
        if not np.all(np.isfinite(points)):
            valid_mask = np.all(np.isfinite(points), axis=1)
            points = points[valid_mask]
            if len(points) < 10:
                return np.array([]).reshape(0, 3)
        
        point_min = np.min(points, axis=0)
        point_max = np.max(points, axis=0)
        point_range = point_max - point_min
        
        self.get_logger().info(f'Point cloud: {len(points)} points')
        self.get_logger().info(f'  Range X: [{point_min[0]:.3f}, {point_max[0]:.3f}]')
        self.get_logger().info(f'  Range Y: [{point_min[1]:.3f}, {point_max[1]:.3f}]')
        self.get_logger().info(f'  Range Z: [{point_min[2]:.3f}, {point_max[2]:.3f}]')
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        try:
            downsampled = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            result = np.asarray(downsampled.points)
            self.get_logger().info(f'✓ Downsampled to {len(result)} points')
            return result
        except RuntimeError as e:
            self.get_logger().error(f'Voxel downsampling failed: {e}')
            point_range_max = np.max(point_range)
            
            if point_range_max == 0 or not np.isfinite(point_range_max):
                return points
            
            adaptive_voxel = max(0.02, point_range_max / 50.0)
            try:
                downsampled = pcd.voxel_down_sample(voxel_size=adaptive_voxel)
                result = np.asarray(downsampled.points)
                return result
            except:
                return points
    
    def detect_features(self, points):
        """Detect surface features"""
        features = {
            'edges': [],
            'holes': [],
            'planes': [],
            'complex_regions': []
        }
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        try:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
        except:
            self.get_logger().warn('Normal estimation failed')
            return features
        
        edges = self.detect_edges(pcd)
        features['edges'] = edges
        
        planes = self.detect_planes(pcd)
        features['planes'] = planes
        
        holes = self.detect_holes(points)
        features['holes'] = holes
        
        complex_regions = self.detect_complex_regions(pcd)
        features['complex_regions'] = complex_regions
        
        self.get_logger().info(f'Features: {len(edges)} edges, {len(holes)} holes, {len(planes)} planes, {len(complex_regions)} complex')
        
        return features
    
    def detect_edges(self, pcd):
        """Detect edges"""
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        if len(normals) == 0:
            return []
        
        edges = []
        from sklearn.neighbors import NearestNeighbors
        
        n_neighbors = min(10, len(points) - 1)
        if n_neighbors < 3:
            return []
        
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(points)
        
        for i, point in enumerate(points):
            distances, indices = nbrs.kneighbors([point])
            neighbor_normals = normals[indices[0]]
            normal_std = np.std(neighbor_normals, axis=0)
            variation = np.linalg.norm(normal_std)
            
            if variation > 0.3:
                edges.append({
                    'position': point,
                    'normal': normals[i],
                    'complexity': variation
                })
        
        return edges
    
    def detect_planes(self, pcd):
        """Detect planar regions"""
        planes = []
        
        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.01,
                ransac_n=3,
                num_iterations=1000
            )
            
            if len(inliers) > 100:
                inlier_cloud = pcd.select_by_index(inliers)
                centroid = np.mean(np.asarray(inlier_cloud.points), axis=0)
                
                planes.append({
                    'centroid': centroid,
                    'normal': plane_model[:3],
                    'points': np.asarray(inlier_cloud.points)
                })
        except:
            pass
        
        return planes
    
    def detect_holes(self, points):
        """Detect holes"""
        holes = []
        
        if len(points) < self.min_points_cluster:
            return holes
        
        try:
            clustering = DBSCAN(eps=0.02, min_samples=min(self.min_points_cluster, len(points))).fit(points)
            labels = clustering.labels_
            
            for label in set(labels):
                if label == -1:
                    continue
                
                cluster_points = points[labels == label]
                
                if len(cluster_points) < 20:
                    continue
                
                centroid = np.mean(cluster_points, axis=0)
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                
                if np.std(distances) < 0.005:
                    holes.append({
                        'centroid': centroid,
                        'radius': np.mean(distances),
                        'points': cluster_points
                    })
        except:
            pass
        
        return holes
    
    def detect_complex_regions(self, pcd):
        """Detect high curvature regions"""
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        if len(normals) == 0:
            return []
        
        complex_regions = []
        from sklearn.neighbors import NearestNeighbors
        
        n_neighbors = min(15, len(points) - 1)
        if n_neighbors < 3:
            return []
        
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(points)
        
        for i, point in enumerate(points):
            distances, indices = nbrs.kneighbors([point])
            neighbor_normals = normals[indices[0]]
            curvature = np.linalg.norm(np.std(neighbor_normals, axis=0))
            
            if curvature > 0.4:
                complex_regions.append({
                    'position': point,
                    'curvature': curvature,
                    'normal': normals[i]
                })
        
        return complex_regions
    
    def generate_inspection_points(self, features):
        """Generate inspection points"""
        inspection_points = []
        
        # Sample edges
        edge_sample = features['edges'][::2]
        for edge in edge_sample:
            point = self.create_inspection_point(
                edge['position'],
                edge['normal'],
                density='high'
            )
            inspection_points.append(point)
        
        # Holes with circular pattern
        for hole in features['holes']:
            circle_points = self.generate_circular_pattern(
                hole['centroid'],
                hole['radius'] * 1.2,
                num_points=self.hole_circle_points
            )
            inspection_points.extend(circle_points)
        
        # Sample complex regions
        complex_sample = features['complex_regions'][::3]
        for region in complex_sample:
            point = self.create_inspection_point(
                region['position'],
                region['normal'],
                density='medium'
            )
            inspection_points.append(point)
        
        # Planar regions with grid
        for plane in features['planes']:
            grid_points = self.generate_grid_pattern(
                plane['points'],
                spacing=self.plane_spacing
            )
            inspection_points.extend(grid_points)
        
        # Filter and sample
        inspection_points = self.filter_reachable_points(inspection_points)
        inspection_points = self.smart_sample_points(inspection_points, self.max_inspection_points)
        
        return inspection_points
    
    def filter_reachable_points(self, inspection_points):
        """Filter points to reachable workspace"""
        reachable = []
        
        for point in inspection_points:
            pos = np.array(point['position'])
            
            # Check workspace bounds
            adjusted_min = self.workspace_min + np.array([0.1, 0.1, 0.1])
            adjusted_max = self.workspace_max - np.array([0.1, 0.1, 0.1])
            
            if not (np.all(pos >= adjusted_min) and np.all(pos <= adjusted_max)):
                continue
            
            # Check reach distance
            distance = np.linalg.norm(pos)
            if distance < self.min_reach_distance or distance > self.max_reach_distance:
                continue
            
            # Check height
            if pos[2] < -0.5 or pos[2] > 0.5:
                continue
            
            # Check lateral distance
            xy_distance = np.linalg.norm(pos[:2])
            if xy_distance > 0.9 * self.max_reach_distance:
                continue
            
            # Check orientation
            normal = point['orientation']
            if abs(normal[2]) > 0.9:
                continue
            
            reachable.append(point)
        
        filtered_count = len(inspection_points) - len(reachable)
        self.get_logger().info(
            f'Filtered: {len(reachable)}/{len(inspection_points)} reachable ({filtered_count} removed)'
        )
        
        return reachable
    
    def smart_sample_points(self, inspection_points, target_count):
        """Sample points by priority"""
        if len(inspection_points) <= target_count:
            return inspection_points
        
        high_priority = [p for p in inspection_points if p['density'] == 'high']
        medium_priority = [p for p in inspection_points if p['density'] == 'medium']
        low_priority = [p for p in inspection_points if p['density'] == 'low']
        
        high_count = min(len(high_priority), target_count // 2)
        medium_count = min(len(medium_priority), target_count // 3)
        low_count = target_count - high_count - medium_count
        
        sampled = []
        
        if len(high_priority) > high_count:
            step = len(high_priority) // high_count
            sampled.extend(high_priority[::step][:high_count])
        else:
            sampled.extend(high_priority)
        
        if len(medium_priority) > medium_count:
            step = len(medium_priority) // medium_count
            sampled.extend(medium_priority[::step][:medium_count])
        else:
            sampled.extend(medium_priority)
        
        if len(low_priority) > low_count and low_count > 0:
            step = len(low_priority) // low_count
            sampled.extend(low_priority[::step][:low_count])
        else:
            sampled.extend(low_priority[:low_count])
        
        return sampled
    
    def create_inspection_point(self, surface_point, normal, density='medium'):
        """Create inspection point with standoff"""
        normal_mag = np.linalg.norm(normal)
        if normal_mag > 0:
            normal_normalized = normal / normal_mag
        else:
            normal_normalized = np.array([0, 0, -1])
        
        inspection_pos = surface_point + normal_normalized * self.standoff_distance
        
        return {
            'position': inspection_pos,
            'orientation': normal_normalized,
            'density': density,
            'surface_point': surface_point
        }
    
    def generate_circular_pattern(self, center, radius, num_points):
        """Generate circular scan pattern"""
        points = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            
            normal = (center - np.array([x, y, z]))
            normal_mag = np.linalg.norm(normal)
            if normal_mag > 0:
                normal = normal / normal_mag
            else:
                normal = np.array([0, 0, -1])
            
            points.append(self.create_inspection_point(
                np.array([x, y, z]),
                normal,
                density='high'
            ))
        
        return points
    
    def generate_grid_pattern(self, plane_points, spacing):
        """Generate grid for planar surface"""
        if len(plane_points) == 0:
            return []
        
        min_vals = np.min(plane_points, axis=0)
        max_vals = np.max(plane_points, axis=0)
        
        points = []
        
        x_range = np.arange(min_vals[0], max_vals[0], spacing)
        y_range = np.arange(min_vals[1], max_vals[1], spacing)
        
        if len(x_range) == 0 or len(y_range) == 0:
            return []
        
        for x in x_range:
            for y in y_range:
                z = np.mean(plane_points[:, 2])
                normal = np.array([0, 0, -1])
                
                points.append(self.create_inspection_point(
                    np.array([x, y, z]),
                    normal,
                    density='low'
                ))
        
        return points
    
    def publish_features(self, features):
        """Publish feature visualization"""
        marker_array = MarkerArray()
        marker_id = 0
        
        for edge in features['edges']:
            marker = self.create_marker(
                marker_id, edge['position'],
                color=[1.0, 0.0, 0.0, 1.0], scale=0.01
            )
            marker_array.markers.append(marker)
            marker_id += 1
        
        for hole in features['holes']:
            marker = self.create_marker(
                marker_id, hole['centroid'],
                color=[0.0, 0.0, 1.0, 1.0], scale=hole['radius']
            )
            marker.type = Marker.CYLINDER
            marker_array.markers.append(marker)
            marker_id += 1
        
        for region in features['complex_regions']:
            marker = self.create_marker(
                marker_id, region['position'],
                color=[1.0, 1.0, 0.0, 1.0], scale=0.008
            )
            marker_array.markers.append(marker)
            marker_id += 1
        
        self.feature_pub.publish(marker_array)
    
    def publish_inspection_points(self, inspection_points):
        """Publish inspection points"""
        marker_array = MarkerArray()
        
        for i, point in enumerate(inspection_points):
            color_map = {
                'high': [1.0, 0.0, 0.0, 0.7],
                'medium': [1.0, 0.5, 0.0, 0.7],
                'low': [0.0, 1.0, 0.0, 0.7]
            }
            
            color = color_map.get(point['density'], [0.5, 0.5, 0.5, 0.7])
            
            marker = self.create_marker(
                i, point['position'],
                color=color, scale=0.015
            )
            marker.type = Marker.SPHERE
            
            marker_array.markers.append(marker)
        
        self.inspection_points_pub.publish(marker_array)
    
    def create_marker(self, marker_id, position, color, scale):
        """Create visualization marker"""
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        
        return marker

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionPipeline()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()