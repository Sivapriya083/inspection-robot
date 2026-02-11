#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints, PositionConstraint, OrientationConstraint,
    BoundingVolume, MoveItErrorCodes, RobotState
)
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import MarkerArray
import time
import numpy as np

class MoveItExecutor(Node):
    def __init__(self):
        super().__init__('moveit_executor')
        
        self.get_logger().info('Waiting for MoveGroup action server...')
        
        self._action_client = ActionClient(
            self,
            MoveGroup,
            '/move_action'
        )
        
        if not self._action_client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error('MoveGroup action server not available!')
            self.get_logger().error('Make sure to launch MoveIt first:')
            self.get_logger().error('  ros2 launch fanuc_moveit_config move_group.launch.py')
            raise RuntimeError('MoveGroup not available')
        
        self.get_logger().info('✓ Connected to MoveGroup')
        
        # Subscribe to inspection points
        self.inspection_sub = self.create_subscription(
            MarkerArray,
            '/inspection_points',
            self.inspection_callback,
            10
        )
        
        # FANUC-SPECIFIC PARAMETERS
        self.planning_group = "manipulator"
        self.end_effector_link = "tool0"
        
        # ADJUSTED PARAMETERS - More conservative for better success
        self.max_velocity_scaling = 0.1         # Slightly faster than before
        self.max_acceleration_scaling = 0.1
        self.position_tolerance = 0.05          # REDUCED from 0.20 to 0.05 (5cm)
        self.orientation_tolerance = 0.5        # REDUCED from 1.5 to 0.5 (~30 degrees)
        
        # State
        self.inspection_trajectory = []
        self.current_idx = 0
        self.is_executing = False
        self.skip_unreachable = True
        
        self.get_logger().info('MoveIt Executor ready!')
        self.get_logger().info(f'Planning group: {self.planning_group}')
        self.get_logger().info(f'End effector: {self.end_effector_link}')
        self.get_logger().info(f'Position tolerance: {self.position_tolerance}m')
        self.get_logger().info(f'Orientation tolerance: {self.orientation_tolerance} rad')
    
    def inspection_callback(self, msg):
        """Receive inspection points"""
        if self.is_executing:
            self.get_logger().warn('Already executing, ignoring new trajectory')
            return
        
        if len(msg.markers) == 0:
            return
        
        # Limit trajectory size
        max_waypoints = 10
        markers = msg.markers[:max_waypoints]
        
        self.get_logger().info(f'Received {len(markers)} inspection points (limited from {len(msg.markers)})')
        
        # Convert to poses
        self.inspection_trajectory = []
        for marker in markers:
            pose = PoseStamped()
            pose.header.frame_id = "world"
            pose.pose = marker.pose
            self.inspection_trajectory.append(pose)
        
        # Start execution
        self.current_idx = 0
        self.is_executing = True
        self.execute_next_waypoint()
    
    def execute_next_waypoint(self):
        """Execute single waypoint"""
        if self.current_idx >= len(self.inspection_trajectory):
            self.get_logger().info('✓ All waypoints executed!')
            self.is_executing = False
            return
        
        target = self.inspection_trajectory[self.current_idx]
        
        self.get_logger().info(
            f'Planning waypoint {self.current_idx + 1}/{len(self.inspection_trajectory)}'
        )
        self.get_logger().info(
            f'Target: pos=({target.pose.position.x:.3f}, {target.pose.position.y:.3f}, {target.pose.position.z:.3f})'
        )
        
        # Create goal
        goal = MoveGroup.Goal()
        goal.request.group_name = self.planning_group
        goal.request.num_planning_attempts = 20  # Increased from 10
        goal.request.allowed_planning_time = 10.0  # Increased from 5.0
        goal.request.max_velocity_scaling_factor = self.max_velocity_scaling
        goal.request.max_acceleration_scaling_factor = self.max_acceleration_scaling
        
        # Set workspace bounds
        goal.request.workspace_parameters.header.frame_id = "world"
        goal.request.workspace_parameters.min_corner.x = -2.0
        goal.request.workspace_parameters.min_corner.y = -2.0
        goal.request.workspace_parameters.min_corner.z = -1.0
        goal.request.workspace_parameters.max_corner.x = 2.0
        goal.request.workspace_parameters.max_corner.y = 2.0
        goal.request.workspace_parameters.max_corner.z = 2.0
        
        # NEW: Use pose constraint with BOTH position and orientation
        # This helps IK solver find valid solutions
        goal.request.goal_constraints.append(
            self.create_pose_constraint_with_orientation(target)
        )
        
        # Plan and execute
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 5
        
        # Send goal
        send_goal_future = self._action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.goal_response_callback)
    
    def create_pose_constraint_with_orientation(self, target_pose):
        """Create constraint with BOTH position and orientation (better for IK)"""
        constraint = Constraints()
        
        # Position constraint
        pos_c = PositionConstraint()
        pos_c.header = target_pose.header
        pos_c.link_name = self.end_effector_link
        
        # Tighter bounding box
        pos_c.constraint_region.primitives.append(SolidPrimitive())
        pos_c.constraint_region.primitives[0].type = SolidPrimitive.BOX
        pos_c.constraint_region.primitives[0].dimensions = [
            self.position_tolerance * 2,
            self.position_tolerance * 2,
            self.position_tolerance * 2
        ]
        
        pos_c.constraint_region.primitive_poses.append(target_pose.pose)
        pos_c.weight = 1.0
        constraint.position_constraints.append(pos_c)
        
        # Orientation constraint - helps IK solver
        orient_c = OrientationConstraint()
        orient_c.header = target_pose.header
        orient_c.link_name = self.end_effector_link
        
        # FIX: Ensure quaternion is normalized
        q = target_pose.pose.orientation
        q_norm = np.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        
        if q_norm < 0.01:  # Invalid quaternion
            # Use default "pointing down" orientation
            orient_c.orientation = Quaternion()
            orient_c.orientation.w = 1.0
            self.get_logger().warn('Invalid target orientation, using default')
        else:
            # Normalize quaternion
            orient_c.orientation = Quaternion()
            orient_c.orientation.x = q.x / q_norm
            orient_c.orientation.y = q.y / q_norm
            orient_c.orientation.z = q.z / q_norm
            orient_c.orientation.w = q.w / q_norm
        
        orient_c.absolute_x_axis_tolerance = self.orientation_tolerance
        orient_c.absolute_y_axis_tolerance = self.orientation_tolerance
        orient_c.absolute_z_axis_tolerance = self.orientation_tolerance
        orient_c.weight = 0.5  # Medium priority (not too strict)
        constraint.orientation_constraints.append(orient_c)
        
        return constraint
    
    def create_position_only_goal(self, target_pose):
        """BACKUP: Position-only constraint (if orientation causes issues)"""
        constraint = Constraints()
        
        pos_c = PositionConstraint()
        pos_c.header = target_pose.header
        pos_c.link_name = self.end_effector_link
        
        pos_c.constraint_region.primitives.append(SolidPrimitive())
        pos_c.constraint_region.primitives[0].type = SolidPrimitive.BOX
        pos_c.constraint_region.primitives[0].dimensions = [
            self.position_tolerance * 2,
            self.position_tolerance * 2,
            self.position_tolerance * 2
        ]
        
        pos_c.constraint_region.primitive_poses.append(target_pose.pose)
        pos_c.weight = 1.0
        constraint.position_constraints.append(pos_c)
        
        return constraint
    
    def goal_response_callback(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error(f'Goal rejected for waypoint {self.current_idx + 1}')
            self.handle_failure()
            return
        
        self.get_logger().info('Goal accepted, planning...')
        
        # Wait for result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)
    
    def result_callback(self, future):
        """Handle execution result"""
        result = future.result().result
        error_code = result.error_code.val
        
        # Map error codes to names for better debugging
        error_names = {
            1: "SUCCESS",
            -1: "FAILURE",
            -2: "PLANNING_FAILED",
            -3: "INVALID_MOTION_PLAN",
            -4: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
            -5: "CONTROL_FAILED",
            -6: "UNABLE_TO_AQUIRE_SENSOR_DATA",
            -7: "TIMED_OUT",
            -10: "NO_IK_SOLUTION",
            -12: "INVALID_GROUP_NAME",
            -21: "GOAL_IN_COLLISION",
            -22: "GOAL_VIOLATES_PATH_CONSTRAINTS",
            -23: "GOAL_CONSTRAINTS_VIOLATED",
            -31: "START_STATE_IN_COLLISION",
            -32: "START_STATE_VIOLATES_PATH_CONSTRAINTS"
        }
        
        error_name = error_names.get(error_code, f"UNKNOWN({error_code})")
        
        if error_code == MoveItErrorCodes.SUCCESS:
            self.get_logger().info(f'✓ Waypoint {self.current_idx + 1} completed successfully')
            self.current_idx += 1
            time.sleep(0.5)
            self.execute_next_waypoint()
        else:
            self.get_logger().warn(
                f'⚠ Waypoint {self.current_idx + 1} failed: {error_name}'
            )
            self.handle_failure()
    
    def handle_failure(self):
        """Handle planning/execution failure"""
        if self.skip_unreachable:
            self.get_logger().info('Skipping to next waypoint...')
            self.current_idx += 1
            time.sleep(0.3)
            self.execute_next_waypoint()
        else:
            self.get_logger().error('Stopping execution due to failure')
            self.is_executing = False

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = MoveItExecutor()
        rclpy.spin(node)
    except RuntimeError as e:
        print(f'Error: {e}')
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()