#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

class PlanningScenePublisher(Node):
    def __init__(self):
        super().__init__('planning_scene_publisher')
        
        self.scene_pub = self.create_publisher(
            PlanningScene,
            '/planning_scene',
            10
        )
        
        # Wait a bit for MoveIt to start
        self.create_timer(2.0, self.publish_scene_once)
    
    def publish_scene_once(self):
        scene_msg = PlanningScene()
        scene_msg.is_diff = True
        
        # Main panel (blue)
        panel = CollisionObject()
        panel.header.frame_id = 'world'
        panel.id = 'inspection_panel'
        
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.02, 1.2, 1.2]  # x, y, z
        
        pose = Pose()
        pose.position.x = 0.8
        pose.position.y = 0.0
        pose.position.z = 0.7
        pose.orientation.w = 1.0
        
        panel.primitives.append(box)
        panel.primitive_poses.append(pose)
        panel.operation = CollisionObject.ADD
        
        scene_msg.world.collision_objects.append(panel)
        
        # Raised edge (red)
        rib = CollisionObject()
        rib.header.frame_id = 'world'
        rib.id = 'inspection_rib'
        
        rib_box = SolidPrimitive()
        rib_box.type = SolidPrimitive.BOX
        rib_box.dimensions = [0.05, 0.8, 0.15]
        
        rib_pose = Pose()
        rib_pose.position.x = 0.79
        rib_pose.position.y = -0.3
        rib_pose.position.z = 0.85
        rib_pose.orientation.w = 1.0
        
        rib.primitives.append(rib_box)
        rib.primitive_poses.append(rib_pose)
        rib.operation = CollisionObject.ADD
        
        scene_msg.world.collision_objects.append(rib)
        
        # Add other objects similarly...
        
        self.scene_pub.publish(scene_msg)
        self.get_logger().info('Published planning scene with collision objects')
        
        # Destroy timer after first publish
        self.destroy_timer(self.create_timer(2.0, self.publish_scene_once))

def main():
    rclpy.init()
    node = PlanningScenePublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()