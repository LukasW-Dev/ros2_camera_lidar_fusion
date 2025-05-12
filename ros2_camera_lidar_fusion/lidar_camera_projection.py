#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
import yaml
import struct

from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import transformations as tf_transformations


def load_extrinsic_matrix(yaml_path: str) -> np.ndarray:
    
    # Expand ~ to the full home directory path
    yaml_path = os.path.expanduser(yaml_path)
    
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"No extrinsic file found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    if 'extrinsic_matrix' not in data:
        raise KeyError(f"YAML {yaml_path} has no 'extrinsic_matrix' key.")

    matrix_list = data['extrinsic_matrix']
    T = np.array(matrix_list, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("Extrinsic matrix is not 4x4.")
    return T

def load_camera_calibration(yaml_path: str) -> (np.ndarray, np.ndarray):
    
    # Expand ~ to the full home directory path
    yaml_path = os.path.expanduser(yaml_path)

    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"No camera calibration file: {yaml_path}")

    with open(yaml_path, 'r') as f:
        calib_data = yaml.safe_load(f)

    cam_mat_data = calib_data['camera_matrix']['data']
    camera_matrix = np.array(cam_mat_data, dtype=np.float64)

    dist_data = calib_data['distortion_coefficients']['data']
    dist_coeffs = np.array(dist_data, dtype=np.float64).reshape((1, -1))

    return camera_matrix, dist_coeffs

def pointcloud2_to_xyz_array_fast(cloud_msg: PointCloud2, skip_rate: int = 1) -> np.ndarray:
    if cloud_msg.height == 0 or cloud_msg.width == 0:
        return np.zeros((0, 3), dtype=np.float32)

    field_names = [f.name for f in cloud_msg.fields]
    if not all(k in field_names for k in ('x','y','z')):
        return np.zeros((0,3), dtype=np.float32)

    dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('_', 'V{}'.format(cloud_msg.point_step - 12))
    ])

    raw_data = np.frombuffer(cloud_msg.data, dtype=dtype)
    points = np.zeros((raw_data.shape[0], 3), dtype=np.float32)
    points[:,0] = raw_data['x']
    points[:,1] = raw_data['y']
    points[:,2] = raw_data['z']

    if skip_rate > 1:
        points = points[::skip_rate]

    return points

class LidarCameraProjectionNode(Node):
    def __init__(self):
        super().__init__('lidar_camera_projection_node')

        # Transform Listener for extrinsic parameter fetching
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # === LIDAR parameters ===
        #self.declare_parameter('lidar.lidar_topic', '/obstacle_point_cloud')
        self.declare_parameter('lidar.lidar_topic', '/left_laser/pandar')
        self.declare_parameter('lidar.colored_cloud_topic', '/rgb_cloud')
        self.declare_parameter('lidar.frame_id', 'left_laser_mount')

        # === CAMERA parameters ===
        self.declare_parameter('camera.image_topic', '/segmentation/image')
        self.declare_parameter('camera.projected_topic', '/projected_image')
        self.declare_parameter('camera.image_size.width', 1280)
        self.declare_parameter('camera.image_size.height', 720)
        self.declare_parameter('camera.frame_id', 'hazard_front_left_camera_optical_frame')

        # === GENERAL parameters ===
        #self.declare_parameter('general.camera_intrinsic_calibration', '~/ros2_ws/src/ros2_camera_lidar_fusion/param/fwt_intrinsics.yaml')
        #self.declare_parameter('general.camera_extrinsic_calibration', '~/ros2_ws/src/ros2_camera_lidar_fusion/param/fwt_extrinsics.yaml')
        self.declare_parameter('general.camera_intrinsic_calibration', '~/ros2_ws/src/ros2_camera_lidar_fusion/param/intrinsics.yaml')
        self.declare_parameter('general.camera_extrinsic_calibration', '~/ros2_ws/src/ros2_camera_lidar_fusion/param/extrinsics.yaml')
        self.declare_parameter('general.slop', 0.1)
        self.declare_parameter('general.max_file_saved', 10)
        self.declare_parameter('general.keyboard_listener', True)
        self.declare_parameter('general.get_intrinsics', True)
        self.declare_parameter('general.get_extrinsics', True)

        
        # config_folder = self.get_parameter('general.config_folder').get_parameter_value()._string_value
        #extrinsic_yaml = self.get_parameter('general.camera_extrinsic_calibration').get_parameter_value()._string_value
        #self.T_lidar_to_cam = load_extrinsic_matrix(extrinsic_yaml)
        self.T_lidar_to_cam = None

        camera_yaml = self.get_parameter('general.camera_intrinsic_calibration').get_parameter_value()._string_value
        self.camera_matrix, self.dist_coeffs = load_camera_calibration(camera_yaml)

        #self.get_logger().info("Loaded extrinsic:\n{}".format(self.T_lidar_to_cam))
        self.get_logger().info("Camera matrix:\n{}".format(self.camera_matrix))

        lidar_topic = self.get_parameter('lidar.lidar_topic').get_parameter_value().string_value
        image_topic = self.get_parameter('camera.image_topic').get_parameter_value().string_value

        self.image_sub = Subscriber(self, Image, image_topic)
        self.lidar_sub = Subscriber(self, PointCloud2, lidar_topic)

        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub],
            queue_size=10,
            slop=0.07
        )
        self.ts.registerCallback(self.sync_callback)

        projected_topic = self.get_parameter('camera.projected_topic').get_parameter_value().string_value
        colored_cloud_topic = self.get_parameter('lidar.colored_cloud_topic').get_parameter_value().string_value

        self.pub_image = self.create_publisher(Image, projected_topic, 1)
        self.pub_cloud = self.create_publisher(PointCloud2, colored_cloud_topic, 1)
        self.bridge = CvBridge()

        self.skip_rate = 1

    # get extrinsics from tf static
    def get_extrinsic_matrix(self, target_frame: str, source_frame: str, timeout_sec: float = 1.0) -> np.ndarray:
        try:
            # Look up transform: source → target
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                target_frame=target_frame,
                source_frame=source_frame,
                time=rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout_sec)
            )

            # Extract translation and quaternion
            t = trans.transform.translation
            q = trans.transform.rotation

            translation = np.array([t.x, t.y, t.z])
            quaternion = np.array([q.w, q.x, q.y, q.z])
            self.get_logger().error(f"Quaternion: {quaternion} q: {q}")

            # Convert quaternion to rotation matrix
            T = tf_transformations.quaternion_matrix(quaternion)  # 4x4 matrix
            T[0:3, 3] = translation

            return T

        except Exception as e:
            self.get_logger().error(f"Failed to get transform from {source_frame} to {target_frame}: {e}")
            raise

    def sync_callback(self, image_msg: Image, lidar_msg: PointCloud2):
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        og_image = cv_image.copy()

        xyz_lidar = pointcloud2_to_xyz_array_fast(lidar_msg, skip_rate=self.skip_rate)

        # Rough filter: remove LiDAR points that are unlikely to project into the image
        # mask_roi = (xyz_lidar[:, 0] > 0.5) & (xyz_lidar[:, 0] < 10.0)  # Forward range
        # xyz_lidar = xyz_lidar[mask_roi]

        n_points = xyz_lidar.shape[0]
        if n_points == 0:
            self.get_logger().warn("Empty cloud. Nothing to project.")
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            out_msg.header = image_msg.header
            self.pub_image.publish(out_msg)
            return

        xyz_lidar_f64 = xyz_lidar.astype(np.float64)
        ones = np.ones((n_points, 1), dtype=np.float64)
        xyz_lidar_h = np.hstack((xyz_lidar_f64, ones))

        if self.T_lidar_to_cam is None:
            self.T_lidar_to_cam = self.get_extrinsic_matrix(image_msg.header.frame_id, lidar_msg.header.frame_id)
            self.get_logger().info("Loaded extrinsic:\n{}".format(self.T_lidar_to_cam))

        xyz_cam_h = xyz_lidar_h @ self.T_lidar_to_cam.T
        xyz_cam = xyz_cam_h[:, :3]

        mask_in_front = (xyz_cam[:, 2] > 0.0)
        xyz_cam_front = xyz_cam[mask_in_front]
        lidar_points_front = xyz_lidar[mask_in_front]

        if xyz_cam_front.shape[0] == 0:
            self.get_logger().info("No points in front of camera (z>0).")
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            out_msg.header = image_msg.header
            self.pub_image.publish(out_msg)
            return

        rvec = np.zeros((3,1), dtype=np.float64)
        tvec = np.zeros((3,1), dtype=np.float64)
        image_points, _ = cv2.projectPoints(
            xyz_cam_front,
            rvec, tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        image_points = image_points.reshape(-1, 2)

        h, w = cv_image.shape[:2]

        colored_points = []
        for i, (u, v) in enumerate(image_points):
            u_int = int(u + 0.5)
            v_int = int(v + 0.5)
            if 0 <= u_int < w and 0 <= v_int < h:
                cv2.circle(cv_image, (u_int, v_int), 2, (0, 255, 0), -1)
                color = og_image[v_int, u_int]
                r, g, b = int(color[2]), int(color[1]), int(color[0])  # OpenCV uses BGR
                rgb_packed = struct.unpack('I', struct.pack('BBBB', b, g, r, 0))[0]
                colored_points.append((*lidar_points_front[i], rgb_packed))

        out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        out_msg.header = image_msg.header
        self.pub_image.publish(out_msg)

        if len(colored_points) == 0:
            self.get_logger().warn("No valid points projected onto the image.")
            return
        
        # Create the PointCloud2 message
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1)
        ]

        point_step = 16  # 4 bytes each for x, y, z, and rgb
        data = bytearray()
        for point in colored_points:
            data.extend(struct.pack('fffI', *point))

        cloud_msg = PointCloud2()
        cloud_msg.header = lidar_msg.header
        cloud_msg.height = 1  # Unstructured point cloud
        cloud_msg.width = len(colored_points)  # Only projected points
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = point_step
        cloud_msg.row_step = point_step * cloud_msg.width
        cloud_msg.is_dense = True
        cloud_msg.data = data

        self.pub_cloud.publish(cloud_msg)
    
def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
