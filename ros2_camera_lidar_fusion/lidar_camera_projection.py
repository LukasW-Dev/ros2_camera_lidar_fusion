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

visualize = True


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
        self.declare_parameter('lidar.lidar_topic', None)
        self.declare_parameter('lidar.colored_cloud_topic', None)
        lidar_topic = self.get_parameter('lidar.lidar_topic').get_parameter_value().string_value
        colored_cloud_topic = self.get_parameter('lidar.colored_cloud_topic').get_parameter_value().string_value
        self.get_logger().info("Lidar topic: " + lidar_topic)
        self.get_logger().info("Colored cloud topic: " + colored_cloud_topic)

        # === CAMERA parameters ===
        self.declare_parameter('camera.image_topic', None)
        self.declare_parameter('camera.confidence_topic', None)
        self.declare_parameter('camera.projected_topic', None)
        image_topic = self.get_parameter('camera.image_topic').get_parameter_value().string_value
        confidence_topic = self.get_parameter('camera.confidence_topic').get_parameter_value().string_value
        projected_topic = self.get_parameter('camera.projected_topic').get_parameter_value().string_value
        self.get_logger().info("Image topic: " + image_topic)
        self.get_logger().info("Confidence topic: " + confidence_topic)
        self.get_logger().info("Projected topic: " + projected_topic)

        # === GENERAL parameters ===
        self.declare_parameter('general.camera_intrinsic_calibration', None)
        self.declare_parameter('general.camera_extrinsic_calibration', None)
        self.declare_parameter('general.slop', None)
        self.declare_parameter('general.max_queue_size', None)
        camera_yaml = self.get_parameter('general.camera_intrinsic_calibration').get_parameter_value()._string_value
        extrinsic_yaml = self.get_parameter('general.camera_extrinsic_calibration').get_parameter_value()._string_value
        slop = self.get_parameter('general.slop').get_parameter_value()._double_value
        max_queue_size = self.get_parameter('general.max_queue_size').get_parameter_value()._integer_value
        self.get_logger().info("Camera intrinsic calibration: " + camera_yaml)
        self.get_logger().info("Camera extrinsic calibration: " + extrinsic_yaml)
        self.get_logger().info("Slop: " + str(slop))
        self.get_logger().info("Max queue size: " + str(max_queue_size))

        if extrinsic_yaml:
          self.T_lidar_to_cam = load_extrinsic_matrix(extrinsic_yaml)
        else:
          self.T_lidar_to_cam = None

        print("Camera yaml: " + camera_yaml)
        self.camera_matrix, self.dist_coeffs = load_camera_calibration(camera_yaml)

        self.get_logger().info("Loaded extrinsic:\n{}".format(self.T_lidar_to_cam))
        self.get_logger().info("Camera matrix:\n{}".format(self.camera_matrix))

        qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            depth=10,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
        )

        self.image_sub = Subscriber(self, Image, image_topic)
        self.confidence_sub = Subscriber(self, Image, confidence_topic)
        self.lidar_sub = Subscriber(self, PointCloud2, lidar_topic, qos_profile=qos)


        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.confidence_sub, self.lidar_sub],
            queue_size=max_queue_size,
            slop=slop,
        )
        self.ts.registerCallback(self.sync_callback)


        self.pub_image = self.create_publisher(Image, projected_topic, 1)
        self.pub_cloud = self.create_publisher(PointCloud2, colored_cloud_topic, 1)
        self.bridge = CvBridge()

        self.skip_rate = 2

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

            # Convert quaternion to rotation matrix
            T = tf_transformations.quaternion_matrix(quaternion)  # 4x4 matrix
            T[0:3, 3] = translation

            return T

        except Exception as e:
            self.get_logger().error(f"Failed to get transform from {source_frame} to {target_frame}: {e}")
            raise

    def sync_callback(self, image_msg: Image, confidence_msg, lidar_msg: PointCloud2):

        #self.get_logger().info("Received synchronized messages.")

        # Measure the time it takes to process the callback
        start_time = self.get_clock().now()

        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        og_image = cv_image.copy()

        conf_image = self.bridge.imgmsg_to_cv2(confidence_msg, desired_encoding='32FC1')

        
        xyz_lidar = pointcloud2_to_xyz_array_fast(lidar_msg, skip_rate=self.skip_rate)

        # Seems to not make much difference (maybe if using more pointclouds)
        # dists = np.linalg.norm(xyz_lidar, axis=1)   # Euclidean distance for each point

        # # Now build a mask based on radial distance AND any other criteria (e.g. height < 2 m)
        # mask = (
        #     (dists > 0.5) &        # farther than 0.5 m from the sensor
        #     (dists < 10.0)       # closer than 10 m
        # )

        # xyz_lidar = xyz_lidar[mask]

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
        xyz_lidar_front = xyz_lidar[mask_in_front]

        # --- after computing xyz_cam_front and xyz_lidar_front, do:
        # 1) Project all points at once:
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        image_points, _ = cv2.projectPoints(
            xyz_cam_front, rvec, tvec,
            self.camera_matrix, self.dist_coeffs
        )
        # image_points is (N,1,2) → reshape to (N,2)
        image_points = image_points.reshape(-1, 2)

        # 2) Convert to integer pixel coordinates (round):
        uv = np.rint(image_points).astype(np.int32)   # shape = (N,2)
        u_coords = uv[:, 0]
        v_coords = uv[:, 1]

        h, w = og_image.shape[:2]

        # 3) Build a boolean “in‐bounds” mask:
        in_bounds = (
            (u_coords >= 0) & (u_coords < w) &
            (v_coords >= 0) & (v_coords < h)
        )

        # Short‐circuit if nothing is in‐bounds:
        if not np.any(in_bounds):
            self.get_logger().warn("No points project inside image bounds.")
            # publish empty, return, etc.
            return

        # 4) Filter arrays by in‐bounds:
        u_in = u_coords[in_bounds]
        v_in = v_coords[in_bounds]
        lidar_in = xyz_lidar_front[in_bounds]      # (M,3) ℝ coordinates of the surviving points

        # 5) Bulk‐index colors and confidences:
        #    OpenCV stores images as BGR, so og_image[v, u] is (B,G,R)
        #    conf_image is float32((H,W)) → round/cast to int if needed.
        bgr_pixels = og_image[v_in, u_in]            # shape = (M,3)
        conf_vals  = conf_image[v_in, u_in].astype(np.int32)  # shape = (M,)

        # 6) Build a “sky”‐mask (r=135, g=206, b=235) in vector form:
        #    Note: bgr_pixels[:,0] == B, bgr_pixels[:,1] == G, bgr_pixels[:,2] == R
        is_sky = (
            (bgr_pixels[:, 2] == 135) &
            (bgr_pixels[:, 1] == 206) &
            (bgr_pixels[:, 0] == 235)
        )
        keep_mask = ~is_sky

        # 7) Final “kept” points:
        lidar_kept   = lidar_in[keep_mask]        # (K,3)
        bgr_kept     = bgr_pixels[keep_mask]       # (K,3)
        conf_kept    = conf_vals[keep_mask]        # (K,)

        if lidar_kept.shape[0] == 0:
            self.get_logger().warn("All points fell on sky color.")
            return
        
        # 8) If visualize=True, draw circles only at those final pixel coords:
        if visualize:
            # Loop over the much-smaller “kept” set (K points):
            for u_i, v_i in zip(u_in, v_in):
                # Note: u_i and v_i are already ints, but if you 
                # used rint.astype(int) above, they’re safe to use directly.
                cv2.circle(cv_image, (u_i, v_i), 2, (0, 255, 0), -1)

            # Publish the image once, after drawing:
            out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            out_msg.header = image_msg.header
            self.pub_image.publish(out_msg)

        # 9) Pack RGB into a single uint32:
        #    struct.unpack('I', struct.pack('BBBB', b, g, r, 0))[0]
        #    But you can vectorize that:
        b = bgr_kept[:, 0].astype(np.uint32)
        g = bgr_kept[:, 1].astype(np.uint32)
        r = bgr_kept[:, 2].astype(np.uint32)
        rgb_packed = (b << 0) | (g << 8) | (r << 16)

        # 10) Build one big array of shape (K,5) with dtype float32, uint32, etc.
        #    Format per‐row is [x, y, z, rgb, confidence]
        dtype = np.dtype([
            ("x",       np.float32),
            ("y",       np.float32),
            ("z",       np.float32),
            ("rgb",     np.uint32),
            ("label",   np.uint32),
        ])
        cloud_array = np.zeros(lidar_kept.shape[0], dtype=dtype)
        cloud_array["x"]     = lidar_kept[:, 0].astype(np.float32)
        cloud_array["y"]     = lidar_kept[:, 1].astype(np.float32)
        cloud_array["z"]     = lidar_kept[:, 2].astype(np.float32)
        cloud_array["rgb"]   = rgb_packed
        cloud_array["label"] = conf_kept.astype(np.uint32)

        # 10) Convert that structured array directly to bytes:
        raw_data = cloud_array.tobytes()

        # 11) Fill PointCloud2 fields and data:
        cloud_msg = PointCloud2()
        cloud_msg.header = lidar_msg.header
        cloud_msg.height = 1
        cloud_msg.width = cloud_array.shape[0]
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = cloud_array.dtype.itemsize  # should be 4+4+4+4+4 = 20
        cloud_msg.row_step   = cloud_msg.point_step * cloud_msg.width
        cloud_msg.is_dense   = True
        cloud_msg.data       = raw_data

        # Define fields:
        cloud_msg.fields = [
            PointField(name="x",     offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",     offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",     offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb",   offset=12, datatype=PointField.UINT32,  count=1),
            PointField(name="label", offset=16, datatype=PointField.UINT32,  count=1),
        ]

        self.pub_cloud.publish(cloud_msg)

        end_time = self.get_clock().now()
        elapsed_time = (end_time - start_time).nanoseconds / 1e6
        self.get_logger().error(f"Processed {n_points} points in {elapsed_time:.2f} ms, projected {10} points.")
    
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
