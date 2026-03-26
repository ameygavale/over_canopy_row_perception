import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import numpy as np

from .exg_branch import ExGBranch
from .yolo_branch import YOLOBranch
from .growth_stage import GrowthStageClassifier
from .depth_projection import DepthProjection


class FusionNode(Node):
    """
    Multi-branch crop row perception fusion node.

    Subscribes:
        /oak/rgb/image_raw     - RGB image from OAK-D Wide
        /oak/stereo/image_raw  - Depth image from OAK-D Wide

    Publishes:
        /crop_row/lateral_m    - lateral offset from row center [meters]
        /crop_row/heading_rad  - heading error relative to row [radians]
        /crop_row/confidence   - ensemble confidence [0, 1]

    Pipeline:
        RGB → Growth Stage Classifier → branch weights
        RGB → ExG Branch              → mask + centerline + confidence
        RGB → YOLOv8-seg Branch       → mask + centerline + confidence
        RGB → YOLOv10-seg Branch      → mask + centerline + confidence
              Weighted Fusion         → fused centerline
        Depth → DepthProjection       → lateral_m, heading_rad
    """

    def __init__(self):
        super().__init__("crop_row_fusion")

        # Load params
        self.declare_parameter("yolov8_weights",  "")
        self.declare_parameter("yolov10_weights", "")
        self.declare_parameter("default_stage",   "corn_mid")
        self.declare_parameter("conf_threshold",  0.25)

        yolov8_weights  = self.get_parameter("yolov8_weights").value
        yolov10_weights = self.get_parameter("yolov10_weights").value
        default_stage   = self.get_parameter("default_stage").value
        conf_threshold  = self.get_parameter("conf_threshold").value

        # Perception branches
        self.bridge    = CvBridge()
        self.exg       = ExGBranch()
        self.yolov8    = YOLOBranch(yolov8_weights,  model_type="yolov8")  if yolov8_weights  else None
        self.yolov10   = YOLOBranch(yolov10_weights, model_type="yolov10") if yolov10_weights else None
        self.classifier = GrowthStageClassifier(default_stage=default_stage)
        self.projector  = DepthProjection()
        self.conf_threshold = conf_threshold

        self.latest_depth = None

        # Subscribers
        self.create_subscription(Image, "/oak/rgb/image_raw",    self._rgb_cb,   10)
        self.create_subscription(Image, "/oak/stereo/image_raw", self._depth_cb, 10)

        # Publishers
        self.pub_lateral  = self.create_publisher(Float32MultiArray, "/crop_row/lateral_m",   10)
        self.pub_heading  = self.create_publisher(Float32MultiArray, "/crop_row/heading_rad",  10)
        self.pub_conf     = self.create_publisher(Float32MultiArray, "/crop_row/confidence",   10)

        self.get_logger().info("FusionNode initialized")

    def _rgb_cb(self, msg: Image):
        bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # 1. Growth stage classification → branch weights
        stage   = self.classifier.predict(bgr)
        weights = self.classifier.get_weights(stage)  # [w_exg, w_yolov8, w_yolov10]

        # 2. Run active branches
        exg_out    = self.exg.compute(bgr)
        yolov8_out = self.yolov8.infer(bgr,  self.conf_threshold) if self.yolov8  else None
        yolov10_out= self.yolov10.infer(bgr, self.conf_threshold) if self.yolov10 else None

        # 3. Confidence-weighted centerline fusion
        centerline = self._fuse_centerlines(
            exg_out, yolov8_out, yolov10_out, weights
        )

        # 4. Depth projection → metric errors
        lateral_m, heading_rad, proj_conf = 0.0, 0.0, 0.0
        if self.latest_depth is not None and len(centerline) > 1:
            lateral_m, heading_rad, proj_conf = self.projector.project(
                centerline, self.latest_depth
            )

        # 5. Publish
        ensemble_conf = float(
            weights[0] * exg_out["confidence"]
            + (weights[1] * yolov8_out["confidence"]  if yolov8_out  else 0.0)
            + (weights[2] * yolov10_out["confidence"] if yolov10_out else 0.0)
        )

        self._publish(self.pub_lateral,  [lateral_m])
        self._publish(self.pub_heading,  [heading_rad])
        self._publish(self.pub_conf,     [ensemble_conf])

        self.get_logger().debug(
            f"stage={stage} | lat={lateral_m:.3f}m | "
            f"hdg={heading_rad:.3f}rad | conf={ensemble_conf:.2f}"
        )

    def _depth_cb(self, msg: Image):
        self.latest_depth = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough"
        ).astype(np.float32)

    def _fuse_centerlines(self, exg_out, yolov8_out, yolov10_out, weights):
        """Weighted average of branch centerlines by confidence × stage weight."""
        candidates = []
        total_weight = 0.0

        def add(branch_out, stage_weight):
            nonlocal total_weight
            if branch_out and len(branch_out["centerline"]) > 1:
                w = stage_weight * branch_out["confidence"]
                candidates.append((branch_out["centerline"], w))
                total_weight += w

        add(exg_out,    weights[0])
        add(yolov8_out, weights[1])
        add(yolov10_out,weights[2])

        if not candidates:
            return np.empty((0, 2), dtype=int)

        # Use highest-weight centerline for now
        # TODO: interpolate between centerlines for smoother fusion
        best = max(candidates, key=lambda x: x[1])
        return best[0]

    def _publish(self, pub, values: list):
        msg = Float32MultiArray()
        msg.data = [float(v) for v in values]
        pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    rclpy.shutdown()
