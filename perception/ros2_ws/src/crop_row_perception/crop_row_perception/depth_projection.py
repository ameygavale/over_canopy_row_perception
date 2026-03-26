import numpy as np
from typing import Optional, Tuple


class DepthProjection:
    """
    Projects 2D pixel centerline to metric lateral and heading errors
    using OAK-D Wide stereo depth map.

    OAK-D Wide specs:
        - 150 degree DFOV
        - Myriad X VPU (depth computed on-device)
        - Baseline: ~7.5cm

    Coordinate convention:
        lateral_m   > 0  →  robot is to the LEFT of row center
        heading_rad > 0  →  robot is rotated CW relative to row
    """

    def __init__(
        self,
        fx: float = 400.0,
        fy: float = 400.0,
        cx: float = 640.0,
        cy: float = 360.0,
    ):
        """
        Args:
            fx, fy: focal lengths in pixels (update with OAK-D calibration)
            cx, cy: principal point in pixels
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def project(
        self,
        centerline_px: np.ndarray,
        depth_map: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Project pixel centerline to metric navigation errors.

        Args:
            centerline_px: Nx2 array of (x, y) pixel coordinates
            depth_map:     H x W float32 depth map in meters

        Returns:
            lateral_m   - lateral offset from row center in meters
            heading_rad - heading error in radians
            confidence  - projection confidence [0, 1]
        """
        if len(centerline_px) < 2:
            return 0.0, 0.0, 0.0

        # Sample depth along centerline
        valid_points = []
        for px, py in centerline_px:
            px, py = int(px), int(py)
            if 0 <= py < depth_map.shape[0] and 0 <= px < depth_map.shape[1]:
                d = float(depth_map[py, px])
                if 0.1 < d < 20.0:  # valid depth range in meters
                    x_m = (px - self.cx) * d / self.fx
                    y_m = (py - self.cy) * d / self.fy
                    valid_points.append((x_m, y_m, d))

        if len(valid_points) < 2:
            return 0.0, 0.0, 0.0

        points = np.array(valid_points)

        # Lateral error: x offset of the midpoint of centerline
        lateral_m = float(np.mean(points[:, 0]))

        # Heading error: slope of centerline in metric space
        if len(points) >= 2:
            dx = points[-1, 0] - points[0, 0]
            dz = points[-1, 2] - points[0, 2]
            heading_rad = float(np.arctan2(dx, dz + 1e-6))
        else:
            heading_rad = 0.0

        # Confidence: fraction of centerline points with valid depth
        confidence = len(valid_points) / (len(centerline_px) + 1e-6)

        return lateral_m, heading_rad, confidence

    def update_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        """Update camera intrinsics from OAK-D calibration."""
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
