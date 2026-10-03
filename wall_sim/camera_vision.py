"""Camera Vision Perception Module (Computer Vision + 3D De-projection).

Captures synthetic RGB-D camera images from an overhead virtual camera in PyBullet,
segments the brick using OpenCV color and contour analysis, extracts 2D pixel
poses (u, v, theta), and de-projects them into 3D world coordinates (X, Y, Z, Yaw)
without querying PyBullet's ground-truth object database.
"""
import math
import os
from typing import Dict, List, Optional

# pyrefly: ignore [missing-import]
import cv2
import numpy as np
# pyrefly: ignore [missing-import]
import pybullet as p

from .config import Config


def _ransac_plane_fit(pts_3d, n_iter=100, threshold=0.004):
    """RANSAC plane fitting on a 3D point cloud.

    Fits a plane to the point cloud and returns the surface normal and
    the fraction of inlier points.  Much more robust than raw PCA for
    noisy depth data and partial occlusion from support wedges.

    Returns:
        normal (np.ndarray): unit normal of the best-fit plane (z > 0).
        inlier_ratio (float): fraction of points within *threshold* of the plane.
    """
    best_normal = np.array([0.0, 0.0, 1.0])
    best_inlier_count = 0
    best_inlier_mask = None
    n = len(pts_3d)
    if n < 3:
        return best_normal, 0.0

    for _ in range(n_iter):
        idx = np.random.choice(n, 3, replace=False)
        p0, p1, p2 = pts_3d[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal /= norm_len
        if normal[2] < 0:
            normal = -normal

        distances = np.abs((pts_3d - p0) @ normal)
        mask = distances < threshold
        count = int(np.sum(mask))
        if count > best_inlier_count:
            best_inlier_count = count
            best_normal = normal.copy()
            best_inlier_mask = mask

    # Refine: least-squares PCA on inliers only for a smoother normal
    if best_inlier_mask is not None and int(np.sum(best_inlier_mask)) >= 10:
        inlier_pts = pts_3d[best_inlier_mask]
        c_mean = np.mean(inlier_pts, axis=0)
        cov = np.cov(inlier_pts - c_mean, rowvar=False)
        _, evecs = np.linalg.eigh(cov)
        refined = evecs[:, 0]   # smallest eigenvalue = normal direction
        if refined[2] < 0:
            refined = -refined
        best_normal = refined

    inlier_ratio = best_inlier_count / n if n > 0 else 0.0
    return best_normal, inlier_ratio


class OverheadCamera:
    """Simulated RGB-D Overhead Camera with OpenCV vision processing."""

    def __init__(self,
                 cfg: Optional[Config] = None,
                 eye_pos: Optional[tuple] = None,
                 target_pos: Optional[tuple] = None,
                 up_vector: Optional[tuple] = None,
                 img_width: Optional[int] = None,
                 img_height: Optional[int] = None,
                 fov: Optional[float] = None,
                 near_val: Optional[float] = None,
                 far_val: Optional[float] = None):
        self.cfg = cfg
        cam = cfg.camera if cfg is not None else None

        self.eye_pos = np.array(
            eye_pos if eye_pos is not None else (cam.eye_pos if cam else (0.48, 0.0, 1.15)),
            dtype=float)
        self.target_pos = np.array(
            target_pos if target_pos is not None else (cam.target_pos if cam else (0.48, 0.0, 0.0)),
            dtype=float)
        self.up_vector = np.array(
            up_vector if up_vector is not None else (cam.up_vector if cam else (0.0, 1.0, 0.0)),
            dtype=float)
        self.width = int(img_width if img_width is not None else (cam.img_width if cam else 640))
        self.height = int(img_height if img_height is not None else (cam.img_height if cam else 640))
        self.fov = float(fov if fov is not None else (cam.fov if cam else 55.0))
        self.near = float(near_val if near_val is not None else (cam.near_val if cam else 0.1))
        self.far = float(far_val if far_val is not None else (cam.far_val if cam else 2.0))

        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=list(self.eye_pos),
            cameraTargetPosition=list(self.target_pos),
            cameraUpVector=list(self.up_vector)
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=float(self.width) / float(self.height),
            nearVal=self.near,
            farVal=self.far
        )

    def capture_rgbd(self):
        """Captures raw RGB image and metric depth map from PyBullet."""
        info = p.getConnectionInfo()
        is_direct = (info.get("connectionMethod") == p.DIRECT)

        if is_direct:
            renderer = p.ER_TINY_RENDERER
        else:
            renderer = p.ER_BULLET_HARDWARE_OPENGL

        try:
            img_arr = p.getCameraImage(
                self.width,
                self.height,
                self.view_matrix,
                self.proj_matrix,
                renderer=renderer
            )
        except Exception:
            img_arr = p.getCameraImage(
                self.width,
                self.height,
                self.view_matrix,
                self.proj_matrix,
                renderer=p.ER_TINY_RENDERER
            )

        rgb_raw = np.array(img_arr[2], dtype=np.uint8).reshape((self.height, self.width, 4))
        rgb = rgb_raw[:, :, :3]

        depth_buffer = np.array(img_arr[3], dtype=np.float32).reshape((self.height, self.width))
        depth_meters = (2.0 * self.near * self.far) / (
            self.far + self.near - (2.0 * depth_buffer - 1.0) * (self.far - self.near)
        )
        return rgb, depth_meters

    def detect_all_bricks(self, debug_save_path: str = None) -> List[dict]:
        """Detects ALL visible bricks in the overhead RGB-D frame and returns their 3D poses."""
        rgb, depth_map = self.capture_rgbd()

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        cam = self.cfg.camera if self.cfg else None
        lower_red1 = np.array(cam.hsv_lower1 if cam else (0, 70, 50))
        upper_red1 = np.array(cam.hsv_upper1 if cam else (15, 255, 255))
        lower_red2 = np.array(cam.hsv_lower2 if cam else (165, 70, 50))
        upper_red2 = np.array(cam.hsv_upper2 if cam else (180, 255, 255))
        min_area = cam.min_contour_area if cam else 100

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        results = []
        tan_half_fov = math.tan(math.radians(self.fov) / 2.0)
        brick_h = self.cfg.brick.height if self.cfg else 0.06

        vis_bgr = None
        if debug_save_path:
            os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
            vis_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            rect = cv2.minAreaRect(contour)
            (center_u, center_v), (rect_w, rect_h), raw_angle = rect

            u_int = int(round(center_u))
            v_int = int(round(center_v))
            u_int = np.clip(u_int, 0, self.width - 1)
            v_int = np.clip(v_int, 0, self.height - 1)

            box = cv2.boxPoints(rect)
            box = np.intp(box)

            edge1 = box[1] - box[0]
            edge2 = box[2] - box[1]
            len1 = np.linalg.norm(edge1)
            len2 = np.linalg.norm(edge2)
            long_edge = edge1 if len1 > len2 else edge2
            pixel_yaw = math.atan2(long_edge[1], long_edge[0])

            world_yaw = -pixel_yaw
            world_yaw = (world_yaw + math.pi / 2.0) % math.pi - math.pi / 2.0

            v_min, v_max = max(0, v_int - 2), min(self.height, v_int + 3)
            u_min, u_max = max(0, u_int - 2), min(self.width, u_int + 3)
            z_dist = float(np.median(depth_map[v_min:v_max, u_min:u_max]))

            ndc_x = (2.0 * center_u / self.width) - 1.0
            ndc_y = 1.0 - (2.0 * center_v / self.height)

            x_cam = ndc_x * z_dist * tan_half_fov
            y_cam = ndc_y * z_dist * tan_half_fov

            world_x = float(self.eye_pos[0] + x_cam)
            world_y = float(self.eye_pos[1] + y_cam)
            world_z = float((self.eye_pos[2] - z_dist) - (brick_h / 2.0))

            # --- 3D Surface Normal & Tilt Estimation via Point Cloud PCA ---
            contour_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            eroded = cv2.erode(contour_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            vs, us = np.where(eroded > 0)
            if len(vs) < 30:
                vs, us = np.where(contour_mask > 0)

            z_pts = depth_map[vs, us]
            valid = (z_pts > self.near) & (z_pts < self.far)
            vs, us, z_pts = vs[valid], us[valid], z_pts[valid]

            normal = np.array([0.0, 0.0, 1.0])
            tilt_deg = 0.0
            tilt_confidence = 0.0
            gripper_quat = None

            if len(vs) >= 20:
                ndc_xi = (2.0 * us / self.width) - 1.0
                ndc_yi = 1.0 - (2.0 * vs / self.height)
                xi_cam = ndc_xi * z_pts * tan_half_fov
                yi_cam = ndc_yi * z_pts * tan_half_fov
                pts_x = self.eye_pos[0] + xi_cam
                pts_y = self.eye_pos[1] + yi_cam
                pts_z = self.eye_pos[2] - z_pts
                pts_3d = np.column_stack([pts_x, pts_y, pts_z])

                # RANSAC plane fit for robust surface normal (filters depth
                # noise and support-wedge outlier pixels)
                ransac_normal, inlier_ratio = _ransac_plane_fit(
                    pts_3d, n_iter=100, threshold=0.004
                )
                tilt_confidence = inlier_ratio

                # PCA for axis decomposition (eigenvec ordering: smallest → largest)
                c_mean = np.mean(pts_3d, axis=0)
                cov = np.cov(pts_3d - c_mean, rowvar=False)
                eigenvals, eigenvecs = np.linalg.eigh(cov)

                # Use RANSAC normal when inlier ratio is good; else PCA fallback
                if inlier_ratio >= 0.55:
                    normal = ransac_normal
                else:
                    est_norm = eigenvecs[:, 0]
                    if est_norm[2] < 0:
                        est_norm = -est_norm
                    normal = est_norm

                # Extract yaw from 3D PCA major axis projected onto the ground
                # plane — immune to perspective distortion on tilted bricks.
                # Only override the 2D yaw for tilted bricks (>5°); for flat
                # bricks the 2D minAreaRect yaw is already accurate and
                # the 3D override can regress those cases.
                tilt_check_rad = math.acos(float(np.clip(normal[2], -1.0, 1.0)))
                tilt_check_deg = math.degrees(tilt_check_rad)
                if tilt_check_deg > 5.0 and len(pts_3d) >= 50:
                    major_axis_3d = eigenvecs[:, 2]
                    major_xy = major_axis_3d[:2].copy()
                    if np.linalg.norm(major_xy) > 1e-6:
                        major_xy /= np.linalg.norm(major_xy)
                        pca_yaw = float(math.atan2(major_xy[1], major_xy[0]))
                        pca_yaw = (pca_yaw + math.pi / 2.0) % math.pi - math.pi / 2.0
                        world_yaw = pca_yaw

                l_axis = eigenvecs[:, 2]
                l_axis = l_axis - np.dot(l_axis, normal) * normal
                if np.linalg.norm(l_axis) > 1e-6:
                    l_axis /= np.linalg.norm(l_axis)
                else:
                    l_axis = np.array([math.cos(world_yaw), math.sin(world_yaw), 0.0])

                tilt_rad = math.acos(float(np.clip(normal[2], -1.0, 1.0)))
                tilt_deg = float(math.degrees(tilt_rad))

                from .robots.panda_arm import gripper_3d_orientation
                gripper_quat = gripper_3d_orientation(normal, l_axis)

            if gripper_quat is None:
                from .robots.panda_arm import gripper_down_quaternion
                gripper_quat = gripper_down_quaternion(world_yaw)

            results.append({
                "pos": (world_x, world_y, world_z),
                "x": world_x,
                "y": world_y,
                "z": world_z,
                "yaw": float(world_yaw),
                "yaw_deg": float(math.degrees(world_yaw)),
                "tilt_deg": float(tilt_deg),
                "normal": normal.tolist(),
                "gripper_quat": gripper_quat,
                "pixel_center": (center_u, center_v),
                "depth_meters": z_dist,
                "contour_area": area,
                "tilt_confidence": float(tilt_confidence),
            })

            if vis_bgr is not None:
                cv2.drawContours(vis_bgr, [box], 0, (0, 255, 0), 2)
                cv2.circle(vis_bgr, (u_int, v_int), 5, (0, 0, 255), -1)
                arrow_len = 40
                end_u = int(u_int + arrow_len * math.cos(pixel_yaw))
                end_v = int(v_int + arrow_len * math.sin(pixel_yaw))
                cv2.arrowedLine(vis_bgr, (u_int, v_int), (end_u, end_v), (255, 255, 0), 2, tipLength=0.3)
                label = f"X:{world_x:.2f} Y:{world_y:.2f} Yaw:{math.degrees(world_yaw):.0f}d Tilt:{tilt_deg:.1f}d"
                cv2.putText(vis_bgr, label, (max(10, u_int - 80), max(20, v_int - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        if vis_bgr is not None and debug_save_path:
            cv2.imwrite(debug_save_path, vis_bgr)
            print(f"[Vision] Annotated camera snapshot ({len(results)} brick(s)) saved to -> {debug_save_path}")

        return results

    def detect_brick_pose(self, debug_save_path: str = None) -> dict:
        """Single brick detection (for Demo 1 compatibility)."""
        all_bricks = self.detect_all_bricks(debug_save_path=debug_save_path)
        if not all_bricks:
            raise RuntimeError("Vision Perception Failed: No brick detected in camera frame!")
        # For single brick scenarios, return the detection nearest the camera optical axis
        all_bricks.sort(key=lambda b: math.hypot(b["x"] - self.eye_pos[0], b["y"] - self.eye_pos[1]))
        return all_bricks[0]
