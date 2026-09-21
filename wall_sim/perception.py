"""Unified Perception Interface.

Provides a common API for brick detection with swappable backends:
  - ``GroundTruthBackend``: reads PyBullet state directly (for testing/baseline)
  - ``VisionBackend``: uses OverheadCamera + OpenCV (for realistic perception)

All backends return the same ``BrickDetection`` dataclass, so downstream code
(workers, brain, metrics) doesn't care which backend is active.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
# pyrefly: ignore [missing-import]
import pybullet as p

from .config import Config
from .wall_plan import BrickSpec


@dataclass
class BrickDetection:
    """A detected brick's estimated pose and metadata."""
    x: float
    y: float
    z: float
    yaw: float              # radians, normalized to [-pi/2, pi/2]
    yaw_deg: float
    confidence: float = 1.0   # 0..1 (ground-truth is always 1.0)
    body_id: Optional[int] = None   # PyBullet body id (if known)

    @property
    def pos(self) -> tuple:
        return (self.x, self.y, self.z)


class PerceptionInterface(ABC):
    """Abstract base class for all perception backends."""

    @abstractmethod
    def detect_bricks(self) -> List[BrickDetection]:
        """Detect all visible bricks in the workspace.

        Returns a list of ``BrickDetection`` sorted by confidence (highest
        first).
        """

    @abstractmethod
    def verify_placement(self, spec: BrickSpec, body_id: int) -> dict:
        """Check whether a placed brick matches its target pose.

        Returns a dict with: pos_err_mm, yaw_err_deg, acceptable (bool).
        """


# ---------------------------------------------------------------------------
# Backend 1: Ground-Truth (reads from PyBullet simulation state)
# ---------------------------------------------------------------------------

class GroundTruthBackend(PerceptionInterface):
    """Reads brick poses directly from PyBullet -- perfect perception.

    Optionally injects Gaussian noise to simulate sensor imperfection.
    """

    def __init__(self, cfg: Config, brick_ids: list = None,
                 add_noise: bool = False,
                 pos_noise_std: float = 0.001,
                 yaw_noise_std_deg: float = 0.5):
        self.cfg = cfg
        self.brick_ids = list(brick_ids or [])
        self.add_noise = add_noise
        self.pos_noise_std = pos_noise_std
        self.yaw_noise_std_deg = yaw_noise_std_deg

    @staticmethod
    def _normalize_yaw(yaw: float) -> float:
        """Normalize to [-pi/2, pi/2] for 180° brick symmetry."""
        return (yaw + math.pi / 2.0) % math.pi - math.pi / 2.0

    def register_brick(self, body_id: int):
        """Add a brick body id to track."""
        if body_id not in self.brick_ids:
            self.brick_ids.append(body_id)

    def detect_bricks(self) -> List[BrickDetection]:
        detections = []
        for bid in self.brick_ids:
            try:
                pos, orn_q = p.getBasePositionAndOrientation(bid)
            except Exception:
                continue
            euler = p.getEulerFromQuaternion(orn_q)
            yaw = self._normalize_yaw(euler[2])

            x, y, z = pos
            if self.add_noise:
                noise = np.random.normal(0, self.pos_noise_std, size=3)
                x += noise[0]
                y += noise[1]
                z += noise[2]
                yaw += math.radians(
                    np.random.normal(0, self.yaw_noise_std_deg))
                yaw = self._normalize_yaw(yaw)

            detections.append(BrickDetection(
                x=x, y=y, z=z,
                yaw=yaw,
                yaw_deg=math.degrees(yaw),
                confidence=1.0,
                body_id=bid,
            ))
        return detections

    def verify_placement(self, spec: BrickSpec, body_id: int) -> dict:
        pos, orn_q = p.getBasePositionAndOrientation(body_id)
        yaw = p.getEulerFromQuaternion(orn_q)[2]
        pos_err_mm = 1000.0 * math.dist(pos, spec.center)
        yaw_err = math.degrees(yaw - spec.yaw)
        yaw_err = (yaw_err + 180.0) % 360.0 - 180.0
        return {
            "pos_err_mm": round(pos_err_mm, 2),
            "yaw_err_deg": round(yaw_err, 2),
            "acceptable": pos_err_mm < 5.0 and abs(yaw_err) < 3.0,
        }


# ---------------------------------------------------------------------------
# Backend 2: Vision (OpenCV camera-based perception)
# ---------------------------------------------------------------------------

class VisionBackend(PerceptionInterface):
    """Uses OverheadCamera + OpenCV for camera-based brick detection."""

    def __init__(self, cfg: Config, debug_dir: str = None):
        from .camera_vision import OverheadCamera
        self.cfg = cfg
        cam = cfg.camera
        self.camera = OverheadCamera(
            eye_pos=cam.eye_pos,
            target_pos=cam.target_pos,
            up_vector=cam.up_vector,
            img_width=cam.img_width,
            img_height=cam.img_height,
            fov=cam.fov,
            near_val=cam.near_val,
            far_val=cam.far_val,
        )
        self.debug_dir = debug_dir
        self._capture_count = 0

    def detect_bricks(self) -> List[BrickDetection]:
        import os
        debug_path = None
        if self.debug_dir:
            self._capture_count += 1
            debug_path = os.path.join(
                self.debug_dir,
                f"detection_{self._capture_count:04d}.png")

        try:
            detections = self.camera.detect_all_bricks(debug_save_path=debug_path)
        except Exception:
            return []

        return [
            BrickDetection(
                x=d["x"], y=d["y"], z=d["z"],
                yaw=d["yaw"], yaw_deg=d["yaw_deg"],
                confidence=1.0,
                body_id=None,
            )
            for d in detections
        ]

    def verify_placement(self, spec: BrickSpec, body_id: int) -> dict:
        """Verify placement by re-detecting and comparing with target.

        Falls back to ground-truth verification if the vision can't
        see the brick (it may be occluded by the wall).
        """
        detections = self.detect_bricks()
        if not detections:
            # Fall back: use ground-truth for now
            gt = GroundTruthBackend(self.cfg, [body_id])
            return gt.verify_placement(spec, body_id)

        # Find closest detection to the expected target
        target = np.array(spec.center[:2])
        best = min(detections,
                   key=lambda d: math.dist((d.x, d.y), target))
        pos_err_mm = 1000.0 * math.dist(best.pos, spec.center)
        yaw_err = math.degrees(best.yaw - spec.yaw)
        yaw_err = (yaw_err + 180.0) % 360.0 - 180.0
        return {
            "pos_err_mm": round(pos_err_mm, 2),
            "yaw_err_deg": round(yaw_err, 2),
            "acceptable": pos_err_mm < 5.0 and abs(yaw_err) < 3.0,
        }
