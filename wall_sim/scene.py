"""PyBullet world construction: ground, bricks, pallet stack, pedestal and
wall-slot outlines (the ghost boxes that show where each brick must go).
"""

# pyrefly: ignore [missing-import]
import pybullet as p
# pyrefly: ignore [missing-import]
import pybullet_data

from .config import Config
from .wall_plan import BrickSpec, mortar_center_for


def build_world(cfg: Config, gui: bool = True):
    """Connect, set physics, ground plane. Call before loading any robot."""
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(1.0 / cfg.sim_hz)
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.loadURDF("plane.urdf")
    if gui:
        p.resetDebugVisualizerCamera(1.5, 45, -35, (0.4, 0.05, 0.15))
    return cid


def make_brick(cfg: Config, center, yaw=0.0, length=None, roll=0.0, pitch=0.0, orn=None):
    b = cfg.brick
    length = length if length is not None else b.length
    half = (length / 2.0, b.width / 2.0, b.height / 2.0)
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                              rgbaColor=cfg.brick_rgba)
    if orn is None:
        orn = p.getQuaternionFromEuler((roll, pitch, yaw))
    bid = p.createMultiBody(baseMass=b.mass, baseCollisionShapeIndex=col,
                            baseVisualShapeIndex=vis, basePosition=center,
                            baseOrientation=orn)
    p.changeDynamics(bid, -1, lateralFriction=0.9, spinningFriction=0.3,
                     rollingFriction=0.001)
    return bid


def make_tilted_brick(cfg: Config, center, roll=0.0, pitch=0.0, yaw=0.0, with_support=True):
    """Spawns a brick tilted relative to the ground plane, with an optional static wedge
    underneath so it physically stays at the inclined angle."""
    bid = make_brick(cfg, center, yaw=yaw, roll=roll, pitch=pitch)
    if with_support and (abs(roll) > 0.01 or abs(pitch) > 0.01):
        # Spawn a small support wedge under the raised edge
        import numpy as np
        rot_mat = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler((roll, pitch, yaw)))).reshape((3, 3))
        # Place support block slightly behind center
        support_offset = rot_mat @ np.array([-0.06, 0.0, -cfg.brick.height / 2.0])
        prop_pos = (center[0] + support_offset[0], center[1] + support_offset[1], 0.015)
        make_static_box(prop_pos, (0.04, 0.06, 0.03), (0.4, 0.4, 0.4, 0.8))
    return bid


def make_static_box(center, size, rgba):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s / 2.0 for s in size])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[s / 2.0 for s in size],
                              rgbaColor=rgba)
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis, basePosition=center)


def make_pedestal(cfg: Config):
    size = cfg.pedestal_size
    return make_static_box((cfg.handover_pos[0], cfg.handover_pos[1], size[2] / 2.0),
                           size, cfg.pedestal_rgba)


def draw_slot_outline(spec: BrickSpec, cfg: Config, color=(0, 0, 1)):
    """Blue wireframe box at the target pose -- makes precision visible."""
    b = cfg.brick
    hx, hy, hz = spec.length / 2.0, b.width / 2.0, b.height / 2.0
    cx, cy, cz = spec.center
    corners = [(cx + sx * hx, cy + sy * hy, cz + sz * hz)
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    # corner ordering: index = (sx+)/2*4 + (sy+)/2*2 + (sz+)/2
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, e in edges:
        p.addUserDebugLine(corners[a], corners[e], color, 1.5, 0)


class Pallet:
    """Single brick stack. Built bottom-to-top in REVERSE plan order, so the
    top of the stack is always the next brick the plan needs (LIFO = plan
    order). This keeps the shuttle/arm delivery order in sync with the DAG
    without any bookkeeping in the robots.
    """

    def __init__(self, cfg: Config, specs):
        self.cfg = cfg
        self.stack = []  # bottom -> top of (body_id, spec, center)
        b = cfg.brick
        for i, spec in enumerate(reversed(list(specs))):
            z = b.height / 2.0 + i * (b.height + 0.002)
            center = (cfg.pallet_xy[0], cfg.pallet_xy[1], z)
            rgba = cfg.half_brick_rgba if spec.half else cfg.brick_rgba
            bid = make_brick(cfg, center, yaw=0.0, length=spec.length)
            # half bricks get their own colour so the pattern is readable
            if spec.half:
                p.changeVisualShape(bid, -1, rgbaColor=rgba)
            self.stack.append((bid, spec, center))

    def top_center(self):
        return self.stack[-1][2]

    def pop(self):
        """Remove and return (body_id, spec, center) of the current top brick."""
        return self.stack.pop()

    def __len__(self):
        return len(self.stack)
