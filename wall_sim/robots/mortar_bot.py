"""Mortar dispenser: a ground-based wheeled cart that drives along the wall
and applies mortar from an extending nozzle arm.

The cart travels on the ground behind the wall (higher Y than the wall face),
parallel to the wall's X axis.  A nozzle arm extends forward and upward from
the cart to reach the mortar joint between courses, spawns a visual mortar
slab, then retracts so the arm can place the brick without collision.

Generator-based: one ``yield`` == one simulation tick, same pattern as every
other robot so the executor can advance them all concurrently.
"""
import numpy as np
import pybullet as p

from ..config import Config
from ..wall_plan import BrickSpec, mortar_center_for
from ..scene import make_static_box


class MortarBot:
    BODY_H = 0.08          # cart body center height above ground
    BODY_SIZE = (0.18, 0.12, 0.08)
    NOZZLE_SIZE = (0.03, 0.03, 0.05)

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pos = np.array(cfg.mortar_park, dtype=float)

        self.body = make_static_box(
            (self.pos[0], self.pos[1], self.BODY_H),
            self.BODY_SIZE, cfg.mortar_head_rgba)

        # visual wheels
        self.wheels = []
        wc = p.createVisualShape(p.GEOM_CYLINDER, radius=0.035, length=0.025,
                                 rgbaColor=(0.15, 0.15, 0.15, 1.0))
        for dx, dy in ((-0.06, -0.06), (0.06, -0.06),
                       (-0.06, 0.06), (0.06, 0.06)):
            wid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=wc)
            q = p.getQuaternionFromEuler((0.0, np.pi / 2.0, 0.0))
            p.resetBasePositionAndOrientation(
                wid, (self.pos[0] + dx, self.pos[1] + dy, 0.035), q)
            self.wheels.append((wid, dx, dy, q))

        # nozzle head (starts resting on top of the cart)
        nz = self.BODY_H * 2 + self.NOZZLE_SIZE[2] / 2.0
        self.nozzle_rest = np.array([self.pos[0], self.pos[1], nz])
        self.nozzle_pos = self.nozzle_rest.copy()
        self.nozzle = make_static_box(
            tuple(self.nozzle_pos), self.NOZZLE_SIZE,
            (0.85, 0.65, 0.10, 1.0))

    # -- internal helpers ----------------------------------------------------

    def _sync_body(self):
        p.resetBasePositionAndOrientation(
            self.body, (self.pos[0], self.pos[1], self.BODY_H), (0, 0, 0, 1))
        for wid, dx, dy, q in self.wheels:
            p.resetBasePositionAndOrientation(
                wid, (self.pos[0] + dx, self.pos[1] + dy, 0.035), q)
        nz = self.BODY_H * 2 + self.NOZZLE_SIZE[2] / 2.0
        self.nozzle_rest = np.array([self.pos[0], self.pos[1], nz])

    def _sync_nozzle(self):
        p.resetBasePositionAndOrientation(
            self.nozzle, tuple(self.nozzle_pos), (0, 0, 0, 1))

    def _tick(self):
        self._sync_body()
        self._sync_nozzle()
        yield

    def _drive_gen(self, target_x, speed=None):
        """Drive the cart along the ground to a new X position (Y stays fixed)."""
        speed = speed or self.cfg.mortar_speed
        start = self.pos.copy()
        target = np.array([target_x, self.pos[1]])
        dist = float(np.linalg.norm(target - start))
        if dist < 0.001:
            return
        steps = max(int(dist / speed * self.cfg.sim_hz), 20)
        for k in range(1, steps + 1):
            self.pos = start + (k / steps) * (target - start)
            self.nozzle_pos[:2] = self.pos
            nz = self.BODY_H * 2 + self.NOZZLE_SIZE[2] / 2.0
            self.nozzle_pos[2] = nz
            yield from self._tick()

    def _move_nozzle_gen(self, target, speed=None):
        """Smoothly move the nozzle head from current position to target (3D)."""
        speed = speed or self.cfg.mortar_speed * 0.8
        start = self.nozzle_pos.copy()
        target = np.asarray(target, dtype=float)
        dist = float(np.linalg.norm(target - start))
        if dist < 0.001:
            return
        steps = max(int(dist / speed * self.cfg.sim_hz), 15)
        for k in range(1, steps + 1):
            self.nozzle_pos = start + (k / steps) * (target - start)
            self._sync_nozzle()
            yield

    def _wait_gen(self, seconds):
        for _ in range(int(seconds * self.cfg.sim_hz)):
            yield from self._tick()

    # -- public behaviour ----------------------------------------------------

    def apply_gen(self, spec: BrickSpec):
        """Drive to position, extend nozzle, apply mortar, retract, retreat.

        Steps:
          1. Drive cart to the brick's X position (ground behind wall)
          2. Extend nozzle forward and up to the mortar joint
          3. Spawn a visual mortar slab
          4. Retract nozzle back to cart
          5. Drive cart back toward park to clear the work zone
        """
        cfg = self.cfg
        mc = mortar_center_for(spec, cfg)

        # 1. drive to the right X position
        yield from self._drive_gen(mc[0])

        # 2. extend nozzle: rise, move forward to above mortar, lower to position
        above_mortar = (mc[0], mc[1], mc[2] + 0.08)
        nozzle_work = (mc[0], mc[1], mc[2] + 0.015)
        yield from self._move_nozzle_gen(above_mortar)
        yield from self._move_nozzle_gen(nozzle_work)

        # 3. extrude mortar slab
        yield from self._wait_gen(0.3)
        make_static_box(
            mc, (spec.length + cfg.brick.gap, cfg.brick.width, cfg.brick.gap),
            cfg.mortar_rgba)
        yield from self._wait_gen(0.15)

        # 4. retract nozzle: rise, then pull back to cart
        yield from self._move_nozzle_gen(above_mortar)
        yield from self._move_nozzle_gen(tuple(self.nozzle_rest))

        # 5. retreat toward park so the cart body doesn't block anything
        park_x = cfg.mortar_park[0]
        yield from self._drive_gen(park_x)
