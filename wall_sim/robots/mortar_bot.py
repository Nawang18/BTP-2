"""Mortar robot: a kinematic dispenser head that moves over each wall slot
and 'extrudes' a static mortar slab (a thin box) into the joint gap under the
next brick. Deliberately visual/simple per project scope -- no fluid sim.
"""
import numpy as np
import pybullet as p

from ..config import Config
from ..wall_plan import BrickSpec, mortar_center_for
from ..scene import make_static_box


class MortarBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pos = np.array(cfg.mortar_park, dtype=float)
        self.body = make_static_box(tuple(self.pos), (0.08, 0.08, 0.06),
                                    cfg.mortar_head_rgba)

    def _go_gen(self, target, speed=None):
        speed = speed or self.cfg.mortar_speed
        start = self.pos.copy()
        target = np.asarray(target, dtype=float)
        dist = float(np.linalg.norm(target - start))
        steps = max(int(dist / speed * self.cfg.sim_hz), 20)
        for k in range(1, steps + 1):
            self.pos = start + (k / steps) * (target - start)
            p.resetBasePositionAndOrientation(
                self.body, tuple(self.pos), (0, 0, 0, 1))
            yield

    def _wait_gen(self, seconds):
        for _ in range(int(seconds * self.cfg.sim_hz)):
            yield

    def apply_gen(self, spec: BrickSpec):
        """Dispense the mortar bed for one brick target, then RETREAT.

        The retreat matters: the head is a solid static box. If it parks
        over the slot it just served, the arm's hand collides with it during
        the place descent and the brick gets shoved off target. Clear the
        work column (up, then back to park) before the task is done.
        """
        cfg = self.cfg
        mc = mortar_center_for(spec, cfg)
        above = (mc[0], mc[1], mc[2] + 0.35)
        nozzle = (mc[0], mc[1], mc[2] + 0.02)

        yield from self._go_gen(above)
        yield from self._go_gen(nozzle)
        yield from self._wait_gen(0.4)      # extrude
        slab = make_static_box(
            mc, (spec.length + cfg.brick.gap, cfg.brick.width, cfg.brick.gap),
            cfg.mortar_rgba)
        yield from self._wait_gen(0.2)
        yield from self._go_gen(above)      # clear the slot column
        park = self.cfg.mortar_park
        yield from self._go_gen(park)
