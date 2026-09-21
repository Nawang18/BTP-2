"""Transport robot: a kinematic cart that ferries bricks from the pallet
stack to the handover pedestal. Kinematic (teleport-lerp) on purpose -- the
project's claims are coordination + precise placement, not locomotion. The
carried brick is repositioned rigidly on the deck each step.
"""
import numpy as np
import pybullet as p

from ..config import Config
from ..scene import make_static_box


class Shuttle:
    DECK_TOP_Z = 0.12     # chassis top height; brick centre sits on this

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.pos = np.array(cfg.shuttle_park, dtype=float)
        self.carrying = None       # brick body id
        self.body = make_static_box(
            (self.pos[0], self.pos[1], 0.08),
            (0.20, 0.15, 0.08), cfg.shuttle_rgba)
        # four visual-only wheels
        self.wheels = []
        wc = p.createVisualShape(p.GEOM_CYLINDER, radius=0.04, length=0.03,
                                 rgbaColor=(0.1, 0.1, 0.1, 1.0))
        for dx, dy in ((-0.07, -0.08), (0.07, -0.08), (-0.07, 0.08), (0.07, 0.08)):
            wid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=wc)
            q = p.getQuaternionFromEuler((0.0, np.pi / 2.0, 0.0))
            p.resetBasePositionAndOrientation(
                wid, (self.pos[0] + dx, self.pos[1] + dy, 0.04), q)
            self.wheels.append((wid, dx, dy, q))

    # -- internals -----------------------------------------------------------
    def _sync_visuals(self):
        p.resetBasePositionAndOrientation(
            self.body, (self.pos[0], self.pos[1], 0.08), (0, 0, 0, 1))
        for wid, dx, dy, q in self.wheels:
            p.resetBasePositionAndOrientation(
                wid, (self.pos[0] + dx, self.pos[1] + dy, 0.04), q)

    def _update_carried(self):
        if self.carrying is None:
            return
        z = self.DECK_TOP_Z + self.cfg.brick.height / 2.0
        p.resetBasePositionAndOrientation(
            self.carrying, (self.pos[0], self.pos[1], z), (0, 0, 0, 1))
        p.resetBaseVelocity(self.carrying, (0, 0, 0), (0, 0, 0))

    def _tick(self):
        self._sync_visuals()
        self._update_carried()
        yield

    def _go_gen(self, x, y, speed=None):
        speed = speed or self.cfg.shuttle_speed
        start = self.pos.copy()
        target = np.array([x, y])
        dist = float(np.linalg.norm(target - start))
        steps = max(int(dist / speed * self.cfg.sim_hz), 20)
        for k in range(1, steps + 1):
            self.pos = start + (k / steps) * (target - start)
            yield from self._tick()

    def _wait_gen(self, seconds):
        for _ in range(int(seconds * self.cfg.sim_hz)):
            yield from self._tick()

    # -- behaviour -----------------------------------------------------------
    def deliver_gen(self, brick_id, from_center, to_center):
        """Dock beside the stack, load, drive to handover, deposit, return.

        Routing rule: the chassis is solid, so its swept area must never
        overlap the dynamic pallet stack (it would knock the tower over) or
        it would shove the brick resting on the pedestal. We therefore drive
        a rectangular lane: south of the stack -> east lane -> north -> slide
        in west to the deposit stop. The brick itself is teleported the last
        few cm onto the pedestal.
        """
        cfg = self.cfg
        lx, ly = cfg.shuttle_lane
        # 1. dock south of the stack (chassis clear of the bricks)
        dock_y = from_center[1] - cfg.shuttle_load_dock_offset
        yield from self._go_gen(from_center[0], dock_y)
        yield from self._wait_gen(0.2)

        # 2. load: brick hops onto the deck
        deck_z = self.DECK_TOP_Z + cfg.brick.height / 2.0
        p.resetBasePositionAndOrientation(
            brick_id, (self.pos[0], self.pos[1], deck_z), (0, 0, 0, 1))
        self.carrying = brick_id
        yield from self._tick()

        # 3. lane route: east along the south lane, north on the east lane,
        #    then west to the deposit stop
        yield from self._go_gen(lx, dock_y)
        yield from self._go_gen(lx, cfg.shuttle_deposit_stop[1])
        yield from self._go_gen(cfg.shuttle_deposit_stop[0],
                                cfg.shuttle_deposit_stop[1])
        yield from self._wait_gen(0.2)

        # 4. deposit: brick teleports onto the pedestal, shuttle releases
        p.resetBasePositionAndOrientation(
            brick_id, (to_center[0], to_center[1], to_center[2]),
            (0, 0, 0, 1))
        p.resetBaseVelocity(brick_id, (0, 0, 0), (0, 0, 0))
        self.carrying = None
        yield from self._wait_gen(0.2)

        # 5. return to park via the same lanes (reversed)
        park = cfg.shuttle_park
        yield from self._go_gen(lx, cfg.shuttle_deposit_stop[1])
        yield from self._go_gen(lx, park[1])
        yield from self._go_gen(park[0], park[1])
