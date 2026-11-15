"""Robot workers: adapt task-graph tasks to each robot's action generators.
The executor only knows `begin(task) -> generator` and `kinds`.

Workers may also define `on_brick_skipped(brick_id)`; the executor calls it
when the supervisor gives up on a brick, so each robot can drop whatever it
was holding for that brick.
"""
import math
from typing import Optional

# pyrefly: ignore [missing-import]
import pybullet as p

from .. import metrics
from ..config import Config
from ..scene import Pallet
from ..tasks.task_graph import Task
from ..perception import PerceptionInterface
from .panda_arm import PandaArm, GripFailed
from .shuttle import Shuttle
from .mortar_bot import MortarBot


class ShuttleWorker:
    kinds = {"TRANSPORT"}

    def __init__(self, cfg: Config, shuttle: Shuttle, pallet: Pallet,
                 handover: dict):
        self.cfg, self.shuttle, self.pallet = cfg, shuttle, pallet
        self.handover = handover           # shared with ArmWorker
        self.in_flight = {}                # brick_id -> (body_id, pallet centre)
        self.cancelled = set()             # brick ids the supervisor gave up on

    def _take_from_pallet(self, brick_id):
        """Top brick of the stack, after discarding cancelled bricks above it."""
        while len(self.pallet):
            _, spec, _ = self.pallet.stack[-1]
            if spec.brick_id == brick_id:
                body_id, _, center = self.pallet.pop()
                return body_id, center
            if spec.brick_id not in self.cancelled:
                raise RuntimeError(
                    f"pallet out of sync: top of stack is {spec.brick_id}, "
                    f"task wants {brick_id}")
            body_id, _, _ = self.pallet.pop()
            p.removeBody(body_id)
            print(f"[shuttle] discarded cancelled brick {spec.brick_id} from the stack")
        raise RuntimeError(f"pallet is empty; no brick for {brick_id}")

    def begin(self, task: Task):
        bid = task.brick.brick_id
        # A retry must carry on with the brick it already took off the stack,
        # not pop the next one.
        if bid not in self.in_flight:
            self.in_flight[bid] = self._take_from_pallet(bid)
        body_id, center = self.in_flight[bid]

        def gen():
            to = self.cfg.pedestal_brick_center()
            yield from self.shuttle.deliver_gen(body_id, center, to)
            self.handover["body"] = body_id
            self.handover["brick"] = bid
            del self.in_flight[bid]

        return gen()

    def on_brick_skipped(self, brick_id):
        self.cancelled.add(brick_id)
        if brick_id in self.in_flight:
            body_id, _ = self.in_flight.pop(brick_id)
            if self.shuttle.carrying == body_id:
                self.shuttle.carrying = None
            p.removeBody(body_id)
            print(f"[shuttle] discarded undelivered brick {brick_id}")


class MortarWorker:
    kinds = {"MORTAR"}

    def __init__(self, cfg: Config, bot: MortarBot):
        self.cfg, self.bot = cfg, bot

    def begin(self, task: Task):
        return self.bot.apply_gen(task.brick)


class ArmWorker:
    kinds = {"PLACE"}

    # A brick is re-pickable only while it still rests on the pedestal.
    PEDESTAL_XY_TOL = 0.08
    PEDESTAL_Z_TOL = 0.02

    def __init__(self, cfg: Config, arm: PandaArm, handover, placements):
        self.cfg, self.arm = cfg, arm
        self.handover = handover           # shared with ShuttleWorker
        self.placements = placements       # list of metric records

    def _pick_pose(self, nominal):
        """(centre, yaw) to grasp at. Base worker trusts the nominal pose."""
        return nominal, 0.0

    def _check_on_pedestal(self, body, brick_id):
        nominal = self.cfg.pedestal_brick_center()
        pos, _ = p.getBasePositionAndOrientation(body)
        off_xy = math.hypot(pos[0] - nominal[0], pos[1] - nominal[1])
        if off_xy > self.PEDESTAL_XY_TOL or abs(pos[2] - nominal[2]) > self.PEDESTAL_Z_TOL:
            raise GripFailed(
                f"brick {brick_id} is no longer on the handover pedestal "
                f"(now at x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}); "
                f"it cannot be picked again")

    def begin(self, task: Task):
        spec = task.brick

        def gen():
            body = self.handover["body"]
            if body is None:
                raise RuntimeError(f"no brick on the handover pedestal for {spec.brick_id}")
            if self.arm.held is not None:
                # a previous attempt ended with the brick still in the gripper
                self.arm.held = None
                yield from self.arm.gripper_gen(self.cfg.gripper_open)
                yield from self.arm.wait_gen(0.3)
            self._check_on_pedestal(body, spec.brick_id)

            from_center, pick_yaw = self._pick_pose(self.cfg.pedestal_brick_center())
            yield from self.arm.pick_gen(body, from_center, yaw=pick_yaw)
            yield from self.arm.place_gen(spec.center, yaw=spec.yaw)
            self.handover["body"] = None
            self.handover["brick"] = None
            self.placements.append(metrics.record_placement(spec, body))

        return gen()

    def on_brick_skipped(self, brick_id):
        body = self.handover.get("body")
        if body is None or self.handover.get("brick", brick_id) != brick_id:
            return
        if self.arm.held == body:
            self.arm.held = None
        self.arm._command_gripper(self.cfg.gripper_open)
        p.removeBody(body)
        self.handover["body"] = None
        self.handover["brick"] = None
        print(f"[arm] discarded brick {brick_id} (cleared from the work area)")


class VisionArmWorker(ArmWorker):
    """Vision-guided Arm Worker: uses perception to locate the brick pose
    on the pedestal rather than assuming perfect yaw=0 alignment."""

    def __init__(self, cfg: Config, arm: PandaArm, handover: dict,
                 placements: list, perception: Optional[PerceptionInterface] = None):
        super().__init__(cfg, arm, handover, placements)
        self.perception = perception

    def _pick_pose(self, nominal):
        if self.perception is None:
            return nominal, 0.0
        try:
            detections = self.perception.detect_bricks()
        except Exception as e:
            print(f"[VisionArmWorker] Note: Vision perception fell back to nominal pedestal pose: {e}")
            return nominal, 0.0
        # Only bricks on or immediately near the handover pedestal count
        dist = lambda d: math.hypot(d.x - nominal[0], d.y - nominal[1])
        near = [d for d in detections if dist(d) < 0.15]
        if not near:
            return nominal, 0.0
        best = min(near, key=dist)
        return (best.x, best.y, nominal[2]), best.yaw
