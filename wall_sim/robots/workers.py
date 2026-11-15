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
            if self.handover.get("body") is not None:
                raise RuntimeError(
                    f"handover pedestal is still occupied by brick "
                    f"{self.handover.get('brick')}; cannot deliver {bid}")
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
    """PICK lifts the brick off the handover pedestal (freeing it for the
    shuttle's next delivery); PLACE carries it to the wall."""
    kinds = {"PICK", "PLACE"}

    # A brick is re-pickable only while it still rests on the pedestal.
    PEDESTAL_XY_TOL = 0.08
    PEDESTAL_Z_TOL = 0.02

    def __init__(self, cfg: Config, arm: PandaArm, handover, placements):
        self.cfg, self.arm = cfg, arm
        self.handover = handover           # shared with ShuttleWorker
        self.placements = placements       # list of metric records
        self.in_hand = None                # (body_id, brick_id) after a PICK

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
        return self._pick_gen(task) if task.kind == "PICK" else self._place_gen(task)

    def _pick_gen(self, task: Task):
        bid = task.brick.brick_id
        body = self.handover.get("body")
        if body is None or self.handover.get("brick", bid) != bid:
            raise RuntimeError(f"brick {bid} is not on the handover pedestal")
        if self.arm.held is not None:
            # a previous attempt ended with a brick still in the gripper
            self.arm.held = None
            yield from self.arm.gripper_gen(self.cfg.gripper_open)
            yield from self.arm.wait_gen(0.3)
        self._check_on_pedestal(body, bid)

        from_center, pick_yaw = self._pick_pose(self.cfg.pedestal_brick_center())
        yield from self.arm.pick_gen(body, from_center, yaw=pick_yaw)
        # lifted clear: the brick is the arm's now and the pedestal is free
        self.in_hand = (body, bid)
        self.handover["body"] = None
        self.handover["brick"] = None

    def _place_gen(self, task: Task):
        spec = task.brick
        if self.in_hand is None or self.in_hand[1] != spec.brick_id:
            raise RuntimeError(f"arm is not holding brick {spec.brick_id}")
        body = self.in_hand[0]
        if self.arm.held != body:
            raise GripFailed(f"brick {spec.brick_id} is no longer in the gripper; "
                             f"it cannot be placed")
        yield from self.arm.place_gen(spec.center, yaw=spec.yaw)
        self.in_hand = None
        self.placements.append(metrics.record_placement(spec, body))

    def on_brick_skipped(self, brick_id):
        if self.in_hand is not None and self.in_hand[1] == brick_id:
            body = self.in_hand[0]
            self.in_hand = None
            if self.arm.held == body:
                self.arm.held = None
            self.arm._command_gripper(self.cfg.gripper_open)
            p.removeBody(body)
            print(f"[arm] discarded brick {brick_id} from the gripper")
        body = self.handover.get("body")
        if body is not None and self.handover.get("brick") == brick_id:
            if self.arm.held == body:
                self.arm.held = None
                self.arm._command_gripper(self.cfg.gripper_open)
            p.removeBody(body)
            self.handover["body"] = None
            self.handover["brick"] = None
            print(f"[arm] discarded brick {brick_id} from the pedestal")


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
