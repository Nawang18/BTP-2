"""Robot workers: adapt task-graph tasks to each robot's action generators.
The executor only knows `begin(task) -> generator` and `kinds`."""
from typing import Optional

from .. import metrics
from ..config import Config
from ..scene import Pallet
from ..tasks.task_graph import Task
from ..perception import PerceptionInterface
from .panda_arm import PandaArm
from .shuttle import Shuttle
from .mortar_bot import MortarBot


class ShuttleWorker:
    kinds = {"TRANSPORT"}

    def __init__(self, cfg: Config, shuttle: Shuttle, pallet: Pallet,
                 handover: dict):
        self.cfg, self.shuttle, self.pallet = cfg, shuttle, pallet
        self.handover = handover           # shared with ArmWorker

    def begin(self, task: Task):
        body_id, spec, center = self.pallet.pop()
        if spec.brick_id != task.brick.brick_id:
            raise RuntimeError(
                f"pallet out of sync: got {spec.brick_id}, "
                f"task wants {task.brick.brick_id}")

        def gen():
            to = self.cfg.pedestal_brick_center()
            yield from self.shuttle.deliver_gen(body_id, center, to)
            self.handover["body"] = body_id

        return gen()


class MortarWorker:
    kinds = {"MORTAR"}

    def __init__(self, cfg: Config, bot: MortarBot):
        self.cfg, self.bot = cfg, bot

    def begin(self, task: Task):
        return self.bot.apply_gen(task.brick)


class ArmWorker:
    kinds = {"PLACE"}

    def __init__(self, cfg: Config, arm: PandaArm, handover, placements):
        self.cfg, self.arm = cfg, arm
        self.handover = handover           # shared with ShuttleWorker
        self.placements = placements       # list of metric records

    def begin(self, task: Task):
        spec = task.brick

        def gen():
            body = self.handover["body"]
            from_center = self.cfg.pedestal_brick_center()
            yield from self.arm.pick_gen(body, from_center, yaw=0.0)
            yield from self.arm.place_gen(spec.center, yaw=spec.yaw)
            self.handover["body"] = None
            self.placements.append(metrics.record_placement(spec, body))

        return gen()


class VisionArmWorker(ArmWorker):
    """Vision-guided Arm Worker: uses perception to locate the brick pose
    on the pedestal rather than assuming perfect yaw=0 alignment."""

    def __init__(self, cfg: Config, arm: PandaArm, handover: dict,
                 placements: list, perception: Optional[PerceptionInterface] = None):
        super().__init__(cfg, arm, handover, placements)
        self.perception = perception

    def begin(self, task: Task):
        spec = task.brick

        def gen():
            body = self.handover["body"]
            from_center = self.cfg.pedestal_brick_center()
            pick_yaw = 0.0

            if self.perception is not None:
                try:
                    detections = self.perception.detect_bricks()
                    # Filter for bricks on or immediately near the handover pedestal
                    pedestal_dets = [
                        d for d in detections
                        if ((d.x - from_center[0])**2 + (d.y - from_center[1])**2)**0.5 < 0.15
                    ]
                    if pedestal_dets:
                        best = min(pedestal_dets, key=lambda d: (d.x - from_center[0])**2 + (d.y - from_center[1])**2)
                        from_center = (best.x, best.y, from_center[2])
                        pick_yaw = best.yaw
                except Exception as e:
                    print(f"[VisionArmWorker] Note: Vision perception fell back to nominal pedestal pose: {e}")

            yield from self.arm.pick_gen(body, from_center, yaw=pick_yaw)
            yield from self.arm.place_gen(spec.center, yaw=spec.yaw)
            self.handover["body"] = None
            self.placements.append(metrics.record_placement(spec, body))

        return gen()
