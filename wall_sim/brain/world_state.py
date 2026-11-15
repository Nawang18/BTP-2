"""World State Model.

Central data structure that tracks the physical and logical state of the
construction site.  The LLM brain queries this to understand progress,
detect problems, and decide next steps.

The state is updated by the supervisor from executor events (task start,
completion, failure, retry, skip).  It is serialisable to a compact text
format for inclusion in LLM prompts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from ..wall_plan import BrickSpec


class BrickStatus(Enum):
    PLANNED = "planned"        # in the wall plan, not yet attempted
    IN_TRANSIT = "in_transit"  # shuttle is carrying it
    AT_HANDOVER = "at_handover"  # on the pedestal, waiting for arm
    PLACING = "placing"        # arm is placing it
    PLACED = "placed"          # successfully placed
    FAILED = "failed"          # a task for it failed; waiting for a decision
    SKIPPED = "skipped"        # given up on (or cancelled: nothing left to rest on)


class RobotStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"


@dataclass
class PlacedBrick:
    """Record of a placed brick's actual vs. target pose."""
    spec: BrickSpec
    body_id: int
    pos_err_mm: float = 0.0
    yaw_err_deg: float = 0.0
    acceptable: bool = True
    attempts: int = 1


@dataclass
class WorldState:
    """Observable state of the entire construction site."""

    # Wall plan and brick statuses
    plan: List[BrickSpec] = field(default_factory=list)
    brick_status: Dict[str, BrickStatus] = field(default_factory=dict)
    placed_bricks: Dict[str, PlacedBrick] = field(default_factory=dict)
    failed_bricks: Dict[str, str] = field(default_factory=dict)  # id -> error msg
    # task_id -> {brick, kind, robot, error, attempts}; one entry per failure
    # that is still waiting for a retry/skip decision
    open_failures: Dict[str, dict] = field(default_factory=dict)

    # Robot statuses
    robot_status: Dict[str, RobotStatus] = field(default_factory=dict)
    robot_current_task: Dict[str, Optional[str]] = field(default_factory=dict)

    # Counters
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0

    def initialize(self, plan: List[BrickSpec], robot_names: List[str]):
        """Set up state from a wall plan and robot roster."""
        self.plan = list(plan)
        self.brick_status = {
            spec.brick_id: BrickStatus.PLANNED for spec in plan
        }
        self.placed_bricks = {}
        self.failed_bricks = {}
        self.open_failures = {}
        self.robot_status = {
            name: RobotStatus.IDLE for name in robot_names
        }
        self.robot_current_task = {
            name: None for name in robot_names
        }

    # -- update methods (called by the supervisor) --------------------------

    def on_transport_start(self, brick_id: str):
        self.brick_status[brick_id] = BrickStatus.IN_TRANSIT

    def on_transport_done(self, brick_id: str):
        self.brick_status[brick_id] = BrickStatus.AT_HANDOVER

    def on_place_start(self, brick_id: str):
        self.brick_status[brick_id] = BrickStatus.PLACING

    def on_place_done(self, brick_id: str, body_id: int,
                      pos_err_mm: float, yaw_err_deg: float,
                      acceptable: bool):
        self.brick_status[brick_id] = BrickStatus.PLACED
        self.placed_bricks[brick_id] = PlacedBrick(
            spec=self._spec_for(brick_id),
            body_id=body_id,
            pos_err_mm=pos_err_mm,
            yaw_err_deg=yaw_err_deg,
            acceptable=acceptable,
        )

    def on_task_failed(self, brick_id: str, error: str,
                       task_id: Optional[str] = None, kind: str = "",
                       robot: str = "", attempts: int = 1):
        if brick_id in self.brick_status:
            self.brick_status[brick_id] = BrickStatus.FAILED
        self.failed_bricks[brick_id] = error
        self.failed_tasks += 1
        if task_id:
            self.open_failures[task_id] = {
                "brick": brick_id, "kind": kind, "robot": robot,
                "error": error, "attempts": attempts,
            }

    def on_task_retried(self, task_id: str, brick_id: str,
                        resume_status: BrickStatus = BrickStatus.PLANNED):
        self.open_failures.pop(task_id, None)
        self.failed_bricks.pop(brick_id, None)
        if brick_id in self.brick_status:
            self.brick_status[brick_id] = resume_status

    def on_bricks_skipped(self, brick_ids: List[str]):
        for bid in brick_ids:
            if bid in self.brick_status:
                self.brick_status[bid] = BrickStatus.SKIPPED
            self.failed_bricks.pop(bid, None)
        self.open_failures = {tid: f for tid, f in self.open_failures.items()
                              if f["brick"] not in brick_ids}

    def on_task_completed(self, task_id: str):
        self.completed_tasks += 1

    def on_robot_busy(self, robot: str, task_id: str):
        self.robot_status[robot] = RobotStatus.WORKING
        self.robot_current_task[robot] = task_id

    def on_robot_idle(self, robot: str):
        self.robot_status[robot] = RobotStatus.IDLE
        self.robot_current_task[robot] = None

    def on_robot_error(self, robot: str, error: str = ""):
        self.robot_status[robot] = RobotStatus.ERROR

    # -- query methods (used by brain / supervisor) -------------------------

    @property
    def wall_progress(self) -> float:
        """Fraction of the wall that is placed (0.0 to 1.0)."""
        if not self.plan:
            return 0.0
        return self.placed_count() / len(self.plan)

    @property
    def is_complete(self) -> bool:
        return all(s == BrickStatus.PLACED
                   for s in self.brick_status.values())

    @property
    def has_failures(self) -> bool:
        return any(s == BrickStatus.FAILED
                   for s in self.brick_status.values())

    def _with_status(self, status: BrickStatus) -> List[str]:
        return [bid for bid, s in self.brick_status.items() if s == status]

    def pending_bricks(self) -> List[str]:
        return self._with_status(BrickStatus.PLANNED)

    def skipped_bricks(self) -> List[str]:
        return self._with_status(BrickStatus.SKIPPED)

    def placed_count(self) -> int:
        return len(self._with_status(BrickStatus.PLACED))

    def mean_placement_error(self) -> float:
        if not self.placed_bricks:
            return 0.0
        return sum(pb.pos_err_mm for pb in self.placed_bricks.values()) \
               / len(self.placed_bricks)

    def max_placement_error(self) -> float:
        if not self.placed_bricks:
            return 0.0
        return max(pb.pos_err_mm for pb in self.placed_bricks.values())

    # -- serialisation for LLM prompts --------------------------------------

    def to_prompt_context(self) -> str:
        """Compact text summary for inclusion in an LLM prompt."""
        lines = [
            f"=== Construction Site State ===",
            f"Wall progress: {self.wall_progress:.0%} "
            f"({self.placed_count()}/{len(self.plan)} bricks placed)",
        ]

        if self.placed_bricks:
            lines.append(
                f"Placement accuracy: mean {self.mean_placement_error():.1f}mm, "
                f"max {self.max_placement_error():.1f}mm")

        for tid, f in self.open_failures.items():
            lines.append(
                f"FAILED task {tid} ({f['kind']} by {f['robot']}, brick "
                f"{f['brick']}, attempt {f['attempts']}): {f['error']}")

        skipped = self.skipped_bricks()
        if skipped:
            lines.append(f"Skipped bricks (will not be built): {', '.join(skipped)}")

        in_progress = [f"{bid}={s.value}" for bid, s in self.brick_status.items()
                       if s in (BrickStatus.IN_TRANSIT, BrickStatus.AT_HANDOVER,
                                BrickStatus.PLACING)]
        if in_progress:
            lines.append("In progress: " + ", ".join(in_progress))

        pending = self.pending_bricks()
        if pending:
            lines.append(f"Pending: {', '.join(pending[:8])}"
                         + (f" (+{len(pending)-8} more)"
                            if len(pending) > 8 else ""))

        lines.append("Robots: " + ", ".join(
            f"{r}={s.value}"
            + (f" ({self.robot_current_task[r]})"
               if self.robot_current_task.get(r) else "")
            for r, s in self.robot_status.items()))

        return "\n".join(lines)

    # -- helpers ------------------------------------------------------------

    def _spec_for(self, brick_id: str) -> Optional[BrickSpec]:
        for spec in self.plan:
            if spec.brick_id == brick_id:
                return spec
        return None
