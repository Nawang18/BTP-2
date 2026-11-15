"""Task graph (DAG): the data structure the brain's plan compiles into."""
from dataclasses import dataclass, field
from enum import Enum

from ..wall_plan import BrickSpec


class Status(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"          # cancelled by the supervisor; never executed


# A dependency is satisfied once its task is finished either way. Whether the
# bricks above a skipped brick may still be built is decided separately, by the
# support check in Executor.skip_task -- not by the ordering edges.
FINISHED = (Status.DONE, Status.SKIPPED)


@dataclass
class Task:
    id: str
    kind: str                    # TRANSPORT | MORTAR | PLACE
    robot: str                   # shuttle | mortar | arm
    brick: BrickSpec = None
    deps: list = field(default_factory=list)   # task ids
    status: Status = Status.PENDING
    start_step: int = None
    end_step: int = None


class TaskGraph:
    def __init__(self):
        self.tasks = {}          # id -> Task (insertion ordered)

    def add(self, task_id, kind, robot, brick=None, deps=()):
        if task_id in self.tasks:
            raise ValueError(f"duplicate task id {task_id}")
        for d in deps:
            if d not in self.tasks:
                raise ValueError(f"task {task_id} depends on unknown {d}")
        t = Task(task_id, kind, robot, brick, list(deps))
        self.tasks[task_id] = t
        return t

    def ready_tasks(self):
        """PENDING tasks whose deps are all DONE, in wall order (row, col)."""
        out = []
        for t in self.tasks.values():
            if t.status is not Status.PENDING:
                continue
            if all(self.tasks[d].status in FINISHED for d in t.deps):
                out.append(t)
        key = lambda t: (t.brick.row if t.brick else 0,
                         t.brick.col if t.brick else 0)
        return sorted(out, key=key)

    def next_for(self, robot, kinds=None):
        for t in self.ready_tasks():
            if t.robot == robot and (kinds is None or t.kind in kinds):
                return t
        return None

    def all_done(self):
        return all(t.status in FINISHED for t in self.tasks.values())

    def has_work(self):
        return not self.all_done()

    def tasks_for_brick(self, brick_id):
        return [t for t in self.tasks.values()
                if t.brick is not None and t.brick.brick_id == brick_id]

    def failed(self):
        return [t for t in self.tasks.values() if t.status is Status.FAILED]
