"""Deliberate task failures, for testing the supervisor's recovery behaviour.

A normal run rarely fails, so the retry/skip logic (and the LLM's handling of
it) would otherwise go unexercised. A fault spec makes one task of one brick
fail a chosen number of times:

    place:r0c1        the PLACE of brick r0c1 fails once (a retry then succeeds)
    transport:r1c0:2  the TRANSPORT of r1c0 fails twice
    place:r0c1:always it fails every time, so the supervisor has to skip it

The fault fires a short moment into the task, by raising from the robot's
action generator -- exactly how a real GripFailed reaches the executor.
"""
from dataclasses import dataclass

KIND_TO_ROBOT = {"TRANSPORT": "shuttle", "MORTAR": "mortar", "PICK": "arm", "PLACE": "arm"}
ALWAYS = 10 ** 6


class InjectedFault(RuntimeError):
    pass


@dataclass
class FaultSpec:
    kind: str            # TRANSPORT | MORTAR | PICK | PLACE
    brick_id: str
    times: int = 1
    fired: int = 0

    @property
    def task_id(self):
        return f"{self.kind.lower()}_{self.brick_id}"


def parse_fault(text: str) -> FaultSpec:
    """'kind:brick[:times]' -> FaultSpec. Raises ValueError on bad input."""
    parts = [s.strip() for s in text.split(":")]
    if len(parts) not in (2, 3) or not all(parts):
        raise ValueError(f"'{text}': expected kind:brick or kind:brick:times "
                         f"(e.g. place:r0c1 or place:r0c1:always)")
    kind = parts[0].upper()
    if kind not in KIND_TO_ROBOT:
        raise ValueError(f"'{text}': kind must be one of "
                         f"{', '.join(k.lower() for k in KIND_TO_ROBOT)}")
    times = 1
    if len(parts) == 3:
        if parts[2].lower() == "always":
            times = ALWAYS
        elif parts[2].isdigit() and int(parts[2]) >= 1:
            times = int(parts[2])
        else:
            raise ValueError(f"'{text}': times must be a positive integer or 'always'")
    return FaultSpec(kind, parts[1].lower(), times)


def install_faults(executor, faults, delay_s: float = 0.5):
    """Wrap the workers of `executor` so the given faults fire.

    Raises ValueError if a fault names a task that is not in the plan.
    """
    by_robot = {}
    for f in faults:
        if f.task_id not in executor.graph.tasks:
            bricks = sorted({t.brick.brick_id for t in executor.graph.tasks.values()
                             if t.brick is not None})
            raise ValueError(f"no task '{f.task_id}' in this plan "
                             f"(bricks: {', '.join(bricks)})")
        by_robot.setdefault(KIND_TO_ROBOT[f.kind], []).append(f)

    delay_ticks = max(1, int(delay_s * executor.cfg.sim_hz))
    for robot, robot_faults in by_robot.items():
        _wrap(executor.workers[robot], robot_faults, delay_ticks)


def _wrap(worker, faults, delay_ticks):
    original_begin = worker.begin

    def begin(task):
        gen = original_begin(task)
        fault = next((f for f in faults
                      if f.task_id == task.id and f.fired < f.times), None)
        if fault is None:
            return gen
        fault.fired += 1

        def faulty():
            for tick, _ in enumerate(gen):
                if tick >= delay_ticks:
                    break
                yield
            gen.close()
            raise InjectedFault(
                f"injected test fault: {task.kind} of brick {fault.brick_id} "
                f"aborted (occurrence {fault.fired}"
                + ("" if fault.times >= ALWAYS else f" of {fault.times}") + ")")

        return faulty()

    worker.begin = begin
