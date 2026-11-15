"""Executor: the deterministic core of the 'brain'. Every tick it
  1. assigns each idle robot its highest-priority ready task,
  2. advances every running robot's action generator by one step,
  3. steps the physics world exactly once.
So all robots move concurrently in one simulation, coordinated only through
the DAG. Timeline data (who did what, when) is recorded for the Gantt report.

Extended from the original with:
  - ``retry_task()``: resets a FAILED task back to PENDING for re-execution
  - ``skip_task()``: cancels the failed brick and any brick left unsupported
  - ``on_task_done`` / ``on_task_failed`` callbacks for the brain supervisor
  - ``run()`` no longer halts on first failure — it continues other robots
    and invokes the failure callback so the supervisor can decide
"""
# pyrefly: ignore [missing-import]
import pybullet as p

from .task_graph import Status
from ..structure_plan import unsupported_bricks


class Executor:
    def __init__(self, graph, workers, cfg):
        self.graph = graph
        self.workers = workers        # robot name -> worker with .begin(task)
        self.cfg = cfg
        self.step = 0
        self.running = {}             # robot -> (task, generator)
        self.timeline = []

        # Callbacks for brain supervisor integration.
        # Set these to functions that accept (task_id, task) or
        # (task_id, task, error_str) respectively.
        self.on_task_start = None     # (task_id: str, task: Task) -> None
        self.on_task_done = None      # (task_id: str, task: Task) -> None
        self.on_task_failed = None    # (task_id: str, task: Task, error: str) -> None

        # Controls whether the executor halts on the first failure or
        # continues and lets the supervisor decide.
        self.halt_on_failure = True   # legacy behaviour; supervisor sets False

    def _start_task(self, robot, task):
        task.start_step = self.step
        try:
            gen = self.workers[robot].begin(task)
        except Exception as e:
            # A worker that cannot even start (e.g. no brick to fetch) is a
            # task failure the supervisor must see, not a crash of the loop.
            print(f"[executor] task {task.id} FAILED to start: {e}")
            self._finish_task(robot, task, Status.FAILED, str(e))
            return
        task.status = Status.RUNNING
        self.running[robot] = (task, gen)
        if self.on_task_start:
            try:
                self.on_task_start(task.id, task)
            except Exception as e:
                print(f"[executor] on_task_start callback error: {e}")

    def _finish_task(self, robot, task, status, error_msg=None):
        task.status = status
        task.end_step = self.step
        hz = self.cfg.sim_hz
        self.timeline.append({
            "task": task.id, "kind": task.kind, "robot": task.robot,
            "brick": task.brick.brick_id if task.brick else "",
            "row": task.brick.row if task.brick else "",
            "col": task.brick.col if task.brick else "",
            "start_s": round(task.start_step / hz, 3),
            "end_s": round(task.end_step / hz, 3),
            "status": status.value,
        })

        # Notify supervisor callbacks
        if status is Status.DONE and self.on_task_done:
            try:
                self.on_task_done(task.id, task)
            except Exception as e:
                print(f"[executor] on_task_done callback error: {e}")

        if status is Status.FAILED and self.on_task_failed:
            try:
                self.on_task_failed(task.id, task, error_msg or "unknown")
            except Exception as e:
                print(f"[executor] on_task_failed callback error: {e}")

    # -- error recovery API --------------------------------------------------

    def retry_task(self, task_id: str) -> bool:
        """Reset a FAILED task back to PENDING so it will be re-dispatched.

        Returns True if the task was reset, False if the task wasn't found
        or wasn't in FAILED state.
        """
        task = self.graph.tasks.get(task_id)
        if task is None or task.status is not Status.FAILED:
            return False
        task.status = Status.PENDING
        task.start_step = None
        task.end_step = None
        print(f"[executor] task {task_id} reset to PENDING for retry")
        return True

    def skip_task(self, task_id: str):
        """Give up on the brick a FAILED task belongs to.

        Skipping one step of a brick makes no sense on its own (a brick that
        was never delivered cannot be placed), so the whole brick is cancelled:
        every unfinished task of it becomes SKIPPED. Bricks in the rows above
        that would no longer be supported are cancelled with it, so the robots
        never build over a hole.

        Returns the list of cancelled brick ids (failed brick first), or an
        empty list if the task wasn't found or wasn't in FAILED state.
        """
        task = self.graph.tasks.get(task_id)
        if task is None or task.status is not Status.FAILED:
            return []
        if task.brick is None:
            task.status = Status.SKIPPED
            task.end_step = self.step
            return []

        specs = {t.brick.brick_id: t.brick
                 for t in self.graph.tasks.values() if t.brick is not None}
        missing = {bid for bid in specs
                   if any(t.status is Status.SKIPPED
                          for t in self.graph.tasks_for_brick(bid))}
        failed_bid = task.brick.brick_id
        missing.add(failed_bid)
        lost = unsupported_bricks(list(specs.values()), missing,
                                  self.cfg.brick.gap)
        cancelled = [failed_bid] + sorted(lost)

        for bid in cancelled:
            for t in self.graph.tasks_for_brick(bid):
                if t.status in (Status.PENDING, Status.FAILED):
                    t.status = Status.SKIPPED
                    t.end_step = self.step
            for worker in self.workers.values():
                hook = getattr(worker, "on_brick_skipped", None)
                if hook:
                    hook(bid)

        print(f"[executor] brick {failed_bid} skipped"
              + (f"; also cancelled (no support left): {', '.join(sorted(lost))}"
                 if lost else ""))
        return cancelled

    # -- main loop -----------------------------------------------------------

    def tick(self):
        # dispatch: one task per idle robot
        for robot, worker in self.workers.items():
            if robot in self.running:
                continue
            task = self.graph.next_for(robot, kinds=getattr(worker, "kinds", None))
            if task is not None:
                self._start_task(robot, task)

        # advance all running robots one simulation step
        finished = []
        for robot, (task, gen) in self.running.items():
            try:
                next(gen)
            except StopIteration:
                finished.append((robot, task, Status.DONE, None))
            except Exception as e:            # e.g. GripFailed
                finished.append((robot, task, Status.FAILED, str(e)))
                print(f"[executor] task {task.id} FAILED: {e}")
        for robot, task, status, err in finished:
            self._finish_task(robot, task, status, err)
            del self.running[robot]

        self.step += 1
        p.stepSimulation()

    def run(self, max_steps=None):
        limit = max_steps or int(1e9)
        while not self.graph.all_done() and self.step < limit:
            if self.halt_on_failure and self.graph.failed():
                print("[executor] halting: a task failed "
                      "(set halt_on_failure=False for supervisor mode)")
                break
            self.tick()
        return self.graph.all_done()
