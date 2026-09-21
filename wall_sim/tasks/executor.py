"""Executor: the deterministic core of the 'brain'. Every tick it
  1. assigns each idle robot its highest-priority ready task,
  2. advances every running robot's action generator by one step,
  3. steps the physics world exactly once.
So all robots move concurrently in one simulation, coordinated only through
the DAG. Timeline data (who did what, when) is recorded for the Gantt report.

Extended from the original with:
  - ``retry_task()``: resets a FAILED task back to PENDING for re-execution
  - ``skip_task()``: marks a FAILED task as DONE to unblock downstream tasks
  - ``on_task_done`` / ``on_task_failed`` callbacks for the brain supervisor
  - ``run()`` no longer halts on first failure — it continues other robots
    and invokes the failure callback so the supervisor can decide
"""
# pyrefly: ignore [missing-import]
import pybullet as p

from .task_graph import Status


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
        self.on_task_done = None      # (task_id: str, task: Task) -> None
        self.on_task_failed = None    # (task_id: str, task: Task, error: str) -> None

        # Controls whether the executor halts on the first failure or
        # continues and lets the supervisor decide.
        self.halt_on_failure = True   # legacy behaviour; supervisor sets False

    def _start_task(self, robot, task):
        gen = self.workers[robot].begin(task)
        task.status = Status.RUNNING
        task.start_step = self.step
        self.running[robot] = (task, gen)

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

    def skip_task(self, task_id: str) -> bool:
        """Mark a FAILED task as DONE to unblock downstream tasks.

        This is for when the supervisor decides a failed brick should be
        skipped rather than retried.  Returns True if successful.
        """
        task = self.graph.tasks.get(task_id)
        if task is None or task.status is not Status.FAILED:
            return False
        task.status = Status.DONE
        task.end_step = self.step
        print(f"[executor] task {task_id} marked DONE (skipped)")
        return True

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
