"""Tactical LLM Supervisor — event-driven oversight of the construction site.

Deterministic code drives every tick (executor dispatches tasks, steps physics).
The Gemini LLM is called only at *decision points*: failures, milestone
completions, and the final summary.  This keeps API usage within the free-tier
limits (15 RPM) while still showing real LLM reasoning in the loop.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .world_state import WorldState, BrickStatus, RobotStatus
from .tools import ToolRegistry, SiteContext
from .prompts import SUPERVISOR_SYSTEM_PROMPT
from ..config import Config
from ..tasks.executor import Executor


@dataclass
class SupervisorDecision:
    action: str                       # continue, retry, skip, halt, complete
    target_task: Optional[str] = None
    reason: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


class TacticalSupervisor:

    def __init__(self,
                 context: SiteContext,
                 model: str = "gemini-2.0-flash",
                 use_llm: bool = True,
                 max_retries_per_task: int = 2):
        self.context = context
        self.world_state = context.world_state
        self.executor = context.executor
        self.model_name = model
        self.max_retries = max_retries_per_task
        self.task_attempts: Dict[str, int] = {}
        self.history: List[SupervisorDecision] = []
        self._last_placed = 0
        self._last_row_done = -1

        self.registry = ToolRegistry(context)

        if self.executor:
            self.executor.halt_on_failure = False
            self.executor.on_task_done = self._on_task_done
            self.executor.on_task_failed = self._on_task_failed

        self.use_llm = use_llm
        self.client = None
        self._init_llm_client()

    def _init_llm_client(self):
        if not self.use_llm:
            return
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        try:
            from google import genai
            if api_key:
                self.client = genai.Client(api_key=api_key)
                print(f"[Supervisor] Gemini API ready (model: {self.model_name})")
            else:
                print("[Supervisor] GEMINI_API_KEY not set — running deterministic mode.")
        except ImportError:
            print("[Supervisor] google-genai not installed — running deterministic mode.")

    # -- Event Callbacks (wired to executor) ---------------------------------

    def _on_task_done(self, task_id: str, task: Any):
        self.world_state.on_task_completed(task_id)
        if task.kind == "PLACE" and task.brick:
            bid = task.brick.brick_id
            self.world_state.brick_status[bid] = BrickStatus.PLACED
            arm_worker = (self.context.executor.workers.get("arm")
                          if self.context.executor else None)
            if (arm_worker
                    and getattr(arm_worker, "placements", None)
                    and arm_worker.placements):
                latest = arm_worker.placements[-1]
                if latest.get("brick") == bid:
                    self.world_state.on_place_done(
                        brick_id=bid, body_id=0,
                        pos_err_mm=latest["pos_err_mm"],
                        yaw_err_deg=latest["yaw_err_deg"],
                        acceptable=latest["pos_err_mm"] < 5.0)

    def _on_task_failed(self, task_id: str, task: Any, error: str):
        self.task_attempts[task_id] = self.task_attempts.get(task_id, 0) + 1
        brick_id = task.brick.brick_id if task.brick else task_id
        self.world_state.on_task_failed(brick_id, error)
        print(f"\n[Supervisor] FAILED: {task_id} (attempt "
              f"{self.task_attempts[task_id]}): {error}")

    # -- Milestone detection -------------------------------------------------

    def _check_row_completed(self) -> Optional[int]:
        ws = self.world_state
        by_row: Dict[int, List[bool]] = {}
        for spec in ws.plan:
            placed = ws.brick_status.get(spec.brick_id) == BrickStatus.PLACED
            by_row.setdefault(spec.row, []).append(placed)
        for row, statuses in sorted(by_row.items()):
            if all(statuses) and row > self._last_row_done:
                self._last_row_done = row
                return row
        return None

    # -- Decision engine -----------------------------------------------------

    def decide(self) -> SupervisorDecision:
        ws = self.world_state

        # completion
        if ws.is_complete or (self.executor and not self.executor.graph.has_work()):
            placed = ws.placed_count()
            total = len(ws.plan)
            msg = (f"Wall complete — {placed}/{total} bricks placed. "
                   f"Mean accuracy: {ws.mean_placement_error():.2f} mm.")
            self._llm_summary(msg)
            return SupervisorDecision(action="complete", reason=msg)

        # failure recovery
        if ws.has_failures and self.executor:
            failed_tasks = [
                tid for tid, t in self.executor.graph.tasks.items()
                if t.status.value == "failed"
            ]
            if failed_tasks:
                return self._handle_failure(failed_tasks[0])

        # normal progression: deterministic advance
        steps_to_run = 480
        res = self.registry.dispatch("execute_next_tasks",
                                     {"max_steps": steps_to_run})

        # milestone check
        row = self._check_row_completed()
        if row is not None:
            placed = ws.placed_count()
            total = len(ws.plan)
            msg = f"Row {row} complete ({placed}/{total} bricks placed so far)."
            print(f"[Supervisor] {msg}")
            self._llm_milestone(msg)

        reason = f"Advanced {res.get('steps_advanced', 0)} ticks."
        return SupervisorDecision(
            action="continue", reason=reason,
            tool_calls=[{"tool": "execute_next_tasks", "result": res}])

    def _handle_failure(self, task_id: str) -> SupervisorDecision:
        attempts = self.task_attempts.get(task_id, 1)

        # try LLM-guided recovery first
        if self.client is not None:
            llm_dec = self._llm_failure_decision(task_id, attempts)
            if llm_dec is not None:
                return llm_dec

        # deterministic fallback
        if attempts <= self.max_retries:
            reason = f"Retrying {task_id} (attempt {attempts}/{self.max_retries})."
            self.registry.dispatch("retry_task", {"task_id": task_id})
            print(f"[Supervisor] {reason}")
            return SupervisorDecision(action="retry", target_task=task_id,
                                      reason=reason)
        else:
            reason = f"Skipping {task_id} after {attempts} failures."
            self.registry.dispatch("skip_task", {"task_id": task_id})
            print(f"[Supervisor] {reason}")
            return SupervisorDecision(action="skip", target_task=task_id,
                                      reason=reason)

    # -- LLM calls (only at decision points) ---------------------------------

    def _llm_call(self, prompt: str) -> Optional[str]:
        if self.client is None:
            return None
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "tools": [{"function_declarations":
                               self.registry.get_tool_definitions()}],
                    "temperature": 0.1,
                }
            )
            if response.function_calls:
                for fc in response.function_calls:
                    fname = fc.name
                    fargs = dict(fc.args) if fc.args else {}
                    self.registry.dispatch(fname, fargs)
            return (response.text
                    if hasattr(response, "text") and response.text
                    else None)
        except Exception as e:
            print(f"[Supervisor] LLM call failed: {e}")
            return None

    def _llm_failure_decision(self, task_id: str,
                              attempts: int) -> Optional[SupervisorDecision]:
        prompt = (
            f"{SUPERVISOR_SYSTEM_PROMPT}\n\n"
            f"Current Site State:\n{self.world_state.to_prompt_context()}\n\n"
            f"EVENT: Task '{task_id}' FAILED (attempt "
            f"{attempts}/{self.max_retries}).\n"
            f"Decide: retry_task or skip_task? Use function calling."
        )
        result = self._llm_call(prompt)
        if result is not None:
            action = "retry" if attempts <= self.max_retries else "skip"
            return SupervisorDecision(
                action=action, target_task=task_id,
                reason=result or f"LLM decided to {action} {task_id}")
        return None

    def _llm_milestone(self, msg: str):
        if self.client is None:
            return
        prompt = (
            f"{SUPERVISOR_SYSTEM_PROMPT}\n\n"
            f"Current Site State:\n{self.world_state.to_prompt_context()}\n\n"
            f"MILESTONE: {msg}\n"
            f"Provide a brief progress update using report_status."
        )
        self._llm_call(prompt)

    def _llm_summary(self, msg: str):
        if self.client is None:
            return
        prompt = (
            f"{SUPERVISOR_SYSTEM_PROMPT}\n\n"
            f"Final Site State:\n{self.world_state.to_prompt_context()}\n\n"
            f"CONSTRUCTION COMPLETE. {msg}\n"
            f"Provide a brief final report using report_status."
        )
        self._llm_call(prompt)

    # -- Supervision loop ----------------------------------------------------

    def step(self) -> SupervisorDecision:
        decision = self.decide()
        self.history.append(decision)
        return decision

    def run_until_complete(self, max_cycles: int = 500) -> Dict[str, Any]:
        start_time = time.time()
        cycles = 0

        print("\n" + "=" * 60)
        print("[Supervisor] Starting autonomous construction oversight")
        print("=" * 60)

        while cycles < max_cycles:
            cycles += 1
            decision = self.step()

            if decision.action == "complete":
                break
            if self.executor and not self.executor.graph.has_work():
                break

        elapsed = time.time() - start_time
        placed = self.world_state.placed_count()
        total = len(self.world_state.plan)

        summary = {
            "success": placed == total,
            "cycles": cycles,
            "elapsed_seconds": round(elapsed, 2),
            "placed_bricks": placed,
            "total_bricks": total,
            "mean_error_mm": round(self.world_state.mean_placement_error(), 2),
            "max_error_mm": round(self.world_state.max_placement_error(), 2),
            "failed_tasks": self.world_state.failed_tasks,
        }

        print("\n" + "=" * 60)
        print(f"[Supervisor] Done in {elapsed:.1f}s ({cycles} cycles)")
        print(f"  Bricks: {placed}/{total}")
        if placed > 0:
            print(f"  Accuracy: mean {summary['mean_error_mm']} mm, "
                  f"max {summary['max_error_mm']} mm")
        print("=" * 60 + "\n")

        return summary
