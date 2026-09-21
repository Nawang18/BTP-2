"""Tactical LLM Supervisor (Layer 2 of the 3-Layer Brain Architecture).

Provides event-driven tactical supervision of the multi-robot construction site:
- Monitors progress, task execution, and placement accuracy.
- Detects failures (GripFailed, vision occlusion, kinematic timeouts).
- Invokes Gemini LLM (or deterministic fallback) with function-calling tools to decide recovery actions (retry, skip, adjust, replan).
- Ensures real-time physics loops remain deterministic while decisions happen at the tactical event level.
"""
from __future__ import annotations

import json
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
    """Action returned by the tactical supervisor."""
    action: str                       # 'continue', 'retry', 'skip', 'adjust', 'halt', 'complete'
    target_task: Optional[str] = None
    reason: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


class TacticalSupervisor:
    """Layer 2 Tactical Supervisor coordinating robots and handling recovery."""

    def __init__(self,
                 context: SiteContext,
                 model: str = "gemini-2.5-flash",
                 use_llm: bool = True,
                 max_retries_per_task: int = 2):
        self.context = context
        self.world_state = context.world_state
        self.executor = context.executor
        self.model_name = model
        self.max_retries = max_retries_per_task
        self.task_attempts: Dict[str, int] = {}
        self.history: List[SupervisorDecision] = []

        # Tool registry
        self.registry = ToolRegistry(context)

        # Wire callbacks to executor if executor is present
        if self.executor:
            self.executor.halt_on_failure = False
            self.executor.on_task_done = self._on_task_done
            self.executor.on_task_failed = self._on_task_failed

        # Check LLM availability
        self.use_llm = use_llm
        self.client = None
        self._init_llm_client()

    def _init_llm_client(self):
        """Initializes Gemini client via google-genai SDK if available and key configured."""
        if not self.use_llm:
            return

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        try:
            from google import genai
            if api_key:
                self.client = genai.Client(api_key=api_key)
                print(f"[Brain Supervisor] Gemini API initialized with model '{self.model_name}'.")
            else:
                print("[Brain Supervisor] Note: GEMINI_API_KEY not found in environment. "
                      "Running in deterministic Tactical Supervisor mode (offline fallback).")
        except ImportError:
            print("[Brain Supervisor] 'google-genai' library not found. "
                  "Running in deterministic Tactical Supervisor mode.")

    # -- Event Callbacks -----------------------------------------------------

    def _on_task_done(self, task_id: str, task: Any):
        """Hook called by executor when a task completes successfully."""
        self.world_state.on_task_completed(task_id)
        # Check if this task finished placing a brick
        if ("place" in task_id or task.kind == "PLACE") and getattr(task, "brick", None):
            bid = task.brick.brick_id
            self.world_state.brick_status[bid] = BrickStatus.PLACED
            arm_worker = self.context.executor.workers.get("arm") if self.context.executor else None
            if arm_worker and getattr(arm_worker, "placements", None) and arm_worker.placements:
                latest = arm_worker.placements[-1]
                if latest.get("brick") == bid:
                    self.world_state.on_place_done(
                        brick_id=bid,
                        body_id=0,
                        pos_err_mm=latest["pos_err_mm"],
                        yaw_err_deg=latest["yaw_err_deg"],
                        acceptable=latest["pos_err_mm"] < 5.0
                    )

    def _on_task_failed(self, task_id: str, task: Any, error: str):
        """Hook called by executor when a task raises an error."""
        self.task_attempts[task_id] = self.task_attempts.get(task_id, 0) + 1
        # Extract brick_id if task name contains it
        brick_id = task_id.split("_")[-1] if "_" in task_id else task_id
        self.world_state.on_task_failed(brick_id, error)
        print(f"\n[Brain Supervisor Event] Task '{task_id}' FAILED (Attempt {self.task_attempts[task_id]}): {error}")

    # -- Tactical Reasoning Engine -------------------------------------------

    def decide(self) -> SupervisorDecision:
        """Evaluates current site conditions and decides next tactical actions."""
        ws = self.world_state

        # Check completion
        if ws.is_complete or (self.executor and not self.executor.graph.has_work()):
            msg = (f"Wall assembly finished. Total bricks placed: {ws.placed_count()}/{len(ws.plan)}. "
                   f"Mean accuracy: {ws.mean_placement_error():.2f} mm.")
            self.registry.dispatch("report_status", {"message": msg})
            return SupervisorDecision(action="complete", reason=msg)

        # Check if there are active failures to resolve
        if ws.has_failures and self.executor:
            failed_tasks = [
                tid for tid, task in self.executor.graph.tasks.items()
                if task.status.value == "failed"
            ]
            if failed_tasks:
                target_tid = failed_tasks[0]
                attempts = self.task_attempts.get(target_tid, 1)

                if attempts <= self.max_retries:
                    reason = f"Recovering from failure in '{target_tid}' (attempt {attempts}/{self.max_retries}). Retrying task."
                    self.registry.dispatch("retry_task", {"task_id": target_tid})
                    self.registry.dispatch("report_status", {"message": reason})
                    return SupervisorDecision(action="retry", target_task=target_tid, reason=reason)
                else:
                    reason = f"Task '{target_tid}' failed {attempts} times consecutively. Skipping task to prevent site deadlock."
                    self.registry.dispatch("skip_task", {"task_id": target_tid})
                    self.registry.dispatch("report_status", {"message": reason})
                    return SupervisorDecision(action="skip", target_task=target_tid, reason=reason)

        # If LLM client is active and live, we can consult Gemini
        if self.client is not None:
            llm_decision = self._llm_decide()
            if llm_decision is not None:
                return llm_decision

        # Default normal progression: advance simulation ticks
        steps_to_run = 240
        res = self.registry.dispatch("execute_next_tasks", {"max_steps": steps_to_run})
        reason = f"Advanced executor by {res.get('steps_advanced', 0)} ticks."
        return SupervisorDecision(
            action="continue",
            reason=reason,
            tool_calls=[{"tool": "execute_next_tasks", "args": {"max_steps": steps_to_run}, "result": res}]
        )

    def _llm_decide(self) -> Optional[SupervisorDecision]:
        """Calls Gemini API with current prompt context and available tools."""
        try:
            prompt = (
                f"{SUPERVISOR_SYSTEM_PROMPT}\n\n"
                f"Current Site Context:\n{self.world_state.to_prompt_context()}\n\n"
                f"Decide the next tactical action using function calling."
            )
            # Query Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "tools": [{"function_declarations": self.registry.get_tool_definitions()}],
                    "temperature": 0.1,
                }
            )

            tool_calls = []
            if response.function_calls:
                for fc in response.function_calls:
                    fname = fc.name
                    fargs = dict(fc.args) if fc.args else {}
                    res = self.registry.dispatch(fname, fargs)
                    tool_calls.append({"tool": fname, "args": fargs, "result": res})
                
                return SupervisorDecision(
                    action="llm_action",
                    reason=f"Executed {len(tool_calls)} LLM tool call(s)",
                    tool_calls=tool_calls
                )
        except Exception as e:
            print(f"[Brain Supervisor Warning] Gemini LLM query failed: {e}. Falling back to deterministic tactical supervisor.")
        return None

    # -- Supervision Execution Loop ------------------------------------------

    def step(self) -> SupervisorDecision:
        """Performs a single supervisory cycle."""
        decision = self.decide()
        self.history.append(decision)
        return decision

    def run_until_complete(self, max_cycles: int = 500) -> Dict[str, Any]:
        """Executes the supervisory loop until the wall is complete or limit reached."""
        start_time = time.time()
        cycles = 0

        print("\n" + "=" * 60)
        print("[Brain Supervisor] Commencing Autonomous Tactical Oversight")
        print("=" * 60)

        while cycles < max_cycles:
            cycles += 1
            decision = self.step()

            if decision.action == "complete":
                break

            # If no work remains in DAG
            if self.executor and not self.executor.graph.has_work():
                break

        elapsed = time.time() - start_time
        placed = self.world_state.placed_count()
        total = len(self.world_state.plan)

        summary = {
            "success": self.world_state.is_complete or placed == total,
            "cycles": cycles,
            "elapsed_seconds": round(elapsed, 2),
            "placed_bricks": placed,
            "total_bricks": total,
            "mean_error_mm": round(self.world_state.mean_placement_error(), 2),
            "max_error_mm": round(self.world_state.max_placement_error(), 2),
            "total_failures": self.world_state.failed_tasks,
        }

        print("\n" + "=" * 60)
        print(f"[Brain Supervisor] Run Complete in {elapsed:.1f}s ({cycles} supervisory cycles)")
        print(f"Wall Progress: {placed}/{total} bricks placed ({self.world_state.wall_progress:.0%})")
        if placed > 0:
            print(f"Mean placement accuracy: {summary['mean_error_mm']} mm | Max: {summary['max_error_mm']} mm")
        print("=" * 60 + "\n")

        return summary
