"""Tactical LLM Supervisor — event-driven oversight of the construction site.

Deterministic code drives every tick (executor dispatches tasks, steps physics).
The Gemini LLM is called only at *decision points*: failures, milestone
completions, and the final summary.  This keeps API usage within the free-tier
limits while still showing real LLM reasoning in the loop.

At a failure the LLM runs a short tool-calling conversation: it can inspect
the site (get_world_state, get_perception), sees what those tools return, and
then resolves the failure with retry_task or skip_task.  Whatever it does is
checked against the task graph afterwards; if the failure is still open, the
rule-based policy resolves it, so a build can never stall on the LLM.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .world_state import WorldState, BrickStatus
from .tools import ToolRegistry, SiteContext
from .prompts import SUPERVISOR_SYSTEM_PROMPT
from ..tasks.task_graph import Status

FAILURE_TOOLS = ("get_world_state", "get_perception", "retry_task",
                 "skip_task", "report_status")
REPORT_TOOLS = ("report_status",)
ACCEPTABLE_POS_ERR_MM = 5.0


@dataclass
class SupervisorDecision:
    action: str                       # continue, retry, skip, halt, complete
    target_task: Optional[str] = None
    reason: str = ""
    source: str = "rule"              # rule | llm  (who made the decision)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


class TacticalSupervisor:

    def __init__(self,
                 context: SiteContext,
                 model: str = "gemini-2.5-flash",
                 use_llm: bool = True,
                 max_retries_per_task: int = 2,
                 max_llm_turns: int = 4):
        self.context = context
        self.world_state = context.world_state
        self.executor = context.executor
        self.model_name = model
        self.max_retries = max_retries_per_task
        self.max_llm_turns = max_llm_turns
        self.task_attempts: Dict[str, int] = {}
        self.history: List[SupervisorDecision] = []
        self.llm_calls = 0
        self._last_row_done = -1

        self.registry = ToolRegistry(context)

        if self.executor:
            self.executor.halt_on_failure = False
            self.executor.on_task_start = self._on_task_start
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

    def _on_task_start(self, task_id: str, task: Any):
        ws = self.world_state
        ws.on_robot_busy(task.robot, task_id)
        if task.brick:
            if task.kind == "TRANSPORT":
                ws.on_transport_start(task.brick.brick_id)
            elif task.kind in ("PICK", "PLACE"):
                ws.on_place_start(task.brick.brick_id)

    def _on_task_done(self, task_id: str, task: Any):
        ws = self.world_state
        ws.on_task_completed(task_id)
        ws.on_robot_idle(task.robot)
        if not task.brick:
            return
        bid = task.brick.brick_id
        if task.kind == "TRANSPORT":
            ws.on_transport_done(bid)
        elif task.kind == "PLACE":
            ws.brick_status[bid] = BrickStatus.PLACED
            arm_worker = self.executor.workers.get("arm") if self.executor else None
            placements = getattr(arm_worker, "placements", None)
            if placements and placements[-1].get("brick") == bid:
                latest = placements[-1]
                ws.on_place_done(
                    brick_id=bid, body_id=0,
                    pos_err_mm=latest["pos_err_mm"],
                    yaw_err_deg=latest["yaw_err_deg"],
                    acceptable=latest["pos_err_mm"] < ACCEPTABLE_POS_ERR_MM)

    def _on_task_failed(self, task_id: str, task: Any, error: str):
        attempts = self.task_attempts.get(task_id, 0) + 1
        self.task_attempts[task_id] = attempts
        brick_id = task.brick.brick_id if task.brick else task_id
        self.world_state.on_task_failed(brick_id, error, task_id=task_id,
                                        kind=task.kind, robot=task.robot,
                                        attempts=attempts)
        self.world_state.on_robot_error(task.robot, error)
        print(f"\n[Supervisor] FAILED: {task_id} (attempt {attempts}): {error}")

    # -- Milestone detection -------------------------------------------------

    def _check_row_completed(self) -> Optional[int]:
        """Lowest not-yet-reported row whose bricks are all placed or skipped
        (with at least one placed)."""
        ws = self.world_state
        by_row: Dict[int, List[BrickStatus]] = {}
        for spec in ws.plan:
            by_row.setdefault(spec.row, []).append(ws.brick_status.get(spec.brick_id))
        for row, statuses in sorted(by_row.items()):
            if row <= self._last_row_done:
                continue
            finished = all(s in (BrickStatus.PLACED, BrickStatus.SKIPPED) for s in statuses)
            if finished and BrickStatus.PLACED in statuses:
                self._last_row_done = row
                return row
            if not finished:
                break
        return None

    # -- Decision engine -----------------------------------------------------

    def decide(self) -> SupervisorDecision:
        ws = self.world_state

        # completion: nothing left to run (bricks are placed or skipped)
        if self.executor and not self.executor.graph.has_work():
            placed, total = ws.placed_count(), len(ws.plan)
            skipped = ws.skipped_bricks()
            msg = (f"Build finished — {placed}/{total} bricks placed"
                   + (f", {len(skipped)} skipped ({', '.join(skipped)})" if skipped else "")
                   + f". Mean accuracy: {ws.mean_placement_error():.2f} mm.")
            self._llm_report(f"CONSTRUCTION FINISHED. {msg}\n"
                             f"Give a brief final report using report_status.")
            return SupervisorDecision(action="complete", reason=msg)

        # failure recovery
        failed = self.executor.graph.failed() if self.executor else []
        if failed:
            return self._handle_failure(failed[0].id)

        # normal progression: deterministic advance
        res = self.registry.dispatch("execute_next_tasks", {"max_steps": 480})
        if not isinstance(res, dict) or res.get("error"):
            reason = f"Executor error, halting: {res.get('error') if isinstance(res, dict) else res}"
            print(f"[Supervisor] {reason}")
            return SupervisorDecision(action="halt", reason=reason)

        # deadlock guard: work remains but nothing is running, ready or failed
        g = self.executor.graph
        if (g.has_work() and not self.executor.running
                and not g.ready_tasks() and not g.failed()):
            reason = "Deadlock: tasks remain but none can run. Halting."
            print(f"[Supervisor] {reason}")
            return SupervisorDecision(action="halt", reason=reason)

        # milestone check
        row = self._check_row_completed()
        if row is not None:
            msg = (f"Row {row} complete ({ws.placed_count()}/{len(ws.plan)} "
                   f"bricks placed so far).")
            print(f"[Supervisor] {msg}")
            self._llm_report(f"MILESTONE: {msg}\n"
                             f"Give a brief progress update using report_status.")

        return SupervisorDecision(
            action="continue",
            reason=f"Advanced {res.get('steps_advanced', 0)} ticks.",
            tool_calls=[{"tool": "execute_next_tasks", "result": res}])

    def _task_outcome(self, task_id: str) -> Optional[str]:
        """What actually happened to a failed task: 'retry', 'skip' or None
        if it is still waiting for a decision."""
        status = self.executor.graph.tasks[task_id].status
        if status is Status.FAILED:
            return None
        return "skip" if status is Status.SKIPPED else "retry"

    def _handle_failure(self, task_id: str) -> SupervisorDecision:
        attempts = self.task_attempts.get(task_id, 1)

        # LLM-guided recovery first
        if self.client is not None:
            session = self._llm_failure_session(task_id, attempts)
            outcome = self._task_outcome(task_id)
            if outcome is not None:
                reason = (session or {}).get("text") or f"LLM chose to {outcome} {task_id}."
                print(f"[Supervisor] LLM decision: {outcome} {task_id}")
                return SupervisorDecision(
                    action=outcome, target_task=task_id, reason=reason,
                    source="llm", tool_calls=(session or {}).get("calls", []))
            print(f"[Supervisor] LLM left {task_id} unresolved; applying the rule-based policy.")

        # rule-based policy (offline mode, or the LLM did not resolve it)
        if attempts <= self.max_retries:
            reason = f"Retrying {task_id} (attempt {attempts}/{self.max_retries})."
            self.registry.dispatch("retry_task", {"task_id": task_id})
            action = "retry"
        else:
            reason = f"Skipping {task_id} after {attempts} failures."
            self.registry.dispatch("skip_task", {"task_id": task_id})
            action = "skip"
        print(f"[Supervisor] {reason}")
        if self._task_outcome(task_id) is None:
            reason = f"Could not {action} {task_id}; halting."
            print(f"[Supervisor] {reason}")
            return SupervisorDecision(action="halt", target_task=task_id, reason=reason)
        return SupervisorDecision(action=action, target_task=task_id, reason=reason)

    # -- LLM calls (only at decision points) ---------------------------------

    def _run_tool(self, name: str, args: dict, allowed) -> Any:
        """Execute one LLM tool call, enforcing the limits the LLM must respect."""
        if name not in allowed:
            return {"error": f"Tool '{name}' is not available here. "
                             f"Available: {', '.join(allowed)}"}
        if name == "retry_task":
            tid = args.get("task_id")
            if self.task_attempts.get(tid, 0) > self.max_retries:
                return {"error": f"Retry budget exhausted for {tid} "
                                 f"({self.task_attempts[tid]} failed attempts). "
                                 f"Use skip_task instead."}
        return self.registry.dispatch(name, args)

    def _llm_session(self, prompt: str, tool_names,
                     max_turns: int = 1,
                     done: Optional[Callable[[], bool]] = None) -> Optional[dict]:
        """One tool-calling conversation with the LLM.

        Each turn: the model answers; any tool calls are executed and their
        results are sent back so the next turn can use them. Stops when the
        model makes no tool call, `done()` becomes true, or `max_turns` is hit.
        Returns {"text": last text or None, "calls": [...]}, or None if the
        API call failed.
        """
        if self.client is None:
            return None
        config = {
            "system_instruction": SUPERVISOR_SYSTEM_PROMPT,
            "tools": [{"function_declarations":
                       self.registry.get_tool_definitions(tool_names)}],
            "automatic_function_calling": {"disable": True},
            "temperature": 0.1,
        }
        contents: List[Any] = [{"role": "user", "parts": [{"text": prompt}]}]
        calls: List[Dict[str, Any]] = []
        text = None
        try:
            for _ in range(max_turns):
                response = self.client.models.generate_content(
                    model=self.model_name, contents=contents, config=config)
                self.llm_calls += 1
                fcalls = list(getattr(response, "function_calls", None) or [])
                text = getattr(response, "text", None) or text
                if not fcalls:
                    break

                # keep the model's own turn (with its tool calls) in the history
                candidates = getattr(response, "candidates", None)
                if candidates:
                    contents.append(candidates[0].content)
                else:
                    contents.append({"role": "model", "parts": [
                        {"function_call": {"name": fc.name,
                                           "args": dict(fc.args or {})}}
                        for fc in fcalls]})

                result_parts = []
                for fc in fcalls:
                    args = dict(fc.args or {})
                    result = self._run_tool(fc.name, args, tool_names)
                    calls.append({"tool": fc.name, "args": args, "result": result})
                    result_parts.append({"function_response": {
                        "name": fc.name,
                        "response": result if isinstance(result, dict)
                        else {"result": result}}})
                contents.append({"role": "user", "parts": result_parts})

                if done is not None and done():
                    break
        except Exception as e:
            print(f"[Supervisor] LLM call failed: {e}")
            return None
        return {"text": text, "calls": calls}

    def _llm_failure_session(self, task_id: str, attempts: int) -> Optional[dict]:
        task = self.executor.graph.tasks[task_id]
        error = self.world_state.open_failures.get(task_id, {}).get("error", "unknown")
        prompt = (
            f"Current site state:\n{self.world_state.to_prompt_context()}\n\n"
            f"EVENT: task '{task_id}' ({task.kind}, robot '{task.robot}') FAILED.\n"
            f"Error: {error}\n"
            f"This was failed attempt {attempts}; at most {self.max_retries} "
            f"retries are allowed per task.\n\n"
            f"Resolve this failure by calling retry_task or skip_task for "
            f"'{task_id}'. You may inspect the site first."
        )
        return self._llm_session(
            prompt, FAILURE_TOOLS, max_turns=self.max_llm_turns,
            done=lambda: self._task_outcome(task_id) is not None)

    def _llm_report(self, event: str):
        """Ask the LLM for an operator-facing report on a milestone."""
        if self.client is None:
            return
        prompt = (f"Current site state:\n{self.world_state.to_prompt_context()}\n\n"
                  f"{event}")
        session = self._llm_session(prompt, REPORT_TOOLS, max_turns=1)
        if not session:
            return
        reported = any(c["tool"] == "report_status" for c in session["calls"])
        if not reported and session["text"]:
            # the model answered in plain text instead of calling the tool
            self.registry.dispatch("report_status", {"message": session["text"].strip()})

    # -- Supervision loop ----------------------------------------------------

    def step(self) -> SupervisorDecision:
        decision = self.decide()
        self.history.append(decision)
        return decision

    def run_until_complete(self, max_cycles: int = 500) -> Dict[str, Any]:
        start_time = time.time()
        cycles = 0
        halted = None

        print("\n" + "=" * 60)
        print("[Supervisor] Starting autonomous construction oversight")
        print("=" * 60)

        while cycles < max_cycles:
            cycles += 1
            decision = self.step()
            if decision.action == "complete":
                break
            if decision.action == "halt":
                halted = decision.reason
                break

        elapsed = time.time() - start_time
        ws = self.world_state
        placed, total = ws.placed_count(), len(ws.plan)

        summary = {
            "success": placed == total,
            "cycles": cycles,
            "elapsed_seconds": round(elapsed, 2),
            "placed_bricks": placed,
            "total_bricks": total,
            "skipped_bricks": ws.skipped_bricks(),
            "mean_error_mm": round(ws.mean_placement_error(), 2),
            "max_error_mm": round(ws.max_placement_error(), 2),
            "failed_tasks": ws.failed_tasks,
            "llm_calls": self.llm_calls,
            "halted": halted,
        }

        print("\n" + "=" * 60)
        print(f"[Supervisor] Done in {elapsed:.1f}s ({cycles} cycles)")
        print(f"  Bricks: {placed}/{total}"
              + (f" (skipped: {', '.join(summary['skipped_bricks'])})"
                 if summary["skipped_bricks"] else ""))
        if placed > 0:
            print(f"  Accuracy: mean {summary['mean_error_mm']} mm, "
                  f"max {summary['max_error_mm']} mm")
        if halted:
            print(f"  HALTED: {halted}")
        print("=" * 60 + "\n")

        return summary
