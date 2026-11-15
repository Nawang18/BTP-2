"""Tools for the LLM Brain.

Tools the Gemini LLM can call (function calling):
- get_world_state: Query current site status, brick progress, robot states
- get_perception: Capture camera vision frame and detect bricks
- retry_task: Reset a failed task back to pending
- skip_task: Give up on a failed task's brick (and any brick it supported)
- report_status: Post an update message to the operator

Internal (used by the supervisor loop, not offered to the LLM):
- execute_next_tasks: Advance the deterministic physics executor
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from .world_state import WorldState
from ..config import Config
from ..perception import PerceptionInterface


@dataclass
class SiteContext:
    """Shared execution context holding references to site state and hardware."""
    world_state: WorldState
    executor: Optional[Any] = None
    perception: Optional[PerceptionInterface] = None
    cfg: Optional[Config] = None
    user_reports: List[str] = field(default_factory=list)


class ToolRegistry:
    """Registry that maps tool names to Python implementations and schemas."""

    def __init__(self, context: SiteContext):
        self.context = context
        self._tools: Dict[str, Callable] = {}
        self._register_defaults()

    def _register_defaults(self):
        ctx = self.context

        def get_world_state() -> dict:
            """Returns the current state of the construction site including wall progress,
            placed bricks, pending tasks, robot statuses, and any failures."""
            ws = ctx.world_state
            return {
                "wall_progress": round(ws.wall_progress, 3),
                "is_complete": ws.is_complete,
                "placed_count": ws.placed_count(),
                "total_bricks": len(ws.plan),
                "mean_pos_err_mm": round(ws.mean_placement_error(), 2),
                "max_pos_err_mm": round(ws.max_placement_error(), 2),
                "brick_status": {bid: s.value for bid, s in ws.brick_status.items()},
                "failed_tasks": {tid: dict(f) for tid, f in ws.open_failures.items()},
                "skipped_bricks": ws.skipped_bricks(),
                "robot_status": {r: s.value for r, s in ws.robot_status.items()},
                "robot_current_task": ws.robot_current_task,
            }

        def get_perception(camera: str = "overhead") -> dict:
            """Captures camera frame and returns detected brick poses in world coordinates."""
            if not ctx.perception:
                return {"error": "No perception backend initialized"}
            try:
                detections = ctx.perception.detect_bricks()
            except Exception as e:
                return {"error": f"Perception failed: {e}"}
            return {"detections": [
                {
                    "x": round(d.x, 3),
                    "y": round(d.y, 3),
                    "z": round(d.z, 3),
                    "yaw_deg": round(d.yaw_deg, 1),
                    "confidence": d.confidence,
                }
                for d in detections
            ]}

        def execute_next_tasks(max_steps: int = 240) -> dict:
            """Steps the executor forward by up to max_steps simulation ticks or until a task fails."""
            if not ctx.executor:
                return {"error": "No executor attached"}

            initial_failed = ctx.world_state.failed_tasks
            initial_done = ctx.world_state.completed_tasks

            executed_steps = 0
            while ctx.executor.graph.has_work() and executed_steps < max_steps:
                ctx.executor.tick()
                executed_steps += 1
                # If a failure occurred, halt step chunk so supervisor can react
                if ctx.world_state.failed_tasks > initial_failed:
                    break

            return {
                "steps_advanced": executed_steps,
                "total_sim_step": ctx.executor.step,
                "new_completed": ctx.world_state.completed_tasks - initial_done,
                "new_failures": ctx.world_state.failed_tasks - initial_failed,
                "has_work_remaining": ctx.executor.graph.has_work(),
                "active_robots": list(ctx.executor.running.keys()),
            }

        def retry_task(task_id: str) -> dict:
            """Resets a FAILED task back to PENDING so the executor will re-attempt it."""
            if not ctx.executor:
                return {"error": "No executor attached"}
            if not ctx.executor.retry_task(task_id):
                return {"status": "failed", "task_id": task_id,
                        "reason": "Task not found or not in FAILED state"}
            task = ctx.executor.graph.tasks[task_id]
            if task.brick is not None:
                ctx.world_state.on_task_retried(task_id, task.brick.brick_id)
            ctx.world_state.on_robot_idle(task.robot)
            return {"status": "retried", "task_id": task_id}

        def skip_task(task_id: str) -> dict:
            """Gives up on the brick of a FAILED task. Bricks above it that lose
            their support are cancelled too."""
            if not ctx.executor:
                return {"error": "No executor attached"}
            task = ctx.executor.graph.tasks.get(task_id)
            if task is None or task.status.value != "failed":
                return {"status": "failed", "task_id": task_id,
                        "reason": "Task not found or not in FAILED state"}
            cancelled = ctx.executor.skip_task(task_id)
            ctx.world_state.on_bricks_skipped(cancelled)
            ctx.world_state.open_failures.pop(task_id, None)
            ctx.world_state.on_robot_idle(task.robot)
            # a skip can abort tasks other robots were running for those bricks
            for robot in ctx.world_state.robot_status:
                if (robot not in ctx.executor.running
                        and ctx.world_state.robot_current_task.get(robot)):
                    ctx.world_state.on_robot_idle(robot)
            return {"status": "skipped", "task_id": task_id,
                    "cancelled_bricks": cancelled}

        def report_status(message: str) -> dict:
            """Logs or broadcasts a status report message for the user/operator."""
            ctx.user_reports.append(message)
            print(f"[Brain Supervisor Report] {message}")
            return {"status": "reported"}

        self._tools = {
            "get_world_state": get_world_state,
            "get_perception": get_perception,
            "execute_next_tasks": execute_next_tasks,
            "retry_task": retry_task,
            "skip_task": skip_task,
            "report_status": report_status,
        }

    _DEFINITIONS = [
        {
            "name": "get_world_state",
            "description": "Returns current site state: wall progress, status of every brick, "
                           "failed tasks with their error messages, skipped bricks, robot statuses.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "get_perception",
            "description": "Captures an overhead camera frame and returns the detected brick poses "
                           "(x, y, z in metres, yaw in degrees) in world coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "camera": {"type": "string", "description": "Camera identifier, default 'overhead'"}
                },
            },
        },
        {
            "name": "retry_task",
            "description": "Resets a FAILED task back to PENDING so the robot attempts it again.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "The ID of the failed task to retry"}
                },
                "required": ["task_id"],
            },
        },
        {
            "name": "skip_task",
            "description": "Gives up on the brick of a FAILED task. The brick is left out of the "
                           "wall, and any brick above it that would no longer be supported is "
                           "cancelled too. Returns the list of cancelled bricks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "The ID of the failed task to skip"}
                },
                "required": ["task_id"],
            },
        },
        {
            "name": "report_status",
            "description": "Sends a status update or progress summary to the human operator.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Informational status message"}
                },
                "required": ["message"],
            },
        },
    ]

    def get_tool_definitions(self, names: Optional[Iterable[str]] = None) -> List[dict]:
        """JSON-Schema declarations for LLM function calling, optionally
        restricted to the given tool names."""
        if names is None:
            return list(self._DEFINITIONS)
        wanted = set(names)
        return [d for d in self._DEFINITIONS if d["name"] in wanted]

    def dispatch(self, tool_name: str, arguments: Optional[dict] = None) -> Any:
        """Executes a tool call by name and returns the result."""
        if tool_name not in self._tools:
            return {"error": f"Unknown tool: {tool_name}"}
        func = self._tools[tool_name]
        args = arguments or {}
        try:
            return func(**args)
        except Exception as e:
            return {"error": f"Execution of {tool_name} failed: {e}"}
