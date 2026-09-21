"""Tool definitions for LLM Brain Function Calling.

Exposes high-level tactical tools that the Gemini LLM can call:
- get_world_state: Query current site status, brick progress, robot states
- get_perception: Capture camera vision frame and detect bricks
- plan_wall: Generate a wall plan specification and TaskGraph
- execute_next_tasks: Advance the deterministic physics executor
- retry_task: Reset a failed task back to pending
- skip_task: Mark a failed task as done (skip) to unblock downstream tasks
- adjust_placement: Apply position/rotation offset corrections
- report_status: Post an update message to the operator
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .world_state import WorldState, BrickStatus, RobotStatus
from ..config import Config
from ..perception import PerceptionInterface, BrickDetection


@dataclass
class SiteContext:
    """Shared execution context holding references to site state and hardware."""
    world_state: WorldState
    executor: Optional[Any] = None
    perception: Optional[PerceptionInterface] = None
    cfg: Optional[Config] = None
    user_reports: List[str] = field(default_factory=list)
    offsets: Dict[str, List[float]] = field(default_factory=dict)


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
                "has_failures": ws.has_failures,
                "placed_count": ws.placed_count(),
                "total_bricks": len(ws.plan),
                "mean_pos_err_mm": round(ws.mean_placement_error(), 2),
                "max_pos_err_mm": round(ws.max_placement_error(), 2),
                "pending_bricks": ws.pending_bricks()[:6],
                "failed_bricks": ws.failed_bricks,
                "robot_status": {r: s.value for r, s in ws.robot_status.items()},
                "robot_current_task": ws.robot_current_task,
            }

        def get_perception(camera: str = "overhead") -> list:
            """Captures camera frame and returns detected brick poses in world coordinates."""
            if not ctx.perception:
                return [{"error": "No perception backend initialized"}]
            try:
                detections = ctx.perception.detect_bricks()
                return [
                    {
                        "x": round(d.x, 3),
                        "y": round(d.y, 3),
                        "z": round(d.z, 3),
                        "yaw_deg": round(d.yaw_deg, 1),
                        "confidence": d.confidence,
                        "body_id": d.body_id,
                    }
                    for d in detections
                ]
            except Exception as e:
                return [{"error": f"Perception failed: {e}"}]

        def plan_wall(bricks_per_row: int = 3, rows: int = 3) -> dict:
            """Generates a wall plan specification with given bricks per row and number of rows."""
            bricks_per_row = max(1, min(6, int(bricks_per_row)))
            rows = max(1, min(5, int(rows)))
            total = rows * bricks_per_row
            return {
                "status": "planned",
                "bricks_per_row": bricks_per_row,
                "rows": rows,
                "total_bricks": total,
                "message": f"Wall plan generated for {bricks_per_row}x{rows} ({total} bricks)."
            }

        def execute_next_tasks(max_steps: int = 240) -> dict:
            """Steps the executor forward by up to max_steps simulation ticks or until a task event occurs.
            Returns the current step count and completed/failed counts."""
            if not ctx.executor:
                return {"error": "No executor attached"}
            
            initial_step = ctx.executor.step
            initial_failed = ctx.world_state.failed_tasks
            initial_done = ctx.world_state.completed_tasks

            # Run up to max_steps
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
            success = ctx.executor.retry_task(task_id)
            if success:
                # Also reset brick status in world state if mapped
                for bid, fail_msg in list(ctx.world_state.failed_bricks.items()):
                    if bid in task_id:
                        ctx.world_state.brick_status[bid] = BrickStatus.PLANNED
                        ctx.world_state.failed_bricks.pop(bid, None)
                return {"status": "retried", "task_id": task_id, "success": True}
            return {"status": "failed", "task_id": task_id, "success": False, "reason": "Task not found or not in FAILED state"}

        def skip_task(task_id: str) -> dict:
            """Marks a FAILED task as DONE (skipped) to unblock downstream dependent tasks."""
            if not ctx.executor:
                return {"error": "No executor attached"}
            success = ctx.executor.skip_task(task_id)
            if success:
                for bid, fail_msg in list(ctx.world_state.failed_bricks.items()):
                    if bid in task_id:
                        ctx.world_state.brick_status[bid] = BrickStatus.FAILED
                return {"status": "skipped", "task_id": task_id, "success": True}
            return {"status": "failed", "task_id": task_id, "success": False, "reason": "Task not found or not in FAILED state"}

        def adjust_placement(task_id: str, offset_x_mm: float = 0.0, offset_y_mm: float = 0.0, offset_z_mm: float = 0.0) -> dict:
            """Applies position offsets (in millimeters) to a brick placement task."""
            ctx.offsets[task_id] = [offset_x_mm / 1000.0, offset_y_mm / 1000.0, offset_z_mm / 1000.0]
            return {
                "status": "adjusted",
                "task_id": task_id,
                "offset_mm": [offset_x_mm, offset_y_mm, offset_z_mm],
            }

        def report_status(message: str) -> dict:
            """Logs or broadcasts a status report message for the user/operator."""
            ctx.user_reports.append(message)
            print(f"[Brain Supervisor Report] {message}")
            return {"status": "reported", "message": message}

        self._tools = {
            "get_world_state": get_world_state,
            "get_perception": get_perception,
            "plan_wall": plan_wall,
            "execute_next_tasks": execute_next_tasks,
            "retry_task": retry_task,
            "skip_task": skip_task,
            "adjust_placement": adjust_placement,
            "report_status": report_status,
        }

    def get_tool_definitions(self) -> List[dict]:
        """Returns OpenAPI/JSON Schema compatible declarations for LLM function calling."""
        return [
            {
                "name": "get_world_state",
                "description": "Returns current site state: wall progress, placed bricks, pending bricks, robot statuses, errors.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                }
            },
            {
                "name": "get_perception",
                "description": "Captures overhead camera frame and returns detected brick poses (X, Y, Z, Yaw) in world coordinates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "camera": {"type": "string", "description": "Camera identifier, default 'overhead'"}
                    }
                }
            },
            {
                "name": "plan_wall",
                "description": "Generates a wall plan specification with given bricks per row and number of rows.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bricks_per_row": {"type": "integer", "description": "Bricks per row (2 to 5)"},
                        "rows": {"type": "integer", "description": "Number of rows / courses (1 to 4)"}
                    },
                    "required": ["bricks_per_row", "rows"]
                }
            },
            {
                "name": "execute_next_tasks",
                "description": "Steps the executor forward by up to max_steps simulation ticks or until an event occurs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max_steps": {"type": "integer", "description": "Maximum simulation ticks to advance (default 240, 1 sec = 240 ticks)"}
                    }
                }
            },
            {
                "name": "retry_task",
                "description": "Resets a FAILED task back to PENDING so the robot will retry it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "The ID of the failed task to retry"}
                    },
                    "required": ["task_id"]
                }
            },
            {
                "name": "skip_task",
                "description": "Marks a FAILED task as DONE (skipped) to unblock downstream dependent tasks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "The ID of the failed task to skip"}
                    },
                    "required": ["task_id"]
                }
            },
            {
                "name": "adjust_placement",
                "description": "Applies a positional correction (in millimeters) to a future brick placement task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "description": "The placement task ID"},
                        "offset_x_mm": {"type": "number", "description": "X offset correction in mm"},
                        "offset_y_mm": {"type": "number", "description": "Y offset correction in mm"},
                        "offset_z_mm": {"type": "number", "description": "Z offset correction in mm"}
                    },
                    "required": ["task_id"]
                }
            },
            {
                "name": "report_status",
                "description": "Sends a status update or progress summary to the human operator.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Informational status message"}
                    },
                    "required": ["message"]
                }
            }
        ]

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
