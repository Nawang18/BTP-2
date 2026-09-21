"""Brain package: Multi-layer LLM orchestration for autonomous construction.

Exports:
- Strategic Planner (Layer 1): PlanSpec, parse_command, llm_plan, LLM_PROMPT
- Tactical Supervisor (Layer 2): TacticalSupervisor, SupervisorDecision
- World State: WorldState, BrickStatus, RobotStatus, PlacedBrick
- Tools: ToolRegistry, SiteContext
- Prompts: PLANNER_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPT
"""
from .planner import PlanSpec, parse_command, llm_plan, LLM_PROMPT
from .world_state import WorldState, BrickStatus, RobotStatus, PlacedBrick
from .tools import ToolRegistry, SiteContext
from .prompts import PLANNER_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPT
from .supervisor import TacticalSupervisor, SupervisorDecision

__all__ = [
    "PlanSpec", "parse_command", "llm_plan", "LLM_PROMPT",
    "WorldState", "BrickStatus", "RobotStatus", "PlacedBrick",
    "ToolRegistry", "SiteContext",
    "PLANNER_SYSTEM_PROMPT", "SUPERVISOR_SYSTEM_PROMPT",
    "TacticalSupervisor", "SupervisorDecision",
]
