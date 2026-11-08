#!/usr/bin/env python3
"""Demo 4 — LLM-Orchestrated Multi-Robot Wall Assembly

The full pipeline:
  1. Natural-language command from the user
  2. Gemini LLM parses it into a wall specification
  3. Wall planner computes exact target poses for every brick
  4. Task DAG generated (transport → mortar → place per brick)
  5. Three robots registered: shuttle, mortar cart, Panda arm
  6. LLM Supervisor monitors execution, handles failures
  7. Metrics and Gantt chart saved to results/

Usage:
  python run_demo4_llm_orchestrated.py "build a 3 by 2 wall"
  python run_demo4_llm_orchestrated.py --interactive
  python run_demo4_llm_orchestrated.py --no-llm "4x3 wall"
  python run_demo4_llm_orchestrated.py --nogui "3x3"
"""
import argparse
import sys
import time

import pybullet as p

from wall_sim.config import Config
from wall_sim.scene import build_world, Pallet, make_pedestal, draw_slot_outline
from wall_sim.wall_plan import plan_wall
from wall_sim.robots.panda_arm import PandaArm
from wall_sim.robots.shuttle import Shuttle
from wall_sim.robots.mortar_bot import MortarBot
from wall_sim.robots.workers import ShuttleWorker, MortarWorker, ArmWorker
from wall_sim.tasks.wall_dag import build_wall_dag
from wall_sim.tasks.executor import Executor
from wall_sim.brain.planner import llm_plan, parse_command
from wall_sim.brain.world_state import WorldState
from wall_sim.brain.tools import SiteContext
from wall_sim.brain.supervisor import TacticalSupervisor
from wall_sim.perception import GroundTruthBackend
from wall_sim import metrics


def main():
    ap = argparse.ArgumentParser(
        description="LLM-orchestrated multi-robot wall assembly")
    ap.add_argument("command", nargs="*",
                    help="Natural-language build command, e.g. 'build a 3x2 wall'")
    ap.add_argument("--interactive", action="store_true",
                    help="Prompt for command interactively")
    ap.add_argument("--nogui", action="store_true",
                    help="Run headless (no PyBullet GUI)")
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip LLM calls; use deterministic parser + supervisor")
    ap.add_argument("--rows", type=int, default=None,
                    help="Override: number of rows")
    ap.add_argument("--bricks", type=int, default=None,
                    help="Override: bricks per row")
    args = ap.parse_args()

    # ---- 1. Get the natural-language command --------------------------------
    if args.interactive:
        nl_command = input("\nWhat would you like to build?\n> ").strip()
    elif args.command:
        nl_command = " ".join(args.command)
    else:
        nl_command = "build a standard 3 by 3 wall"
        print(f"[Demo4] No command given, using default: \"{nl_command}\"")

    # ---- 2. LLM Planning (or deterministic fallback) -----------------------
    print("\n" + "=" * 60)
    print("[Demo4] PHASE 1 — Planning")
    print("=" * 60)
    print(f"  Command: \"{nl_command}\"")

    if args.no_llm:
        spec = parse_command(nl_command)
        print("  Using deterministic parser (--no-llm)")
    else:
        spec = llm_plan(nl_command)

    # CLI overrides
    if args.rows is not None:
        spec.rows = max(1, min(5, args.rows))
    if args.bricks is not None:
        spec.bricks_per_row = max(1, min(6, args.bricks))

    print(f"  Plan: {spec.bricks_per_row} bricks/row × {spec.rows} rows "
          f"({spec.bond_pattern} bond)")

    # ---- 3. Build the simulation world -------------------------------------
    print("\n" + "=" * 60)
    print("[Demo4] PHASE 2 — Setting up simulation")
    print("=" * 60)

    cfg = Config(wall_bricks_per_row=spec.bricks_per_row,
                 wall_rows=spec.rows)
    gui = not args.nogui
    cid = build_world(cfg, gui=gui)

    # Robots
    arm = PandaArm(cfg)
    shuttle = Shuttle(cfg)
    mortar_bot = MortarBot(cfg)
    print("  Robots: Panda arm, shuttle cart, mortar cart — all loaded")

    # Wall plan
    wall_specs = plan_wall(cfg)
    total_bricks = len(wall_specs)
    print(f"  Wall plan: {total_bricks} bricks (including half-bricks)")

    # Pallet, pedestal, slot outlines
    pallet = Pallet(cfg, wall_specs)
    make_pedestal(cfg)
    for s in wall_specs:
        draw_slot_outline(s, cfg)

    # ---- 4. Task DAG -------------------------------------------------------
    graph = build_wall_dag(wall_specs)
    task_count = len(graph.tasks)
    print(f"  Task DAG: {task_count} tasks "
          f"({total_bricks} transport + {total_bricks} mortar + "
          f"{total_bricks} place)")

    # ---- 5. Workers + Executor ---------------------------------------------
    handover = {"body": None}
    placements = []

    workers = {
        "shuttle": ShuttleWorker(cfg, shuttle, pallet, handover),
        "mortar":  MortarWorker(cfg, mortar_bot),
        "arm":     ArmWorker(cfg, arm, handover, placements),
    }

    executor = Executor(graph, workers, cfg)

    # ---- 6. Brain (LLM Supervisor) -----------------------------------------
    world_state = WorldState()
    world_state.initialize(wall_specs, ["shuttle", "mortar", "arm"])

    perception = GroundTruthBackend(cfg)

    context = SiteContext(
        world_state=world_state,
        executor=executor,
        perception=perception,
        cfg=cfg,
    )

    supervisor = TacticalSupervisor(
        context=context,
        use_llm=not args.no_llm,
    )

    # ---- 7. Run construction -----------------------------------------------
    print("\n" + "=" * 60)
    print("[Demo4] PHASE 3 — Construction")
    print("=" * 60)

    summary = supervisor.run_until_complete()

    # ---- 8. Results --------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Demo4] PHASE 4 — Results")
    print("=" * 60)

    metrics.print_summary(placements)
    metrics.save_reports(placements, executor.timeline,
                         results_dir=cfg.results_dir, tag="demo4")

    if summary["success"]:
        print(f"\n  Wall built successfully! "
              f"{summary['placed_bricks']}/{summary['total_bricks']} bricks.")
    else:
        print(f"\n  Wall incomplete: "
              f"{summary['placed_bricks']}/{summary['total_bricks']} bricks placed.")

    # LLM reports
    if context.user_reports:
        print("\n  --- LLM Supervisor Reports ---")
        for r in context.user_reports:
            print(f"  • {r}")

    print(f"\n  Time: {summary['elapsed_seconds']}s | "
          f"Cycles: {summary['cycles']} | "
          f"Failures: {summary['failed_tasks']}")

    # ---- 9. Keep GUI open --------------------------------------------------
    if gui:
        print("\nDone. Close the PyBullet window to exit.")
        try:
            while p.isConnected():
                p.stepSimulation()
                time.sleep(1.0 / 240.0)
        except KeyboardInterrupt:
            pass
    else:
        p.disconnect()


if __name__ == "__main__":
    main()
