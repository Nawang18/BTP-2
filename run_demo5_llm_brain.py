"""Demo 5 -- Full Autonomous Multi-Robot Construction with 3-Layer LLM Brain Orchestration.

This is the flagship demonstration connecting all components developed across Months 1-4:
1. Layer 1 Strategic Planner (Gemini LLM):
   Translates natural-language user commands (e.g., "Build a 3x2 wall") into structured
   wall specifications (PlanSpec) and task DAG.
2. Layer 2 Tactical Supervisor (Gemini LLM):
   Oversees concurrent multi-robot execution, monitors site world state, detects errors
   (grip slips, occlusion), and invokes recovery tools (retry, skip, adjust, report).
3. Layer 3 Deterministic Motion Controllers:
   Concurrent execution of Shuttle, MortarBot, and PandaArm via DAG scheduler (PyBullet physics).
4. Vision Perception (OpenCV RGB-D):
   Overhead camera verifies and locates brick pose on the handover pedestal in 3D (X, Y, Z, Yaw)
   before gripper acquisition.

Run:
    python run_demo5_llm_brain.py
    python run_demo5_llm_brain.py --command "Build a wall with 3 bricks per row and 2 rows" --nogui
    python run_demo5_llm_brain.py --command "Construct a 2x2 test wall"
"""
import argparse
import os
import sys
import time

# pyrefly: ignore [missing-import]
import pybullet as p

from wall_sim.config import Config
from wall_sim.scene import build_world, make_pedestal, draw_slot_outline, Pallet
from wall_sim.wall_plan import plan_wall
from wall_sim.perception import VisionBackend, GroundTruthBackend
from wall_sim.robots import PandaArm, Shuttle, MortarBot
from wall_sim.robots.workers import ShuttleWorker, MortarWorker, VisionArmWorker
from wall_sim.tasks import build_wall_dag, Executor
from wall_sim.brain import (
    llm_plan, WorldState, TacticalSupervisor, SiteContext
)
from wall_sim import metrics


def main():
    ap = argparse.ArgumentParser(
        description="Demo 5: Autonomous Multi-Robot Construction with 3-Layer LLM Brain Orchestration"
    )
    ap.add_argument(
        "--command",
        default="build a wall, 3 bricks per row, 2 rows",
        help="Natural language instruction for the LLM brain"
    )
    ap.add_argument("--rows", type=int, default=None, help="Override wall rows")
    ap.add_argument("--bricks", type=int, default=None, help="Override bricks per row")
    ap.add_argument("--nogui", action="store_true", help="Run headless in DIRECT mode")
    ap.add_argument("--no-vision", action="store_true", help="Use ground-truth perception instead of camera vision")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    ap.add_argument("--offline", action="store_true", help="Force offline deterministic supervisor mode")
    args = ap.parse_args()

    print("\n" + "=" * 70)
    print("DEMO 5: AUTONOMOUS MULTI-ROBOT WALL ASSEMBLY WITH 3-LAYER LLM BRAIN")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Layer 1 Strategic Planning (Command -> PlanSpec)
    # -----------------------------------------------------------------------
    print(f"\n[Layer 1: Strategic Planner] Processing user command:\n  \"{args.command}\"")
    plan_spec = llm_plan(args.command, model=args.model)

    cfg = Config()
    cfg.wall_rows = args.rows or plan_spec.rows
    cfg.wall_bricks_per_row = args.bricks or plan_spec.bricks_per_row

    print(f"[Layer 1 Output] PlanSpec: {cfg.wall_bricks_per_row} bricks/row, "
          f"{cfg.wall_rows} rows ({plan_spec.bond_pattern} bond).")

    # -----------------------------------------------------------------------
    # Step 2: Physical World & Scene Construction
    # -----------------------------------------------------------------------
    print("\n[Simulation] Initializing PyBullet physics environment...")
    build_world(cfg, gui=not args.nogui)

    plan = plan_wall(cfg)
    for spec in plan:
        draw_slot_outline(spec, cfg)
    make_pedestal(cfg)

    pallet = Pallet(cfg, plan)
    arm = PandaArm(cfg)
    arm.run(arm.home_gen())
    shuttle = Shuttle(cfg)
    mortar = MortarBot(cfg)

    # -----------------------------------------------------------------------
    # Step 3: Perception Backend
    # -----------------------------------------------------------------------
    if args.no_vision:
        print("[Perception] Backend: GroundTruth (direct PyBullet state)")
        perception = GroundTruthBackend(cfg)
    else:
        print("[Perception] Backend: Vision (Overhead RGB-D Camera + OpenCV)")
        perception = VisionBackend(cfg, debug_dir=os.path.join(cfg.results_dir, "demo5_vision"))

    # -----------------------------------------------------------------------
    # Step 4: Layer 3 Deterministic DAG & Robot Workers
    # -----------------------------------------------------------------------
    graph = build_wall_dag(plan)
    handover = {"body": None}
    placements = []
    workers = {
        "shuttle": ShuttleWorker(cfg, shuttle, pallet, handover),
        "mortar": MortarWorker(cfg, mortar),
        "arm": VisionArmWorker(cfg, arm, handover, placements, perception=perception),
    }

    executor = Executor(graph, workers, cfg)
    executor.halt_on_failure = False  # Allow supervisor to handle failures

    # -----------------------------------------------------------------------
    # Step 5: World State & Layer 2 Tactical Supervisor
    # -----------------------------------------------------------------------
    world_state = WorldState()
    world_state.initialize(plan, ["shuttle", "mortar", "arm"])

    site_context = SiteContext(
        world_state=world_state,
        executor=executor,
        perception=perception,
        cfg=cfg
    )

    supervisor = TacticalSupervisor(
        context=site_context,
        model=args.model,
        use_llm=not args.offline
    )

    print(f"\n[System Roster] 3 Robots online: Shuttle, MortarBot, PandaArm")
    print(f"[TaskGraph] {len(graph.tasks)} tasks scheduled across {len(plan)} bricks.")
    print(f"[Supervisor] Tactical LLM Supervisor active (model: {args.model}).")

    # -----------------------------------------------------------------------
    # Step 6: Autonomous Execution Loop under LLM Supervision
    # -----------------------------------------------------------------------
    summary = supervisor.run_until_complete()

    # -----------------------------------------------------------------------
    # Step 7: Verification & Final Metrics
    # -----------------------------------------------------------------------
    tasks_done = sum(1 for t in graph.tasks.values() if t.status.value == "done")
    total_tasks = len(graph.tasks)
    print(f"Tasks Completed: {tasks_done}/{total_tasks}")

    if placements:
        metrics.print_summary(placements)
        metrics.save_reports(
            placements,
            executor.timeline,
            cfg.results_dir,
            tag="demo5_llm_brain"
        )
        print(f"[Outputs] Visual reports & Gantt timeline written to: {cfg.results_dir}/")

    # -----------------------------------------------------------------------
    # Step 8: Interactive Window Hold
    # -----------------------------------------------------------------------
    if not args.nogui:
        print("\nAutonomous construction complete! Close the PyBullet window to exit.")
        try:
            while p.isConnected():
                p.stepSimulation()
                time.sleep(1.0 / cfg.sim_hz)
        except (KeyboardInterrupt, p.error):
            pass

    if p.isConnected():
        p.disconnect()


if __name__ == "__main__":
    main()
