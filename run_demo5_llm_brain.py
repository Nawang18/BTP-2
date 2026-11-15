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

Testing failure recovery (make a task fail on purpose and watch the supervisor):
    python run_demo5_llm_brain.py --inject-failure place:r0c1            # fails once -> retry
    python run_demo5_llm_brain.py --inject-failure place:r0c1:always     # never works -> skip
    python run_demo5_llm_brain.py --inject-failure transport:r1c0:2 --inject-failure mortar:r0c0

Coordination baseline (robots take turns) to compare makespan against:
    python run_demo5_llm_brain.py --sequential
"""
import argparse
import os
import sys
import time

# pyrefly: ignore [missing-import]
import pybullet as p

from wall_sim.config import Config
from wall_sim.scene import build_world, make_pedestal, draw_slot_outline, Pallet
from wall_sim.structure_plan import (layout_to_specs, rectangular_layout,
                                     validate_layout, render_ascii)
from wall_sim.perception import VisionBackend, GroundTruthBackend
from wall_sim.robots import PandaArm, Shuttle, MortarBot
from wall_sim.robots.workers import ShuttleWorker, MortarWorker, VisionArmWorker
from wall_sim.tasks import build_wall_dag, Executor
from wall_sim.tasks.fault_injection import parse_fault, install_faults
from wall_sim.brain import (
    llm_plan, PlanSpec, WorldState, TacticalSupervisor, SiteContext
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
    ap.add_argument("--model", default="gemini-3.5-flash", help="Gemini model name")
    ap.add_argument("--offline", action="store_true", help="Force offline deterministic supervisor mode")
    ap.add_argument("--sequential", action="store_true",
                    help="Baseline: robots take turns (shuttle waits until the previous brick "
                         "is placed). Default is the parallel plan, where the shuttle fetches "
                         "the next brick while the arm is placing.")
    ap.add_argument("--inject-failure", action="append", default=[], metavar="KIND:BRICK[:TIMES]",
                    help="Make a task fail on purpose to test recovery, e.g. place:r0c1 "
                         "(fails once), transport:r1c0:2, place:r0c1:always. "
                         "KIND is transport, mortar, pick or place. Can be given more than once.")
    args = ap.parse_args()

    try:
        faults = [parse_fault(text) for text in args.inject_failure]
    except ValueError as e:
        ap.error(f"--inject-failure {e}")

    print("\n" + "=" * 70)
    print("DEMO 5: AUTONOMOUS MULTI-ROBOT WALL ASSEMBLY WITH 3-LAYER LLM BRAIN")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Layer 1 Strategic Planning (Command -> PlanSpec)
    # -----------------------------------------------------------------------
    print(f"\n[Layer 1: Strategic Planner] Processing user command:\n  \"{args.command}\"")
    cfg = Config()
    if args.rows or args.bricks:
        plan_spec = PlanSpec(rectangular_layout(args.bricks or 3, args.rows or 3), "cli")
    else:
        plan_spec = llm_plan(args.command, model=args.model, cfg=cfg)

    errors = validate_layout(plan_spec.layout, cfg)
    if errors:
        print("[Layer 1] Layout is not buildable:")
        for err in errors:
            print(f"    - {err}")
        sys.exit(1)

    cfg.wall_rows = plan_spec.rows
    cfg.wall_bricks_per_row = plan_spec.bricks_per_row
    print(f"[Layer 1 Output] {plan_spec.layout.description} "
          f"({plan_spec.layout.brick_count()} bricks, source: {plan_spec.source})")
    print(render_ascii(plan_spec.layout))

    # -----------------------------------------------------------------------
    # Step 2: Physical World & Scene Construction
    # -----------------------------------------------------------------------
    print("\n[Simulation] Initializing PyBullet physics environment...")
    build_world(cfg, gui=not args.nogui)

    plan = layout_to_specs(plan_spec.layout, cfg)
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
    graph = build_wall_dag(plan, parallel=not args.sequential)
    print(f"[TaskGraph] Plan: {'sequential baseline' if args.sequential else 'parallel'}")
    handover = {"body": None}
    placements = []
    workers = {
        "shuttle": ShuttleWorker(cfg, shuttle, pallet, handover),
        "mortar": MortarWorker(cfg, mortar),
        "arm": VisionArmWorker(cfg, arm, handover, placements, perception=perception),
    }

    executor = Executor(graph, workers, cfg)
    executor.halt_on_failure = False  # Allow supervisor to handle failures

    if faults:
        try:
            install_faults(executor, faults)
        except ValueError as e:
            print(f"[Fault injection] {e}")
            p.disconnect()
            sys.exit(1)
        for f in faults:
            print(f"[Fault injection] {f.task_id} will fail "
                  + ("every time" if f.times > 1000 else f"{f.times} time(s)"))

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

    recoveries = [d for d in supervisor.history if d.action in ("retry", "skip", "halt")]
    if recoveries:
        print("\n[Supervisor decisions]")
        for d in recoveries:
            print(f"  {d.action.upper():<5} {d.target_task or '':<18} "
                  f"decided by {d.source.upper():<4} | {d.reason}")
        if summary["skipped_bricks"]:
            print(f"  Bricks left out of the wall: {', '.join(summary['skipped_bricks'])}")
        print(f"  LLM calls used: {summary['llm_calls']}")

    if placements:
        metrics.print_summary(placements)
        metrics.print_coordination(executor.timeline)
        metrics.save_reports(
            placements,
            executor.timeline,
            cfg.results_dir,
            tag="demo5_sequential" if args.sequential else "demo5_llm_brain"
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
