#!/usr/bin/env python3
"""Demo 4 (Vision) — LLM-orchestrated wall assembly where the arm must SEE
each brick before picking it.

Same pipeline as Demo 4 (LLM plan -> task DAG -> shuttle / mortar / arm ->
LLM supervisor), with one difference: the shuttle drops every brick on the
handover pedestal with a random position and yaw error. The arm is never told
where the brick landed. Before each PICK the overhead RGB-D camera captures a
frame, OpenCV segments the brick and de-projects it to (x, y, yaw), and the
arm grasps at that estimate. If the camera does not find the brick, the PICK
fails and the supervisor has to deal with it (no blind grasp).

The true pose is read from PyBullet ONLY to score the camera afterwards
(results/vision_demo4_vision.csv).

Usage:
  python run_demo4_vision.py "build a 3 by 2 wall"
  python run_demo4_vision.py --interactive
  python run_demo4_vision.py --no-llm "3x2 wall"
  python run_demo4_vision.py --noise-mm 30 --noise-deg 20 "3x2"
  python run_demo4_vision.py --nogui --no-llm "3x2"
"""
import argparse
import csv
import os
import random
import statistics
import sys
import time

import pybullet as p

from wall_sim.config import Config
from wall_sim.scene import build_world, Pallet, make_pedestal, draw_slot_outline
from wall_sim.structure_plan import (layout_to_specs, rectangular_layout,
                                     validate_layout, render_ascii)
from wall_sim.robots.panda_arm import PandaArm
from wall_sim.robots.shuttle import Shuttle
from wall_sim.robots.mortar_bot import MortarBot
from wall_sim.robots.workers import ShuttleWorker, MortarWorker, VisionArmWorker
from wall_sim.tasks.wall_dag import build_wall_dag
from wall_sim.tasks.executor import Executor
from wall_sim.brain.planner import llm_plan, parse_command, PlanSpec
from wall_sim.brain.world_state import WorldState
from wall_sim.brain.tools import SiteContext
from wall_sim.brain.supervisor import TacticalSupervisor
from wall_sim.perception import VisionBackend
from wall_sim import metrics


class CameraFeed:
    """Keeps the overhead camera visible while the robots work.

    * Every ``every`` sim steps it renders the overhead camera, which refreshes
      PyBullet's "Synthetic Camera RGB / Depth / Segmentation" panels, and
      shows the frame in an OpenCV window ("Overhead camera - live").
    * After each vision detection it shows the annotated frame the arm acted
      on ("Overhead camera - detection").
    """

    def __init__(self, perception: VisionBackend, vision_log: list,
                 every: int = 24, size: int = 320, show_windows: bool = True):
        self.cam = perception.camera
        self.perception = perception
        self.vision_log = vision_log
        self.every = every
        self.size = size
        self.show = show_windows
        self._seen = 0
        self._cv2 = None
        if show_windows:
            try:
                import cv2
                self._cv2 = cv2
            except Exception:
                self.show = False

    def __call__(self, step):
        if step % self.every:
            return
        img = p.getCameraImage(self.size, self.size, self.cam.view_matrix,
                               self.cam.proj_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
        if not self.show:
            return
        import numpy as np
        cv2 = self._cv2
        rgb = np.array(img[2], dtype=np.uint8).reshape(self.size, self.size, 4)[:, :, :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, "LIVE overhead RGB", (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Overhead camera - live", bgr)

        if len(self.vision_log) != self._seen and self.perception.debug_dir:
            self._seen = len(self.vision_log)
            path = os.path.join(self.perception.debug_dir,
                                f"detection_{self.perception._capture_count:04d}.png")
            det = cv2.imread(path)
            if det is not None:
                rec = self.vision_log[-1]
                cv2.putText(det, f"{rec['brick_id']}: err {rec['vis_pos_err_mm']:.1f} mm, "
                                 f"{rec['vis_yaw_err_deg']:+.1f} deg",
                            (10, det.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Overhead camera - detection", det)
        cv2.waitKey(1)

    def close(self):
        if self._cv2 is not None:
            try:
                self._cv2.destroyAllWindows()
            except Exception:
                pass


def print_vision_summary(vision_log, path):
    if not vision_log:
        print("  Vision: no detections recorded.")
        return
    pos = [r["vis_pos_err_mm"] for r in vision_log]
    yaw = [abs(r["vis_yaw_err_deg"]) for r in vision_log]
    drop = [r["drop_offset_mm"] for r in vision_log]
    print(f"  Vision picks: {len(vision_log)}")
    print(f"    Drop offset from nominal : mean {statistics.mean(drop):.1f} mm, "
          f"max {max(drop):.1f} mm  (what a blind grasp would have missed by)")
    print(f"    Camera position error    : mean {statistics.mean(pos):.2f} mm, "
          f"max {max(pos):.2f} mm")
    print(f"    Camera yaw error         : mean {statistics.mean(yaw):.2f} deg, "
          f"max {max(yaw):.2f} deg")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(vision_log[0].keys()))
        w.writeheader()
        w.writerows(vision_log)
    print(f"    Saved per-pick vision log -> {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Demo 4 with vision-guided picking of randomly dropped bricks")
    ap.add_argument("command", nargs="*",
                    help="Natural-language build command, e.g. 'build a 3x2 wall'")
    ap.add_argument("--interactive", action="store_true",
                    help="Prompt for command interactively")
    ap.add_argument("--nogui", action="store_true",
                    help="Run headless (no PyBullet GUI, no camera windows)")
    ap.add_argument("--no-llm", action="store_true",
                    help="Skip LLM calls; use deterministic parser + supervisor")
    ap.add_argument("--rows", type=int, default=None, help="Override: number of rows")
    ap.add_argument("--bricks", type=int, default=None, help="Override: bricks per row")
    ap.add_argument("--noise-mm", type=float, default=25.0,
                    help="Max drop position error on the pedestal, +/- mm (default 25)")
    ap.add_argument("--noise-deg", type=float, default=15.0,
                    help="Max drop yaw error on the pedestal, +/- deg (default 15)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed for the drop noise (default: random)")
    ap.add_argument("--no-camera-window", action="store_true",
                    help="Do not open the OpenCV camera windows")
    args = ap.parse_args()

    # ---- 1. Command ---------------------------------------------------------
    if args.interactive:
        nl_command = input("\nWhat would you like to build?\n> ").strip()
    elif args.command:
        nl_command = " ".join(args.command)
    else:
        nl_command = "build a standard 3 by 2 wall"
        print(f"[Demo4-Vision] No command given, using default: \"{nl_command}\"")

    # ---- 2. Planning --------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Demo4-Vision] PHASE 1 — Planning")
    print("=" * 60)
    print(f"  Command: \"{nl_command}\"")

    cfg = Config()
    if args.rows is not None or args.bricks is not None:
        spec = PlanSpec(rectangular_layout(args.bricks or 3, args.rows or 3), "cli")
    elif args.no_llm:
        spec = parse_command(nl_command)
        print("  Using offline parser (--no-llm)")
    else:
        spec = llm_plan(nl_command, cfg=cfg)

    errors = validate_layout(spec.layout, cfg)
    if errors:
        print("  Layout is not buildable:")
        for err in errors:
            print(f"    - {err}")
        sys.exit(1)

    print(f"  Plan: {spec.layout.description} "
          f"({spec.layout.brick_count()} bricks, source: {spec.source})")
    print(render_ascii(spec.layout))

    # ---- 3. World -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Demo4-Vision] PHASE 2 — Setting up simulation")
    print("=" * 60)

    cfg.wall_rows = spec.rows
    cfg.wall_bricks_per_row = spec.bricks_per_row
    gui = not args.nogui
    build_world(cfg, gui=gui)
    if gui:
        # make sure the synthetic camera preview panels are on
        for flag in (p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                     p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                     p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
            p.configureDebugVisualizer(flag, 1)

    arm = PandaArm(cfg)
    shuttle = Shuttle(cfg)
    mortar_bot = MortarBot(cfg)
    print("  Robots: Panda arm, shuttle cart, mortar cart — all loaded")

    wall_specs = layout_to_specs(spec.layout, cfg)
    total_bricks = len(wall_specs)
    print(f"  Wall plan: {total_bricks} bricks (including half-bricks)")

    pallet = Pallet(cfg, wall_specs)
    make_pedestal(cfg)
    for s in wall_specs:
        draw_slot_outline(s, cfg)

    seed = args.seed if args.seed is not None else random.randrange(10**6)
    rng = random.Random(seed)
    print(f"  Drop noise: +/-{args.noise_mm:.0f} mm, +/-{args.noise_deg:.0f} deg "
          f"(seed {seed})")

    # ---- 4. Task DAG --------------------------------------------------------
    graph = build_wall_dag(wall_specs)
    print(f"  Task DAG: {len(graph.tasks)} tasks")

    # ---- 5. Perception + workers -------------------------------------------
    vision_dir = os.path.join(cfg.results_dir, "demo4_vision")
    perception = VisionBackend(cfg, debug_dir=vision_dir)
    print("  Perception: overhead RGB-D camera + OpenCV (arm picks only what it sees)")

    handover = {"body": None}
    placements = []
    vision_log = []

    workers = {
        "shuttle": ShuttleWorker(cfg, shuttle, pallet, handover,
                                 drop_noise_xy=args.noise_mm / 1000.0,
                                 drop_noise_yaw_deg=args.noise_deg, rng=rng),
        "mortar": MortarWorker(cfg, mortar_bot),
        "arm": VisionArmWorker(cfg, arm, handover, placements,
                               perception=perception, strict=True,
                               vision_log=vision_log),
    }
    executor = Executor(graph, workers, cfg)

    feed = None
    if gui:
        feed = CameraFeed(perception, vision_log,
                          show_windows=not args.no_camera_window)
        executor.tick_hooks.append(feed)

    # ---- 6. Brain -----------------------------------------------------------
    world_state = WorldState()
    world_state.initialize(wall_specs, ["shuttle", "mortar", "arm"])
    context = SiteContext(world_state=world_state, executor=executor,
                          perception=perception, cfg=cfg)
    supervisor = TacticalSupervisor(context=context, use_llm=not args.no_llm)

    # ---- 7. Run -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Demo4-Vision] PHASE 3 — Construction")
    print("=" * 60)
    summary = supervisor.run_until_complete()

    # ---- 8. Results ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Demo4-Vision] PHASE 4 — Results")
    print("=" * 60)
    metrics.print_summary(placements)
    metrics.save_reports(placements, executor.timeline,
                         results_dir=cfg.results_dir, tag="demo4_vision")
    print_vision_summary(vision_log,
                         os.path.join(cfg.results_dir, "vision_demo4_vision.csv"))

    if summary["success"]:
        print(f"\n  Wall built successfully! "
              f"{summary['placed_bricks']}/{summary['total_bricks']} bricks.")
    else:
        print(f"\n  Wall incomplete: "
              f"{summary['placed_bricks']}/{summary['total_bricks']} bricks placed.")
    if context.user_reports:
        print("\n  --- LLM Supervisor Reports ---")
        for r in context.user_reports:
            print(f"  • {r}")
    print(f"\n  Time: {summary['elapsed_seconds']}s | Cycles: {summary['cycles']} | "
          f"Failures: {summary['failed_tasks']}")

    # ---- 9. Keep GUI open ---------------------------------------------------
    if gui:
        print("\nDone. Close the PyBullet window to exit.")
        try:
            step = 0
            while p.isConnected():
                p.stepSimulation()
                step += 1
                if feed is not None:
                    feed(step)
                time.sleep(1.0 / 240.0)
        except (KeyboardInterrupt, p.error):
            pass
        if feed is not None:
            feed.close()
    else:
        p.disconnect()


if __name__ == "__main__":
    main()
