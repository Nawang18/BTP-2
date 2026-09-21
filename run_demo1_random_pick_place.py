"""Demo 1 (Autonomous Perception) -- Pick & Place with Dynamic Brick Detection.

Instead of hardcoded coordinates, the brick is spawned at a random location and
orientation. The Perception sensor identifies the brick's pose (X, Y, Z, Yaw),
and the Franka Panda arm autonomously aligns its gripper, picks it, and places
it into the target wall slot.

Run:
    python run_demo1_random_pick_place.py
    python run_demo1_random_pick_place.py --noise
    python run_demo1_random_pick_place.py --x 0.45 --y -0.15 --yaw_deg 30
"""
import argparse
import math
import random
import time

# pyrefly: ignore [missing-import]
import pybullet as p

from wall_sim.config import Config
from wall_sim.scene import (build_world, make_brick, make_static_box,
                            draw_slot_outline)
from wall_sim.wall_plan import plan_wall, mortar_center_for
from wall_sim.robots import PandaArm
from wall_sim.perception import Perception
from wall_sim import metrics


def spawn_random_brick(cfg: Config, x: float = None, y: float = None,
                       yaw_deg: float = None):
    """Spawns a brick on the ground at a random or specified reachable pose."""
    # Safe dexterous workspace for Franka Panda: radius ~0.40m to 0.65m
    rand_x = x if x is not None else random.uniform(*cfg.spawn_x_range)
    rand_y = y if y is not None else random.uniform(*cfg.spawn_y_range)
    rand_z = cfg.brick.height / 2.0
    rand_yaw = math.radians(yaw_deg) if yaw_deg is not None else random.uniform(-math.pi / 4, math.pi / 4)

    brick_id = make_brick(cfg, (rand_x, rand_y, rand_z), yaw=rand_yaw)
    return brick_id, (rand_x, rand_y, rand_z), rand_yaw


def main():
    ap = argparse.ArgumentParser(description="Autonomous Pick & Place with Perception")
    ap.add_argument("--nogui", action="store_true", help="headless (DIRECT) mode")
    ap.add_argument("--noise", action="store_true", help="simulate realistic perception sensor noise")
    ap.add_argument("--x", type=float, default=None, help="manual brick X position (m)")
    ap.add_argument("--y", type=float, default=None, help="manual brick Y position (m)")
    ap.add_argument("--yaw_deg", type=float, default=None, help="manual brick Yaw rotation (degrees)")
    ap.add_argument("--seed", type=int, default=None, help="random seed")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    cfg = Config()
    build_world(cfg, gui=not args.nogui)

    # 1. Target wall slot
    plan = plan_wall(cfg)
    target_slot = plan[0]
    draw_slot_outline(target_slot, cfg)
    make_static_box(mortar_center_for(target_slot, cfg),
                    (target_slot.length + cfg.brick.gap, cfg.brick.width,
                     cfg.brick.gap), cfg.mortar_rgba)

    # 2. Spawn brick at random/custom position on the table
    brick_id, true_pos, true_yaw = spawn_random_brick(cfg, x=args.x, y=args.y, yaw_deg=args.yaw_deg)
    print(f"\n[Environment] Brick spawned at True Pose: X={true_pos[0]:.3f}m, "
          f"Y={true_pos[1]:.3f}m, Z={true_pos[2]:.3f}m, Yaw={math.degrees(true_yaw):.1f}°")

    # 3. Perception scan: robot identifies the brick without hardcoded coordinates
    perception = Perception(add_noise=args.noise)
    detected = perception.detect_brick(brick_id)
    print(f"[Perception]  Detected Brick Pose:  X={detected['x']:.3f}m, "
          f"Y={detected['y']:.3f}m, Z={detected['z']:.3f}m, Yaw={detected['yaw_deg']:.1f}°")

    # Reachability validation
    if not perception.is_reachable(detected["pos"]):
        print(f"[Perception] ERROR: Target at {detected['pos']} is out of robot reachable workspace!")
        p.disconnect()
        return

    # 4. Robot Arm Execution
    arm = PandaArm(cfg)
    arm.run(arm.home_gen())

    print(f"[Controller]  Aligning gripper to Yaw={detected['yaw_deg']:.1f}° and picking brick...")
    arm.run(arm.pick_gen(brick_id, detected["pos"], yaw=detected["yaw"]))

    print(f"[Controller]  Placing brick into target slot {target_slot.brick_id}...")
    arm.run(arm.place_gen(target_slot.center, yaw=target_slot.yaw))

    # 5. Measure and report accuracy
    rec = metrics.record_placement(target_slot, brick_id)
    metrics.print_summary([rec])
    metrics.save_reports([rec], [], cfg.results_dir, tag="demo1_random")

    # 6. Keep interactive window open until closed by user
    if not args.nogui:
        print("\nSimulation complete! Close the PyBullet window to exit.")
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
