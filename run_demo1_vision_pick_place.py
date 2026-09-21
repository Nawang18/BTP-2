
"""Demo 1 (Real Computer Vision) -- Autonomous Pick & Place via RGB-D Camera + OpenCV.

This demo does NOT query PyBullet for object coordinates.
1. An overhead virtual RGB-D camera captures the workspace.
2. OpenCV color segmentation + contour analysis identifies the brick.
3. 3D de-projection computes real-world (X, Y, Z, Yaw) directly from pixels.
4. The Franka Panda arm aligns its gripper, picks the brick, and places it into the target wall slot.
5. An annotated camera snapshot is saved to results/camera_detection.png.

Run:
    python run_demo1_vision_pick_place.py
    python run_demo1_vision_pick_place.py --x 0.48 --y -0.12 --yaw_deg 30
"""
import argparse
import math
import os
import random
import time

# pyrefly: ignore [missing-import]
import pybullet as p

from wall_sim.config import Config
from wall_sim.scene import (build_world, make_brick, make_static_box,
                            draw_slot_outline)
from wall_sim.wall_plan import plan_wall, mortar_center_for
from wall_sim.robots import PandaArm, GripFailed, PlaceFailed
from wall_sim.camera_vision import OverheadCamera
from wall_sim import metrics


def spawn_random_brick(cfg: Config, x: float = None, y: float = None,
                       yaw_deg: float = None):
    """Spawns a brick at a random or custom reachable position on the ground."""
    rand_x = x if x is not None else random.uniform(*cfg.spawn_x_range)
    rand_y = y if y is not None else random.uniform(*cfg.spawn_y_range)
    rand_z = cfg.brick.height / 2.0
    rand_yaw = math.radians(yaw_deg) if yaw_deg is not None else random.uniform(-math.pi / 4, math.pi / 4)

    brick_id = make_brick(cfg, (rand_x, rand_y, rand_z), yaw=rand_yaw)
    return brick_id, (rand_x, rand_y, rand_z), rand_yaw


def main():
    ap = argparse.ArgumentParser(description="Autonomous Pick & Place with Real Computer Vision (OpenCV)")
    ap.add_argument("--nogui", action="store_true", help="headless (DIRECT) mode")
    ap.add_argument("--x", type=float, default=None, help="manual brick X position (m)")
    ap.add_argument("--y", type=float, default=None, help="manual brick Y position (m)")
    ap.add_argument("--yaw_deg", type=float, default=None, help="manual brick Yaw rotation (degrees)")
    ap.add_argument("--speed", type=float, default=None, help="robot arm cartesian speed (m/s, default 0.15)")
    ap.add_argument("--seed", type=int, default=None, help="random seed")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    cfg = Config()
    if args.speed is not None:
        cfg.arm_speed = args.speed
    build_world(cfg, gui=not args.nogui)

    # 1. Target wall slot
    plan = plan_wall(cfg)
    target_slot = plan[0]
    draw_slot_outline(target_slot, cfg)
    make_static_box(mortar_center_for(target_slot, cfg),
                    (target_slot.length + cfg.brick.gap, cfg.brick.width,
                     cfg.brick.gap), cfg.mortar_rgba)

    # 2. Spawn brick at random or specified position
    brick_id, true_pos, true_yaw = spawn_random_brick(cfg, x=args.x, y=args.y, yaw_deg=args.yaw_deg)
    print(f"\n[Environment] True Physical Pose: X={true_pos[0]:.3f}m, "
          f"Y={true_pos[1]:.3f}m, Z={true_pos[2]:.3f}m, Yaw={math.degrees(true_yaw):.1f}°")

    # Step simulation a few times to let the world settle before taking picture
    for _ in range(5):
        p.stepSimulation()

    # 3. Computer Vision: Overhead Camera takes picture and analyzes pixels with OpenCV
    print("[Vision] Capturing overhead RGB-D camera frame and running OpenCV detection...")
    camera = OverheadCamera(eye_pos=(0.48, 0.0, 1.15), img_width=640, img_height=640)
    debug_img_path = os.path.join(cfg.results_dir, "camera_detection.png")

    try:
        detected = camera.detect_brick_pose(debug_save_path=debug_img_path)
    except RuntimeError as e:
        print(f"[Vision Error] {e}")
        p.disconnect()
        return

    # Compare vision estimate with physical ground truth
    pos_err = 1000.0 * math.dist(detected["pos"][:2], true_pos[:2])
    yaw_err = abs(detected["yaw_deg"] - math.degrees(true_yaw))
    print(f"[Vision Output] Detected Pose:  X={detected['x']:.3f}m, "
          f"Y={detected['y']:.3f}m, Z={detected['z']:.3f}m, Yaw={detected['yaw_deg']:.1f}°")
    print(f"[Vision Accuracy] 2D Pos Estimation Error: {pos_err:.2f} mm | Yaw Error: {yaw_err:.2f}°")

    # 4. Robot Arm Execution with Closed-Loop Failure Detection & Redo
    arm = PandaArm(cfg)
    arm.run(arm.home_gen())

    MAX_PICK_RETRIES = 3
    pick_succeeded = False

    for pick_attempt in range(1, MAX_PICK_RETRIES + 1):
        print(f"\n[Controller] Pick attempt {pick_attempt}/{MAX_PICK_RETRIES} "
              f"(Vision Yaw={detected['yaw_deg']:.1f}°)...")
        pick_target = (detected["x"], detected["y"], cfg.brick.height / 2.0)
        try:
            arm.run(arm.pick_gen(brick_id, pick_target, yaw=detected["yaw"]))
            pick_succeeded = True
            print("[Controller] Brick grasped and lifted successfully.")
            break
        except GripFailed as e:
            print(f"[Fault Detected] Pick failed: {e}")
            if pick_attempt < MAX_PICK_RETRIES:
                print("[Recovery] Retracting arm to home and re-scanning workspace with camera...")
                arm.run(arm.home_gen())
                for _ in range(10):
                    p.stepSimulation()
                try:
                    detected = camera.detect_brick_pose(debug_save_path=debug_img_path)
                    print(f"[Vision Re-acquire] Updated Pose: X={detected['x']:.3f}m, "
                          f"Y={detected['y']:.3f}m, Yaw={detected['yaw_deg']:.1f}°")
                except RuntimeError as ve:
                    print(f"[Vision Error during recovery] {ve}")
            else:
                print(f"[Failure] Exceeded max pick attempts ({MAX_PICK_RETRIES}). Aborting.")

    if not pick_succeeded:
        p.disconnect()
        return

    # Place into target slot with Closed-Loop Failure Detection & Redo
    MAX_PLACE_RETRIES = 2
    place_succeeded = False

    for place_attempt in range(1, MAX_PLACE_RETRIES + 1):
        print(f"\n[Controller] Place attempt {place_attempt}/{MAX_PLACE_RETRIES} "
              f"into target slot {target_slot.brick_id}...")
        try:
            arm.run(arm.place_gen(target_slot.center, yaw=target_slot.yaw))
            place_succeeded = True
            print("[Controller] Brick successfully detached and placed onto target slot.")
            break
        except PlaceFailed as e:
            print(f"[Fault Detected] Place failed: {e}")
            if place_attempt < MAX_PLACE_RETRIES:
                print("[Recovery] Brick stuck in arm. Initiating dislodge maneuver...")
                try:
                    arm.run(arm.dislodge_gen(target_slot.center, yaw=target_slot.yaw))
                    place_succeeded = True
                    break
                except PlaceFailed as de:
                    print(f"[Recovery Warning] Dislodge attempt failed: {de}")
            else:
                print(f"[Failure] Exceeded max place attempts ({MAX_PLACE_RETRIES}).")

    # 5. Metrics report
    rec = metrics.record_placement(target_slot, brick_id)
    metrics.print_summary([rec])
    metrics.save_reports([rec], [], cfg.results_dir, tag="demo1_vision")

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
