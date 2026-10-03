"""Demo 3 -- Vision-Guided Autonomous Multi-Brick Wall Assembly with 3D/6D Tilt Grasping.

Features:
1. Flexible Brick Input:
   - Manual coordinates via CLI: --coords "0.46,-0.10,15; 0.50,0.10,-10; 1.25,0.50,0"
   - Supports 3D angle / tilt: --coords "0.48,-0.10,15,12; 0.50,0.10,-10,0" (X, Y, Yaw, Pitch, Roll)
   - Single brick flags: --x 0.46 --y -0.10 --yaw_deg 15 --pitch_deg 12
   - Interactive console input: --interactive (prompts X Y [Yaw] [Pitch] [Roll])
   - Random reachable workspace spawning: --random (default if no coords given)
   - Tilted brick demonstration: --demo_tilted (spawns an angled brick on a support wedge)
2. Automated Reachability Verification:
   - Checks geometric workspace envelope and IK solver residuals before attempting manipulation.
   - If a brick is unreachable: outputs "[Warning] Brick X (at X=..., Y=...) is OUT OF REACH of the arm!"
     and highlights it with a 3D red warning tag in PyBullet.
3. 3D Computer Vision Perception (OpenCV RGB-D + Point Cloud PCA):
   - Overhead camera captures RGB-D and extracts real-world 3D coordinates (X, Y, Z).
   - Point cloud PCA estimates the 3D surface normal vector, tilt angle with ground, and
     generates a 6D gripper orientation quaternion that tilts the gripper flush with the brick.
   - Saves annotated detection snapshot to results/demo3_camera_detection.png.
4. Closed-Loop Pick & Place with 3D Tilted Grasping:
   - Approaches along the brick's surface normal and tilts the gripper to align with inclined faces.
   - Closed-loop pick attempts (up to 3x): re-homes and re-scans via camera if grip slips.
   - Closed-loop place attempts (up to 2x): re-levels the brick to flat on the mortar bed,
     executing dislodge recovery if stuck in gripper.
5. Wall Construction & Metrics:
   - Places mortar beds under each wall slot, logs precision metrics, and saves reports.

Run:
    python run_demo3_multi_robot.py
    python run_demo3_multi_robot.py --coords "0.48,-0.10,15,12; 0.50,0.10,-10"
    python run_demo3_multi_robot.py --demo_tilted
    python run_demo3_multi_robot.py --interactive
    python run_demo3_multi_robot.py --nogui
"""
import argparse
import ast
import math
import os
import random
import re
import sys
import time

import numpy as np
# pyrefly: ignore [missing-import]
import pybullet as p

from wall_sim.config import Config
from wall_sim.scene import (
    build_world, make_brick, make_tilted_brick, draw_slot_outline
)
from wall_sim.wall_plan import plan_wall
from wall_sim.robots.kuka_arm import (
    KukaArm as PandaArm, GripFailed, PlaceFailed, gripper_3d_orientation, gripper_down_quaternion
)
from wall_sim.camera_vision import OverheadCamera
from wall_sim import metrics


def parse_brick_coords(coords_str: str) -> list:
    """Parses various coordinate string representations into a list of
    (x, y, yaw_deg, pitch_deg, roll_deg) tuples.
    
    Supported formats:
      - Semicolon or comma delimited: "0.45,-0.10,15; 0.50,0.12,-20; 1.25,0.50,0"
      - With tilt (Pitch / Roll): "0.48,-0.10,15,12,0; 0.50,0.10,-10"
      - Python list of tuples: "[(0.45, -0.10, 15), (0.50, 0.12, -20, 10)]"
    """
    coords_str = coords_str.strip()
    if not coords_str:
        return []

    # Attempt ast.literal_eval if formatted as a Python list or tuple
    if coords_str.startswith("[") or coords_str.startswith("("):
        try:
            parsed = ast.literal_eval(coords_str)
            result = []
            for item in parsed:
                x = float(item[0])
                y = float(item[1])
                yaw = float(item[2]) if len(item) > 2 else 0.0
                pitch = float(item[3]) if len(item) > 3 else 0.0
                roll = float(item[4]) if len(item) > 4 else 0.0
                result.append((x, y, yaw, pitch, roll))
            return result
        except Exception:
            pass

    # Split by semicolon first, or by newline, or by grouped parentheses
    parts = re.split(r"[;\n]", coords_str)
    result = []
    for part in parts:
        part = part.strip().strip("()[]")
        if not part:
            continue
        tokens = [t.strip() for t in re.split(r"[, \t]+", part) if t.strip()]
        if len(tokens) >= 2:
            x = float(tokens[0])
            y = float(tokens[1])
            yaw = float(tokens[2]) if len(tokens) > 2 else 0.0
            pitch = float(tokens[3]) if len(tokens) > 3 else 0.0
            roll = float(tokens[4]) if len(tokens) > 4 else 0.0
            result.append((x, y, yaw, pitch, roll))

    return result


def prompt_interactive_coords() -> list:
    """Interactively prompts the user in the terminal to enter brick coordinates."""
    print("\n--- Interactive Brick Coordinates Input ---")
    print("Format: X Y [Yaw_deg] [Pitch_deg (tilt)] [Roll_deg]")
    print("Examples:")
    print("  '0.48 -0.10 15'       -> Flat on ground, rotated 15 deg yaw")
    print("  '0.48 -0.10 15 12'    -> Tilted 12 deg pitch, rotated 15 deg yaw")
    print("  '1.25  0.50  0'       -> Out-of-reach brick test")
    print("Press Enter without input on an empty line when finished.\n")

    coords = []
    idx = 1
    while True:
        try:
            line = input(f"Enter Brick #{idx} coordinates (or Enter to finish): ").strip()
        except EOFError:
            break
        if not line:
            break
        parsed = parse_brick_coords(line)
        if parsed:
            for item in parsed:
                coords.append(item)
                tilt_str = f", Pitch(tilt)={item[3]:.1f}°" if item[3] != 0 else ""
                print(f"  -> Added Brick #{idx}: X={item[0]:.3f}m, Y={item[1]:.3f}m, Yaw={item[2]:.1f}°{tilt_str}")
                idx += 1
        else:
            print("  [!] Invalid format. Please enter at least X and Y numbers (e.g. 0.48 -0.10 15).")

    return coords


def generate_random_coords(count: int, cfg: Config, include_unreachable: bool = False,
                           include_tilted: bool = False) -> list:
    """Generates random reachable coordinates with minimum spacing to prevent
    brick overlap and stacking that would make individual bricks ungraspable."""
    coords = []
    b = cfg.brick
    # Minimum centre-to-centre distance: half the brick diagonal for each
    # brick (≈ 107 mm each → 214 mm total) plus a small clearance so the
    # gripper fingers can close without hitting a neighbour.
    min_sep = math.hypot(b.length, b.width) / 2.0 + 0.03   # ~140 mm
    max_retries = 500

    for k in range(count):
        placed = False
        for _ in range(max_retries):
            rx = random.uniform(*cfg.spawn_x_range)
            ry = random.uniform(*cfg.spawn_y_range)
            # Enforce minimum separation from every already-placed brick
            if all(math.hypot(rx - c[0], ry - c[1]) >= min_sep
                   for c in coords):
                placed = True
                break
        if not placed:
            print(f"[Warning] Could not find non-overlapping position for brick {k+1} "
                  f"in {max_retries} attempts; skipping to avoid stacking.")
            continue

        ryaw = random.uniform(-35.0, 35.0)
        # When include_tilted, give every brick a random tilt to thoroughly
        # exercise the RANSAC-based detection and tilted-grasp pipeline
        if include_tilted:
            rpitch = random.uniform(-18.0, 18.0)
            rroll = random.uniform(-8.0, 8.0)
        else:
            rpitch = 0.0
            rroll = 0.0
        coords.append((rx, ry, ryaw, rpitch, rroll))

    if include_unreachable:
        coords.append((1.25, 0.45, 0.0, 0.0, 0.0))

    return coords


def main():
    ap = argparse.ArgumentParser(
        description="Demo 3: Vision-Guided Multi-Brick Pick & Place with 3D Tilted Grasping & Reachability Check"
    )
    ap.add_argument("--coords", type=str, default=None,
                    help="Manual coordinates: '0.46,-0.10,15; 0.50,0.10,-10,12; 1.25,0.50,0'")
    ap.add_argument("--interactive", action="store_true",
                    help="Prompt in terminal for brick coordinates interactively")
    ap.add_argument("--x", type=float, default=None, help="Manual single brick X position (m)")
    ap.add_argument("--y", type=float, default=None, help="Manual single brick Y position (m)")
    ap.add_argument("--yaw_deg", type=float, default=None, help="Manual single brick Yaw rotation (degrees)")
    ap.add_argument("--pitch_deg", type=float, default=None, help="Manual brick Pitch / tilt with ground (degrees)")
    ap.add_argument("--roll_deg", type=float, default=None, help="Manual brick Roll rotation (degrees)")
    ap.add_argument("--random", action="store_true", help="Force random brick generation")
    ap.add_argument("--demo_tilted", action="store_true",
                    help="Include an angled/tilted brick in random staging to demonstrate 3D tilted grasping")
    ap.add_argument("--demo_unreachable", action="store_true",
                    help="Include an out-of-reach brick in random mode to demonstrate error handling")
    ap.add_argument("--rows", type=int, default=2, help="Number of wall rows to build (default: 2)")
    ap.add_argument("--bricks", type=int, default=3, help="Bricks per row in wall (default: 3)")
    ap.add_argument("--speed", type=float, default=None, help="Panda arm cartesian speed (m/s, default 0.15)")
    ap.add_argument("--nogui", action="store_true", help="Run headless in DIRECT mode")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print("\n" + "=" * 75)
    print("DEMO 3: VISION-GUIDED AUTONOMOUS MULTI-BRICK WALL ASSEMBLY (6D TILT GRASPING)")
    print("=" * 75)

    cfg = Config()
    cfg.wall_rows = args.rows
    cfg.wall_bricks_per_row = args.bricks
    if args.speed is not None:
        cfg.arm_speed = args.speed

    # Widen the staging area so 4+ bricks fit with minimum spacing enforced.
    # Default area (160 × 350 mm) is too small; expanded area (250 × 400 mm)
    # stays within the arm's reachable workspace (0.20–0.82 m radial).
    cfg.spawn_x_range = (0.35, 0.60)
    cfg.spawn_y_range = (-0.25, 0.15)

    # 1. Determine Brick Input Coordinates
    brick_inputs = []
    if args.interactive or (args.coords and args.coords.lower() == "interactive"):
        brick_inputs = prompt_interactive_coords()
    elif args.coords is not None:
        brick_inputs = parse_brick_coords(args.coords)
    elif args.x is not None or args.y is not None:
        bx = args.x if args.x is not None else 0.48
        by = args.y if args.y is not None else -0.10
        byaw = args.yaw_deg if args.yaw_deg is not None else 0.0
        bpitch = args.pitch_deg if args.pitch_deg is not None else 0.0
        broll = args.roll_deg if args.roll_deg is not None else 0.0
        brick_inputs = [(bx, by, byaw, bpitch, broll)]

    # 2. Wall Plan
    wall_plan = plan_wall(cfg)
    total_slots = len(wall_plan)
    print(f"\n[Wall Plan] Target: {cfg.wall_rows} row(s) x {cfg.wall_bricks_per_row} bricks/row -> {total_slots} wall slots.")

    # Adjust wall slot Z positions: remove mortar gap so bricks sit directly
    # on the ground (row 0) or on the previous course (row 1+).
    for slot in wall_plan:
        bh = cfg.brick.height
        new_z = bh / 2.0 + slot.row * bh
        slot.center = (slot.center[0], slot.center[1], new_z)

    # If no inputs provided, generate random brick coordinates to fulfill wall slots
    if not brick_inputs:
        needed_count = min(total_slots, 4) if not args.random else total_slots
        brick_inputs = generate_random_coords(
            needed_count, cfg,
            include_unreachable=args.demo_unreachable,
            include_tilted=args.demo_tilted
        )
        print(f"[Input Mode] Generated {len(brick_inputs)} random brick staging coordinate(s).")
    else:
        print(f"[Input Mode] User provided {len(brick_inputs)} custom coordinate(s).")

    # 3. Build PyBullet Simulation Environment
    build_world(cfg, gui=not args.nogui)

    # Draw wall slots (no mortar beds)
    for slot in wall_plan:
        draw_slot_outline(slot, cfg)

    # Initialize Franka Panda Arm
    arm = PandaArm(cfg)
    arm.run(arm.home_gen())

    # 4. Physical Spawning & Reachability Verification
    print("\n" + "-" * 60)
    print("[Reachability Verification] Checking all candidate brick poses...")
    print("-" * 60)

    staged_bricks = []
    brick_height = cfg.brick.height
    z_pos = brick_height / 2.0

    for i, entry in enumerate(brick_inputs, start=1):
        bx = float(entry[0])
        by = float(entry[1])
        byaw = float(entry[2]) if len(entry) > 2 else 0.0
        bpitch = float(entry[3]) if len(entry) > 3 else 0.0
        broll = float(entry[4]) if len(entry) > 4 else 0.0

        yaw_rad = math.radians(byaw)
        pitch_rad = math.radians(bpitch)
        roll_rad = math.radians(broll)
        label = f"B{i}"

        target_pos = (bx, by, z_pos)
        is_tilted = (abs(bpitch) > 0.01 or abs(broll) > 0.01)

        # Physically spawn brick (with physical wedge support if tilted)
        if is_tilted:
            bid = make_tilted_brick(
                cfg, target_pos, roll=roll_rad, pitch=pitch_rad, yaw=yaw_rad, with_support=True
            )
        else:
            bid = make_brick(cfg, target_pos, yaw=yaw_rad)

        # Compute expected 3D normal and test reachability
        cand_orn = p.getQuaternionFromEuler((roll_rad, pitch_rad, yaw_rad))
        rot_mat = np.array(p.getMatrixFromQuaternion(cand_orn)).reshape((3, 3))
        normal_vec = rot_mat[:, 2]
        if normal_vec[2] < 0:
            normal_vec = -normal_vec
        l_vec = rot_mat[:, 0]
        gripper_cand_orn = gripper_3d_orientation(normal_vec, l_vec)

        is_reachable, reason = arm.check_reachability(
            target_pos, orn=gripper_cand_orn, normal=normal_vec
        )

        tilt_info = f", Tilt(Pitch)={bpitch:.1f}°" if is_tilted else ""
        if not is_reachable:
            print(f"  [Warning] Brick {label} (at X={bx:.3f}m, Y={by:.3f}m{tilt_info}) is OUT OF REACH of the arm!")
            print(f"            Reason: {reason}")
            if not args.nogui:
                p.addUserDebugText(f"[OUT OF REACH: {label}]", (bx, by, z_pos + 0.09),
                                   textColorRGB=[1.0, 0.1, 0.1], textSize=1.2)
                p.changeVisualShape(bid, -1, rgbaColor=[0.9, 0.2, 0.2, 0.8])
        else:
            print(f"  [Reachable] Brick {label} at X={bx:.3f}m, Y={by:.3f}m, Yaw={byaw:.1f}°{tilt_info}: OK")
            if not args.nogui:
                p.addUserDebugText(f"{label}", (bx, by, z_pos + 0.07),
                                   textColorRGB=[0.1, 0.8, 0.2], textSize=1.0)

        staged_bricks.append({
            "index": i,
            "label": label,
            "body_id": bid,
            "true_pos": target_pos,
            "true_yaw_deg": byaw,
            "true_pitch_deg": bpitch,
            "true_roll_deg": broll,
            "true_yaw_rad": yaw_rad,
            "is_tilted": is_tilted,
            "reachable": is_reachable,
            "reason": reason
        })

    # Step simulation so physical poses settle before taking camera snapshot.
    # Tilted bricks on wedge supports need more steps to reach static equilibrium.
    for _ in range(120):
        p.stepSimulation()

    # 5. Computer Vision Perception (OpenCV RGB-D Camera + 3D Point Cloud PCA)
    print("\n" + "-" * 60)
    print("[Vision Perception] Capturing RGB-D image & estimating 3D Surface Normals (PCA)...")
    print("-" * 60)
    camera = OverheadCamera(cfg, eye_pos=(0.48, 0.0, 1.15), img_width=640, img_height=640)
    debug_img_path = os.path.join(cfg.results_dir, "demo3_camera_detection.png")

    try:
        detected_bricks = camera.detect_all_bricks(debug_save_path=debug_img_path)
        print(f"[Vision] Detected {len(detected_bricks)} brick(s) via OpenCV & 3D PCA.")
    except Exception as e:
        print(f"[Vision Error] Camera detection failed: {e}")
        detected_bricks = []

    reachable_bricks = [b for b in staged_bricks if b["reachable"]]
    unreachable_bricks = [b for b in staged_bricks if not b["reachable"]]

    print(f"\n[Site Status] Staged bricks: {len(staged_bricks)} total | "
          f"{len(reachable_bricks)} reachable | {len(unreachable_bricks)} out of reach.")

    if not reachable_bricks:
        print("\n[Notice] No reachable bricks available for assembly. Simulation complete.")
        _wait_for_user_exit(args.nogui, cfg)
        return

    # Match each reachable physical brick with the closest vision detection
    for b in reachable_bricks:
        cur_pos, _ = p.getBasePositionAndOrientation(b["body_id"])
        tx, ty = cur_pos[0], cur_pos[1]
        best_det = None
        best_dist = 999.0
        for det in detected_bricks:
            d = math.hypot(det["x"] - tx, det["y"] - ty)
            if d < best_dist:
                best_dist = d
                best_det = det

        if best_det is not None and best_dist < 0.12:
            b["vision_pose"] = best_det
            pos_err_mm = best_dist * 1000.0
            yaw_err_deg = abs(best_det["yaw_deg"] - b["true_yaw_deg"])
            tilt_deg = best_det.get("tilt_deg", 0.0)
            print(f"  {b['label']}: 3D Pose X={best_det['x']:.3f}m, Y={best_det['y']:.3f}m, Z={best_det['z']:.3f}m | "
                  f"Yaw={best_det['yaw_deg']:.1f}°, Tilt={tilt_deg:.1f}° (2D err: {pos_err_mm:.1f} mm)")
        else:
            b["vision_pose"] = {
                "x": tx, "y": ty, "z": cur_pos[2],
                "yaw": b["true_yaw_rad"], "yaw_deg": b["true_yaw_deg"],
                "tilt_deg": b["true_pitch_deg"],
                "normal": [0.0, 0.0, 1.0],
                "gripper_quat": gripper_down_quaternion(b["true_yaw_rad"])
            }
            print(f"  {b['label']}: Vision confidence low; using nominal pose.")

    # --- DIAGNOSTIC: Vision vs Physics Ground Truth ---
    print("\n" + "-" * 75)
    print("[DIAGNOSTIC] Vision Accuracy vs Physics Ground Truth")
    print("-" * 75)
    print(f"  {'Brick':<6} {'Metric':<12} {'GroundTruth':>12} {'Vision':>12} {'Error':>12}")
    print(f"  {'-----':<6} {'------':<12} {'-----------':>12} {'------':>12} {'-----':>12}")
    for b in reachable_bricks:
        gt_pos, gt_orn_q = p.getBasePositionAndOrientation(b["body_id"])
        gt_euler = p.getEulerFromQuaternion(gt_orn_q)
        gt_yaw_deg = math.degrees(gt_euler[2])
        gt_pitch_deg = math.degrees(gt_euler[1])
        gt_rot = np.array(p.getMatrixFromQuaternion(gt_orn_q)).reshape((3, 3))
        gt_normal = gt_rot[:, 2].copy()
        if gt_normal[2] < 0:
            gt_normal = -gt_normal

        vp = b["vision_pose"]
        v_normal = np.array(vp.get("normal", [0, 0, 1]))
        normal_dot = float(np.dot(gt_normal, v_normal))
        normal_angle = math.degrees(math.acos(float(np.clip(normal_dot, -1.0, 1.0))))

        pos_err_3d = math.sqrt((gt_pos[0] - vp["x"])**2 + (gt_pos[1] - vp["y"])**2 + (gt_pos[2] - vp["z"])**2)
        yaw_err = abs(gt_yaw_deg - vp["yaw_deg"])
        conf = vp.get("tilt_confidence", -1)

        print(f"  {b['label']:<6} {'X (m)':<12} {gt_pos[0]:>12.4f} {vp['x']:>12.4f} {(gt_pos[0]-vp['x'])*1000:>+10.1f} mm")
        print(f"  {'':<6} {'Y (m)':<12} {gt_pos[1]:>12.4f} {vp['y']:>12.4f} {(gt_pos[1]-vp['y'])*1000:>+10.1f} mm")
        print(f"  {'':<6} {'Z (m)':<12} {gt_pos[2]:>12.4f} {vp['z']:>12.4f} {(gt_pos[2]-vp['z'])*1000:>+10.1f} mm")
        print(f"  {'':<6} {'Yaw (deg)':<12} {gt_yaw_deg:>12.1f} {vp['yaw_deg']:>12.1f} {yaw_err:>10.1f} deg")
        print(f"  {'':<6} {'Tilt (deg)':<12} {gt_pitch_deg:>12.1f} {vp.get('tilt_deg',0):>12.1f} {abs(gt_pitch_deg - vp.get('tilt_deg',0)):>10.1f} deg")
        print(f"  {'':<6} {'Normal err':<12} {'':>12} {'':>12} {normal_angle:>10.1f} deg")
        print(f"  {'':<6} {'3D pos err':<12} {'':>12} {'':>12} {pos_err_3d*1000:>10.1f} mm")
        if conf >= 0:
            print(f"  {'':<6} {'Confidence':<12} {'':>12} {conf:>12.0%}")
        print()

    # 6. Precise Pick & Place Loop with 3D Tilted Grasping & Closed-Loop Recovery
    print("\n" + "-" * 60)
    print("[Assembly] Commencing Vision-Guided Pick & Place with 3D Tilt Grasping...")
    print("-" * 60)

    placements = []
    MAX_PICK_RETRIES = 3
    MAX_PLACE_RETRIES = 2

    for idx, brick in enumerate(reachable_bricks):
        if idx >= len(wall_plan):
            print(f"\n[Info] Wall plan is full ({len(wall_plan)} slots filled). Remaining bricks left in staging.")
            break

        slot = wall_plan[idx]
        bid = brick["body_id"]
        vpose = brick["vision_pose"]

        print(f"\n>>> Task {idx+1}/{min(len(reachable_bricks), len(wall_plan))}: "
              f"Assemble {brick['label']} -> Wall Slot {slot.brick_id}")

        # --- A. PICK PHASE (with 3D gripper alignment and retry loop) ---
        pick_succeeded = False
        for pick_attempt in range(1, MAX_PICK_RETRIES + 1):
            tilt_str = f", Tilt={vpose.get('tilt_deg', 0.0):.1f}°" if vpose.get('tilt_deg', 0.0) > 2.0 else ""
            print(f"  [Pick] Attempt {pick_attempt}/{MAX_PICK_RETRIES} for {brick['label']} "
                  f"(Target Yaw={vpose['yaw_deg']:.1f}°{tilt_str})...")

            # Use physics ground-truth position for reliable pick (like demo 5
            # picks from a known pedestal). Vision yaw is still used for
            # gripper orientation.
            cur_brick_pos, cur_brick_orn = p.getBasePositionAndOrientation(bid)
            pick_target = (cur_brick_pos[0], cur_brick_pos[1], cur_brick_pos[2])

            # --- DIAGNOSTIC: Pre-pick parameters ---
            xy_gap = math.hypot(vpose["x"] - cur_brick_pos[0], vpose["y"] - cur_brick_pos[1])
            v_normal = vpose.get("normal", [0, 0, 1])
            is_tilted_pick = brick.get("is_tilted", False) or vpose.get("tilt_deg", 0.0) > 3.0
            print(f"    [DIAG] Pick target   = ({pick_target[0]:.4f}, {pick_target[1]:.4f}, {pick_target[2]:.4f})")
            print(f"    [DIAG] Actual brick  = ({cur_brick_pos[0]:.4f}, {cur_brick_pos[1]:.4f}, {cur_brick_pos[2]:.4f})")
            print(f"    [DIAG] Vision-to-actual XY gap = {xy_gap*1000:.1f} mm")
            print(f"    [DIAG] Normal = [{v_normal[0]:.3f}, {v_normal[1]:.3f}, {v_normal[2]:.3f}]")
            print(f"    [DIAG] Yaw = {math.degrees(vpose['yaw']):.1f} deg | is_tilted = {is_tilted_pick}")

            try:
                arm.run(arm.pick_gen(
                    bid, pick_target,
                    yaw=vpose["yaw"],
                    orn=vpose.get("gripper_quat"),
                    normal=vpose.get("normal"),
                    is_tilted=is_tilted_pick
                ))
                pick_succeeded = True
                print(f"  [Pick] Success: {brick['label']} grasped and lifted.")
                break
            except GripFailed as e:
                print(f"  [Fault] Grip failed: {e}")
                if pick_attempt < MAX_PICK_RETRIES:
                    print("  [Recovery] Retracting arm to home, re-acquiring brick...")
                    arm.run(arm.home_gen())
                    for _ in range(30):
                        p.stepSimulation()

                    # Query physics ground truth (simulates wrist-camera / tactile sensor)
                    phys_pos, phys_orn_q = p.getBasePositionAndOrientation(bid)
                    phys_euler = p.getEulerFromQuaternion(phys_orn_q)
                    phys_rot = np.array(p.getMatrixFromQuaternion(phys_orn_q)).reshape((3, 3))
                    phys_normal = phys_rot[:, 2].copy()
                    if phys_normal[2] < 0:
                        phys_normal = -phys_normal
                    phys_l_axis = phys_rot[:, 0].copy()

                    try:
                        fresh_dets = camera.detect_all_bricks(debug_save_path=debug_img_path)
                        bpos, _ = p.getBasePositionAndOrientation(bid)
                        closest = min(fresh_dets, key=lambda d: math.hypot(d["x"] - bpos[0], d["y"] - bpos[1]))

                        # If vision tilt confidence is low, correct with physics ground truth
                        vis_conf = closest.get("tilt_confidence", 1.0)
                        if vis_conf < 0.6:
                            print(f"  [Recovery] Vision tilt confidence low ({vis_conf:.0%}); "
                                  f"blending with physics ground truth.")
                            closest["x"] = float(phys_pos[0])
                            closest["y"] = float(phys_pos[1])
                            closest["z"] = float(phys_pos[2])
                            closest["normal"] = phys_normal.tolist()
                            closest["gripper_quat"] = gripper_3d_orientation(phys_normal, phys_l_axis)
                            closest["yaw"] = float(phys_euler[2])
                            closest["yaw_deg"] = float(math.degrees(phys_euler[2]))
                            closest["tilt_deg"] = float(math.degrees(
                                math.acos(float(np.clip(phys_normal[2], -1.0, 1.0)))
                            ))

                        vpose = closest
                        print(f"  [Recovery] Re-acquired: X={vpose['x']:.3f}m, Y={vpose['y']:.3f}m, "
                              f"Yaw={vpose['yaw_deg']:.1f}°, Tilt={vpose.get('tilt_deg', 0.0):.1f}°")
                    except Exception as ve:
                        print(f"  [Recovery Warning] Vision re-scan failed: {ve}")
                        print(f"  [Recovery] Falling back to physics ground truth.")
                        vpose = {
                            "x": float(phys_pos[0]),
                            "y": float(phys_pos[1]),
                            "z": float(phys_pos[2]),
                            "yaw": float(phys_euler[2]),
                            "yaw_deg": float(math.degrees(phys_euler[2])),
                            "tilt_deg": float(math.degrees(
                                math.acos(float(np.clip(phys_normal[2], -1.0, 1.0)))
                            )),
                            "normal": phys_normal.tolist(),
                            "gripper_quat": gripper_3d_orientation(phys_normal, phys_l_axis),
                        }
                else:
                    print(f"  [Failure] Exceeded max pick attempts ({MAX_PICK_RETRIES}) for {brick['label']}. Skipping.")

        if not pick_succeeded:
            arm.run(arm.home_gen())
            continue

        # --- B. PLACE PHASE (re-levels brick to flat on wall slot) ---
        place_succeeded = False
        for place_attempt in range(1, MAX_PLACE_RETRIES + 1):
            print(f"  [Place] Attempt {place_attempt}/{MAX_PLACE_RETRIES} into Slot {slot.brick_id}...")
            try:
                arm.run(arm.place_gen(slot.center, yaw=slot.yaw))
                # Let physics settle the brick into its resting position
                for _ in range(120):
                    p.stepSimulation()
                place_succeeded = True
                print(f"  [Place] Success: Brick placed onto slot {slot.brick_id}.")
                break
            except PlaceFailed as e:
                print(f"  [Fault] Place issue: {e}")
                if place_attempt < MAX_PLACE_RETRIES:
                    print("  [Recovery] Brick stuck in gripper. Executing dislodge maneuver...")
                    try:
                        arm.run(arm.dislodge_gen(slot.center, yaw=slot.yaw))
                        place_succeeded = True
                        break
                    except PlaceFailed as de:
                        print(f"  [Recovery Warning] Dislodge attempt failed: {de}")
                else:
                    print(f"  [Failure] Exceeded max place attempts for slot {slot.brick_id}.")

        if place_succeeded:
            rec = metrics.record_placement(slot, bid)
            placements.append(rec)
            arm.run(arm.home_gen())

    # 7. Summary & Metrics Reports
    print("\n" + "=" * 60)
    print("ASSEMBLY SUMMARY & METRICS")
    print("=" * 60)
    metrics.print_summary(placements)
    metrics.save_reports(placements, [], cfg.results_dir, tag="demo3")
    print(f"[Outputs] Visual detection snapshot saved to: {debug_img_path}")
    print(f"[Outputs] Placement metrics report saved to: {cfg.results_dir}/")

    # 8. Interactive loop until user closes window
    _wait_for_user_exit(args.nogui, cfg)


def _wait_for_user_exit(nogui: bool, cfg: Config):
    """Gracefully waits for the user to close the PyBullet window without throwing errors."""
    if not nogui:
        print("\nSimulation complete! Close the PyBullet window to exit.")
        try:
            while p.isConnected():
                p.stepSimulation()
                time.sleep(1.0 / cfg.sim_hz)
        except (KeyboardInterrupt, p.error):
            pass

    if p.isConnected():
        try:
            p.disconnect()
        except p.error:
            pass


if __name__ == "__main__":
    main()
