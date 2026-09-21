"""Demo 1 -- single scripted pick & place (month-1 / month-2 core skill).

Arm picks one brick off the handover pedestal and places it into the first
wall slot, then reports placement error. Run:  python run_demo1_pick_place.py
"""
import argparse

# pyrefly: ignore [missing-import]
import pybullet as p

from wall_sim.config import Config
from wall_sim.scene import (build_world, make_brick, make_static_box,
                            make_pedestal, draw_slot_outline)
from wall_sim.wall_plan import plan_wall, mortar_center_for
from wall_sim.robots import PandaArm
from wall_sim import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nogui", action="store_true", help="headless (DIRECT) mode")
    args = ap.parse_args()

    cfg = Config()
    build_world(cfg, gui=not args.nogui)

    plan = plan_wall(cfg)
    slot = plan[0]
    draw_slot_outline(slot, cfg)
    make_pedestal(cfg)
    make_static_box(mortar_center_for(slot, cfg),
                    (slot.length + cfg.brick.gap, cfg.brick.width,
                     cfg.brick.gap), cfg.mortar_rgba)
    brick = make_brick(cfg, cfg.pedestal_brick_center())

    arm = PandaArm(cfg)
    arm.run(arm.home_gen())

    print("demo1: picking brick from pedestal and placing into slot "
          f"{slot.brick_id}")
    arm.run(arm.pick_gen(brick, cfg.pedestal_brick_center()))
    arm.run(arm.place_gen(slot.center, yaw=slot.yaw))

    rec = metrics.record_placement(slot, brick)
    metrics.print_summary([rec])
    metrics.save_reports([rec], [], cfg.results_dir, tag="demo1")
    import time

    if not args.nogui:
        print("\nSimulation complete! The window will stay open. Close the PyBullet window to exit.")
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
