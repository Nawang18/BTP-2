"""Demo 2 -- precise wall building by the arm alone (the project's core demo).

The Panda builds the whole running-bond wall from a brick stack: for every
brick, pick from the stack, place at the exact planned pose. A static mortar
slab is spawned under each slot (visual only). Produces the placement-error
report + task timeline. The transport/mortar robots are absent here -- this
demo isolates the placement-precision claim.

Run:  python run_demo2_wall.py [--rows 2] [--bricks 3] [--nogui]
"""
import argparse

# pyrefly: ignore [missing-import]
import pybullet as p

from wall_sim.config import Config
from wall_sim.scene import (build_world, make_static_box, draw_slot_outline,
                            Pallet)
from wall_sim.wall_plan import plan_wall, mortar_center_for
from wall_sim.robots import PandaArm
from wall_sim import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--bricks", type=int, default=3, help="full bricks per row")
    ap.add_argument("--nogui", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    cfg.wall_rows = args.rows
    cfg.wall_bricks_per_row = args.bricks
    build_world(cfg, gui=not args.nogui)

    plan = plan_wall(cfg)
    for spec in plan:
        draw_slot_outline(spec, cfg)

    pallet = Pallet(cfg, plan)
    arm = PandaArm(cfg)
    arm.run(arm.home_gen())

    placements, timeline = [], []
    steps = 0

    def run(gen):
        nonlocal steps
        for _ in gen:
            p.stepSimulation()
            steps += 1

    print(f"demo2: building wall {cfg.wall_bricks_per_row}x{cfg.wall_rows} "
          f"({len(plan)} bricks)")
    for spec in plan:
        body, pop_spec, center = pallet.pop()
        assert pop_spec.brick_id == spec.brick_id  # stack order == plan order
        start_s = steps / cfg.sim_hz
        make_static_box(mortar_center_for(spec, cfg),
                        (spec.length + cfg.brick.gap, cfg.brick.width,
                         cfg.brick.gap), cfg.mortar_rgba)
        run(arm.pick_gen(body, center))
        run(arm.place_gen(spec.center, yaw=spec.yaw))
        placements.append(metrics.record_placement(spec, body))
        timeline.append({
            "task": f"place_{spec.brick_id}", "kind": "PLACE", "robot": "arm",
            "brick": spec.brick_id, "row": spec.row, "col": spec.col,
            "start_s": round(start_s, 3),
            "end_s": round(steps / cfg.sim_hz, 3), "status": "done",
        })

    run(arm.home_gen())
    metrics.print_summary(placements)
    metrics.save_reports(placements, timeline, cfg.results_dir, tag="demo2")
    p.disconnect()


if __name__ == "__main__":
    main()
