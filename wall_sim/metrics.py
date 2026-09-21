"""Placement-error records and task-timeline reports (CSV + optional Gantt)."""
import csv
import math
import os

import pybullet as p

from .wall_plan import BrickSpec


def record_placement(spec: BrickSpec, body_id: int):
    """Compare a placed brick's actual pose against its target spec."""
    pos, orn_q = p.getBasePositionAndOrientation(body_id)
    yaw = p.getEulerFromQuaternion(orn_q)[2]
    pos_err_mm = 1000.0 * math.dist(pos, spec.center)
    yaw_err = math.degrees(yaw - spec.yaw)
    # Rectangular bricks are symmetric under 180° rotation
    yaw_err = (yaw_err + 90.0) % 180.0 - 90.0
    rec = {
        "brick": spec.brick_id, "row": spec.row, "col": spec.col,
        "half": int(spec.half),
        "pos_err_mm": round(pos_err_mm, 2), "yaw_err_deg": round(yaw_err, 2),
    }
    print(f"[place] {spec.brick_id:<6} pos error {pos_err_mm:5.2f} mm, "
          f"yaw error {yaw_err:5.2f} deg")
    return rec


def print_summary(placements):
    if not placements:
        print("no placements recorded")
        return
    errs = [r["pos_err_mm"] for r in placements]
    print("\n=== placement summary ===")
    print(f"bricks placed : {len(errs)}")
    print(f"mean |pos err|: {sum(errs)/len(errs):.2f} mm")
    print(f"max  |pos err|: {max(errs):.2f} mm")


def save_reports(placements, timeline, results_dir="results", tag="run"):
    os.makedirs(results_dir, exist_ok=True)

    ppath = os.path.join(results_dir, f"placement_{tag}.csv")
    if placements:
        with open(ppath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(placements[0].keys()))
            w.writeheader()
            w.writerows(placements)

    tpath = os.path.join(results_dir, f"timeline_{tag}.csv")
    if timeline:
        with open(tpath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(timeline[0].keys()))
            w.writeheader()
            w.writerows(timeline)

    _save_gantt(timeline, os.path.join(results_dir, f"gantt_{tag}.png"))
    print(f"reports written to {results_dir}/ ({tag})")


def _save_gantt(timeline, png_path):
    if not timeline:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping gantt png")
        return

    colors = {"shuttle": "#4c72b0", "mortar": "#8a9a5b", "arm": "#c44e52"}
    rows = sorted(timeline, key=lambda t: (t["start_s"], t["task"]))
    fig, ax = plt.subplots(figsize=(9, max(2, 0.35 * len(rows))))
    for i, t in enumerate(rows):
        start, dur = t["start_s"], max(t["end_s"] - t["start_s"], 0.02)
        ax.barh(i, dur, left=start, height=0.6,
                color=colors.get(t["robot"], "gray"))
        ax.text(start + dur / 2, i, t["kind"], ha="center", va="center",
                fontsize=7, color="white")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([t["task"] for t in rows], fontsize=7)
    ax.set_xlabel("time (s)")
    ax.set_title("robot task timeline")
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
