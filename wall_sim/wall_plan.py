"""Turns a wall spec into an ordered list of exact target brick poses.

Running bond: even rows are N full bricks; odd rows are
half | (N-1) x full | half, offset by half a brick, so both row types span
the same total length. Every pose computed here is the ground truth the arm
is measured against in the placement-error report.
"""
from dataclasses import dataclass

from .config import Config


@dataclass
class BrickSpec:
    brick_id: str        # e.g. "r1c2"
    row: int
    col: int             # position within the row (left to right)
    half: bool
    length: float        # brick.length or brick.length/2
    center: tuple        # (x, y, z) of brick centre in world
    yaw: float           # rotation about world z (bricks run along +x)


def plan_wall(cfg: Config):
    b = cfg.brick
    pitch = b.length + b.gap
    bricks = []

    for r in range(cfg.wall_rows):
        z = b.gap + b.height / 2.0 + r * (b.height + b.gap)
        if r % 2 == 0:
            for c in range(cfg.wall_bricks_per_row):
                x = cfg.wall_origin[0] + c * pitch + b.length / 2.0
                bricks.append(BrickSpec(f"r{r}c{c}", r, c, False, b.length,
                                        (x, cfg.wall_origin[1], z), 0.0))
        else:
            seq = [True] + [False] * (cfg.wall_bricks_per_row - 1) + [True]
            x = cfg.wall_origin[0]
            for c, half in enumerate(seq):
                ln = b.length / 2.0 if half else b.length
                bricks.append(BrickSpec(f"r{r}c{c}", r, c, half, ln,
                                        (x + ln / 2.0, cfg.wall_origin[1], z), 0.0))
                x += ln + b.gap
    return bricks


def mortar_center_for(spec, cfg: Config):
    """Centre of the mortar bed directly under a brick target (top of the
    previous course / ground for row 0). Slab thickness == joint gap."""
    b = cfg.brick
    z = spec.center[2] - b.height / 2.0 - b.gap / 2.0
    return (spec.center[0], spec.center[1], z)
