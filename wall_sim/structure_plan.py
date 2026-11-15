"""Free-form brick structures: a row-by-row layout the LLM writes, a geometry
checker that decides whether it can actually be built, and the conversion to
exact BrickSpec target poses.

Layout units: every row lists its bricks left to right ("full" or "half") and
an `offset` in half-brick units measured from the wall origin. One unit is
half a brick pitch (~102 mm), so offset 1 shifts a row by half a brick --
exactly what a running bond or a pedestal needs.
"""
import math
from dataclasses import dataclass, field

from .config import Config
from .wall_plan import BrickSpec

MAX_ROWS = 5
MAX_BRICKS = 15
ARM_REACH = 0.85            # max horizontal distance of a brick centre from the arm base
MIN_SUPPORT = 0.5           # fraction of a brick's length that must rest on the row below


@dataclass
class RowLayout:
    bricks: list                 # ["full", "half", ...] left to right
    offset: int = 0              # half-brick units from the wall origin


@dataclass
class Layout:
    rows: list = field(default_factory=list)   # bottom row first
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Layout":
        rows = []
        for i, r in enumerate(data.get("rows", [])):
            bricks = r.get("bricks", [])
            if isinstance(bricks, int):
                bricks = ["full"] * bricks
            bricks = [str(b).strip().lower() for b in bricks]
            try:
                offset = int(r.get("offset", 0))
            except (TypeError, ValueError):
                raise ValueError(f"row {i}: offset must be an integer")
            rows.append(RowLayout(bricks, offset))
        return cls(rows, str(data.get("description", "")))

    def to_dict(self) -> dict:
        return {"description": self.description,
                "rows": [{"bricks": r.bricks, "offset": r.offset} for r in self.rows]}

    def brick_count(self) -> int:
        return sum(len(r.bricks) for r in self.rows)


def rectangular_layout(bricks_per_row: int, rows: int) -> Layout:
    """Running bond, identical to the original plan_wall()."""
    out = []
    for r in range(rows):
        if r % 2 == 0:
            out.append(RowLayout(["full"] * bricks_per_row, 0))
        else:
            out.append(RowLayout(["half"] + ["full"] * (bricks_per_row - 1) + ["half"], 0))
    return Layout(out, f"{bricks_per_row}-wide running bond wall, {rows} rows")


def centered_layout(counts) -> Layout:
    """Rows of full bricks, each centred over the row below (pedestal/pyramid)."""
    offsets = [0]
    for prev, cur in zip(counts, counts[1:]):
        offsets.append(offsets[-1] + (prev - cur))
    shift = -min(offsets)
    rows = [RowLayout(["full"] * n, o + shift) for n, o in zip(counts, offsets)]
    return Layout(rows, "rows of " + "-".join(str(n) for n in counts)
                  + " full bricks, each centred on the row below")


def layout_to_specs(layout: Layout, cfg: Config):
    b = cfg.brick
    half_unit = (b.length + b.gap) / 2.0
    specs = []
    for r, row in enumerate(layout.rows):
        z = b.gap + b.height / 2.0 + r * (b.height + b.gap)
        x = cfg.wall_origin[0] + row.offset * half_unit
        for c, kind in enumerate(row.bricks):
            half = kind == "half"
            ln = b.length / 2.0 if half else b.length
            specs.append(BrickSpec(f"r{r}c{c}", r, c, half, ln,
                                   (x + ln / 2.0, cfg.wall_origin[1], z), 0.0))
            x += ln + b.gap
    return specs


def _merged_spans(specs, gap):
    spans = sorted((s.center[0] - s.length / 2.0, s.center[0] + s.length / 2.0)
                   for s in specs)
    merged = []
    for lo, hi in spans:
        if merged and lo - merged[-1][1] <= gap * 1.5:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return merged


def validate_layout(layout: Layout, cfg: Config):
    """Return a list of human-readable problems; empty list means buildable."""
    errors = []
    if not layout.rows:
        return ["layout has no rows"]
    if len(layout.rows) > MAX_ROWS:
        errors.append(f"{len(layout.rows)} rows requested, maximum is {MAX_ROWS}")
    if layout.brick_count() > MAX_BRICKS:
        errors.append(f"{layout.brick_count()} bricks requested, maximum is {MAX_BRICKS}")
    for r, row in enumerate(layout.rows):
        if not row.bricks:
            errors.append(f"row {r} is empty")
        bad = [k for k in row.bricks if k not in ("full", "half")]
        if bad:
            errors.append(f"row {r} has unknown brick types {bad} (use 'full' or 'half')")
        if row.offset < 0:
            errors.append(f"row {r} has negative offset {row.offset}")
    if errors:
        return errors

    specs = layout_to_specs(layout, cfg)
    by_row = {}
    for s in specs:
        by_row.setdefault(s.row, []).append(s)

    for s in specs:
        reach = math.hypot(s.center[0], s.center[1])
        if reach > ARM_REACH:
            errors.append(f"brick {s.brick_id} is {reach:.2f} m from the arm base "
                          f"(limit {ARM_REACH} m) -- row too long or offset too large")

    for r in range(1, len(layout.rows)):
        for s in by_row[r]:
            ok, frac = _support(s, by_row[r - 1], cfg.brick.gap)
            if not ok:
                errors.append(f"brick {s.brick_id} is only {frac:.0%} "
                              f"supported by row {r - 1} -- it would fall off")
    return errors


def _support(spec, below_specs, gap):
    """(is_supported, supported_fraction) of a brick resting on `below_specs`."""
    below = _merged_spans(below_specs, gap)
    lo, hi = spec.center[0] - spec.length / 2.0, spec.center[0] + spec.length / 2.0
    overlap = sum(max(0.0, min(hi, b) - max(lo, a)) for a, b in below)
    centred = any(a <= spec.center[0] <= b for a, b in below)
    return (overlap >= MIN_SUPPORT * spec.length and centred), overlap / spec.length


def unsupported_bricks(specs, missing_ids, gap):
    """Brick ids that can no longer be built once `missing_ids` are absent.

    Walks the rows bottom-up: a brick is lost if the bricks that will really
    exist in the row below no longer carry it. Losses propagate upward, so one
    missing brick can take out a whole column above it. The result does not
    include `missing_ids` themselves.
    """
    by_row = {}
    for s in specs:
        by_row.setdefault(s.row, []).append(s)
    gone = set(missing_ids)
    lost = set()
    for r in sorted(by_row)[1:]:
        below = [s for s in by_row.get(r - 1, []) if s.brick_id not in gone]
        for s in by_row[r]:
            if s.brick_id in gone:
                continue
            if not below or not _support(s, below, gap)[0]:
                gone.add(s.brick_id)
                lost.add(s.brick_id)
    return lost


def render_ascii(layout: Layout) -> str:
    """Side view of the layout, top row first. One half-brick unit = 4 chars."""
    lines = []
    for r in reversed(range(len(layout.rows))):
        row = layout.rows[r]
        body = "".join("[######]" if k == "full" else "[##]" for k in row.bricks)
        lines.append(f"  row {r}: " + " " * (4 * row.offset) + body)
    return "\n".join(lines)
