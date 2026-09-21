"""Compiles a wall plan into a task DAG for the three-robot team.

Dependency rules (masonry-ordered):
  transport(b)  : waits for the PLACE of the previous brick in plan order
                  (last brick of the previous row when b starts a new row).
                  This serialises the shuttle AND keeps deliveries aligned
                  with the pallet stack, which is consumed top-first in
                  exactly plan order.
  mortar(b)     : waits for every PLACE of the row below (course must exist)
  place(b)      : waits for transport(b) + mortar(b)
"""


def build_wall_dag(plan):
    from .task_graph import TaskGraph

    g = TaskGraph()
    prev_row_places = []
    by_row = {}
    for spec in plan:
        by_row.setdefault(spec.row, []).append(spec)

    for row, specs in by_row.items():
        for i, spec in enumerate(specs):
            bid = spec.brick_id
            t_deps = []
            if i > 0:
                t_deps.append(f"place_{by_row[row][i - 1].brick_id}")
            elif row > 0:
                t_deps.append(f"place_{by_row[row - 1][-1].brick_id}")
            g.add(f"transport_{bid}", "TRANSPORT", "shuttle", spec, t_deps)

            m_deps = [f"place_{p.brick_id}" for p in prev_row_places]
            g.add(f"mortar_{bid}", "MORTAR", "mortar", spec, m_deps)

            g.add(f"place_{bid}", "PLACE", "arm", spec,
                  [f"transport_{bid}", f"mortar_{bid}"])
        prev_row_places = specs
    return g
