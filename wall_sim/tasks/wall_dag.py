"""Compiles a wall plan into a task DAG for the three-robot team.

Each brick gets four tasks: TRANSPORT (shuttle), MORTAR (mortar cart),
PICK and PLACE (arm).

Dependency rules (masonry-ordered), with "prev" = previous brick in plan order:
  transport(b) : waits for pick(prev). The handover pedestal holds one brick,
                 and it is free again the moment the arm has lifted prev off
                 it -- so the shuttle fetches the next brick WHILE the arm is
                 still placing the current one. This edge is the pedestal
                 mutex, and it also keeps deliveries in pallet (plan) order.
  mortar(b)    : waits for every PLACE of the row below (course must exist)
  pick(b)      : waits for transport(b) and place(prev) (the gripper holds
                 one brick at a time)
  place(b)     : waits for pick(b) + mortar(b)

`parallel=False` gives the sequential baseline used for comparison: there
transport(b) waits for place(prev), so the shuttle and the arm take turns.
"""


def build_wall_dag(plan, parallel=True):
    from .task_graph import TaskGraph

    g = TaskGraph()
    by_row = {}
    for spec in plan:
        by_row.setdefault(spec.row, []).append(spec)

    prev = None                  # previous brick in plan order
    prev_row_places = []
    for row, specs in by_row.items():
        for spec in specs:
            bid = spec.brick_id

            t_deps = []
            if prev is not None:
                t_deps.append(f"pick_{prev.brick_id}" if parallel
                              else f"place_{prev.brick_id}")
            g.add(f"transport_{bid}", "TRANSPORT", "shuttle", spec, t_deps)

            m_deps = [f"place_{p.brick_id}" for p in prev_row_places]
            g.add(f"mortar_{bid}", "MORTAR", "mortar", spec, m_deps)

            k_deps = [f"transport_{bid}"]
            if prev is not None:
                k_deps.append(f"place_{prev.brick_id}")
            g.add(f"pick_{bid}", "PICK", "arm", spec, k_deps)

            g.add(f"place_{bid}", "PLACE", "arm", spec,
                  [f"pick_{bid}", f"mortar_{bid}"])
            prev = spec
        prev_row_places = specs
    return g
