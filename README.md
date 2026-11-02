# Multi-Robot Task Orchestration for Construction Assembly (BTP)

PyBullet simulation of a three-robot team that builds a brick wall from a
single natural-language command:

```
"build a wall, 3 bricks per row, 2 rows"
        |
        v
   [ brain / planner ]          mock LLM front-end (regex today, LLM in month 4)
        |  wall spec
        v
   [ wall planner ]             computes exact target pose for every brick
        |  (x, y, z, yaw) list  running bond, 5 mm mortar joints
        v
   [ task DAG builder ]         transport / mortar / place tasks + dependencies
        |
        v
   [ executor ]  <-- deterministic scheduler; dispatches tasks whose
        |          dependencies are done, advances all robots concurrently
        v
  shuttle (transport)   mortar bot (dispenser)   Franka Panda (assembler)
```

## Quick start

```bash
pip install -r requirements.txt

python run_demo1_pick_place.py            # GUI: single pick & place
python run_demo1_pick_place.py --nogui    # headless

python run_demo2_wall.py                  # arm alone builds the whole wall
python run_demo2_wall.py --rows 3 --bricks 4

python run_demo3_multi_robot.py           # vision pick-and-place wall assembly (random staging)
python run_demo3_multi_robot.py --coords "0.46,-0.10,15; 0.50,0.10,-10"  # custom flat coordinates
python run_demo3_multi_robot.py --coords "0.48,-0.10,15,12; 0.50,0.10,-10,0"  # with 3D pitch tilt (12 deg)
python run_demo3_multi_robot.py --coords "0.48,-0.10,0; 1.25,0.50,0"     # with out-of-reach warning
python run_demo3_multi_robot.py --demo_tilted                            # demonstrate tilted brick 3D grasping
python run_demo3_multi_robot.py --interactive                           # interactive terminal input
```

Close the PyBullet window when the script prints its final summary
(headless runs exit on their own).

## Demos

| demo | robots | shows |
|------|--------|-------|
| 1 | arm | single pick & place (scripted or vision-guided) |
| 2 | arm | whole running-bond wall from brick stack |
| 3 | arm + vision | multi-brick vision pick & place, reachability checks, custom coords, closed-loop recovery |

Expected placement accuracy (headless, measured by `metrics.py`):
mean ~1-2 mm, max < 2 mm, yaw < 2 deg. Reports land in `results/`:
`placement_*.csv`, `timeline_*.csv`, `gantt_*.png` (task timeline chart).

## Layout

```
wall_sim/
  config.py        every physical/layout constant (single source of truth)
  wall_plan.py     wall spec -> exact brick target poses (ground truth)
  scene.py         pybullet world, bricks, pallet stack, slot outlines
  robots/
    panda_arm.py   IK, gripper, pick/place generators; grasp-error compensation
    shuttle.py     kinematic transport cart with collision-safe lane routing
    mortar_bot.py  dispenser head; spawns a visual mortar slab per joint
    workers.py     adapt task-graph tasks -> robot action generators
  tasks/
    task_graph.py  DAG data structure
    wall_dag.py    masonry dependency rules (transport/mortar/place)
    executor.py    concurrent scheduler; steps physics once per tick
  brain/
    planner.py     command -> PlanSpec (mock parser + month-4 LLM prompt)
  metrics.py       placement errors, CSV reports, Gantt chart
```

