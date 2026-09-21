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

## Design decisions worth defending in a viva

1. **Generators as robot actions.** Each robot behaviour is a Python
   generator that yields once per simulation step. The executor advances
   all robots one step each tick, so robots run concurrently in one
   simulation without threads.
2. **LLM only at plan time.** The LLM (currently a regex mock behind the
   same interface) converts the command into a wall spec ONCE. Everything
   downstream is deterministic, so a bad LLM answer can never corrupt the
   control loop.
3. **Wall planner as the precision contract.** Every brick has an exact
   planned pose; the executor measures the placed brick against it and
   writes the error to CSV. This is the project's core measurable claim.
4. **Grasp-error compensation.** A friction grip leaves the brick a few mm
   off where the fingers centre it. After grasping, the arm reads the
   brick's actual pose (stand-in for a wrist camera) and subtracts the
   offset when commanding the place -- this took placement from ~10 mm to
   ~1-2 mm.
5. **Kinematic transport & mortar robots.** The project's claims are
   orchestration + placement precision, not locomotion or fluid dynamics,
   so the shuttle is a kinematic cart (brick rides rigidly on its deck)
   and mortar is a spawned slab. Both physically interact with the scene
   only through carefully routed paths (see the routing rules in their
   docstrings -- they exist because solid bodies knocked bricks around).
6. **Dependency rules encode masonry logic.** A brick's transport waits
   for the previous place; its mortar waits for the row below; its place
   waits for its transport + mortar. The executor needs no robot-specific
   logic -- the DAG is the coordination.

## Known limitations (say these before the examiners find them)

- Shuttle "picks up" and "drops" bricks by teleporting between dock and
  deck; only the driving is animated.
- Mortar is a static slab, not a fluid or a shrinking extrusion.
- Placement precision assumes perfect perception (the brick pose is read
  from the simulation state).
- The arm works in a fixed workspace tuned in `config.py`; larger walls
  need a different base placement.
