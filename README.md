# Multi-Robot Task Orchestration for Construction Assembly

This is my B.Tech final year project (BTP-2). The idea is simple — can we get
multiple robots to build a brick wall together, coordinated by an LLM that
understands natural language?

The whole thing runs in PyBullet (physics simulation). You type something like
"build a 3 by 2 wall" and the system figures out the rest: plans the wall,
generates tasks for each robot, coordinates them so they don't crash into each
other, and handles things when they go wrong.


## What's in here

Three robots working together:

- **Panda arm** — the main assembler. Picks bricks from the handover pedestal,
  places them on the wall with ~1-2mm accuracy. Uses inverse kinematics,
  computer vision (OpenCV + depth camera), and closed-loop error compensation.

- **Shuttle cart** — a wheeled platform on the ground that ferries bricks from
  the supply pallet to the handover pedestal. Follows a lane-based path to
  avoid knocking over the brick stack.

- **Mortar cart** — another wheeled platform that drives along the ground
  behind the wall. Has a nozzle arm that extends to apply mortar between
  brick courses, then retracts before the Panda arm comes in to place.

An **LLM brain** (Google Gemini) sits on top:

- **Planner**: takes your natural language command and turns it into a wall
  specification (how many bricks wide, how many rows, bond pattern).
- **Supervisor**: monitors the build, handles failures (retry or skip a
  brick), reports progress at milestones. Calls the LLM only when there's
  an actual decision to make (not every frame), so it stays within free-tier
  API limits.

The robots are coordinated through a **task DAG** (directed acyclic graph) —
each brick needs transport, then mortar, then placement, with dependencies
between them. The executor runs all three robots concurrently using Python
generators (one yield per simulation step, no threads).


## Quick start

```bash
pip install -r requirements.txt

# Set your Gemini API key (free tier works fine)
# On Windows:
set GEMINI_API_KEY=your_key_here
# On Linux/Mac:
export GEMINI_API_KEY=your_key_here
```

### Demo 4 — Full LLM-orchestrated multi-robot build (the main one)

```bash
python run_demo4_llm_orchestrated.py "build a 3 by 2 wall"
python run_demo4_llm_orchestrated.py --interactive
python run_demo4_llm_orchestrated.py --no-llm "4x3 wall"    # skip LLM, use regex parser
python run_demo4_llm_orchestrated.py --nogui "3x3"           # headless
python run_demo4_llm_orchestrated.py --rows 2 --bricks 4     # override dimensions
```

### Earlier demos (still work)

```bash
python run_demo1_pick_place.py              # single pick & place
python run_demo2_wall.py                    # arm alone builds a wall
python run_demo2_wall.py --rows 3 --bricks 4

python run_demo3_multi_robot.py             # vision-guided pick & place
python run_demo3_multi_robot.py --coords "0.46,-0.10,15; 0.50,0.10,-10"
python run_demo3_multi_robot.py --demo_tilted
python run_demo3_multi_robot.py --interactive
```

Close the PyBullet window when you're done (headless runs exit on their own).


## How it works

```
User: "build a wall, 3 bricks wide, 2 rows"
        |
        v
  [ LLM Planner ]        Gemini parses NL → wall spec JSON
        |
        v
  [ Wall Planner ]        computes exact (x,y,z,yaw) for every brick
        |
        v
  [ DAG Builder ]         transport → mortar → place tasks + dependencies
        |
        v
  [ Executor ]            deterministic tick scheduler, steps physics
        |
   +---------+---------+
   |         |         |
 Shuttle   Mortar    Panda
  cart      cart      arm

  [ LLM Supervisor ]     watches for failures, decides retry/skip,
                          reports progress (called only at decision points)
```


## Demos

| demo | robots | what it shows |
|------|--------|---------------|
| 1 | arm | single pick & place |
| 2 | arm | full wall from brick stack |
| 3 | arm + camera | vision-guided multi-brick pick & place, 3D tilt grasping |
| 4 | arm + shuttle + mortar + LLM | full orchestrated multi-robot wall build |

Placement accuracy: mean ~1-2mm, max <2mm.
Reports go into `results/`: placement CSVs, timeline CSVs, Gantt charts.


## Project structure

```
run_demo1_pick_place.py
run_demo2_wall.py
run_demo3_multi_robot.py
run_demo4_llm_orchestrated.py     ← the main demo

wall_sim/
  config.py          all physical constants (single source of truth)
  wall_plan.py       wall spec → exact brick target poses
  scene.py           PyBullet world setup, bricks, pallet, slot outlines
  camera_vision.py   OpenCV RGB-D brick detection + 3D pose estimation
  perception.py      unified perception interface (ground-truth / vision backends)
  metrics.py         placement errors, CSV reports, Gantt chart

  robots/
    panda_arm.py     IK, gripper, pick/place generators, grasp-error compensation
    shuttle.py       wheeled cart — ferries bricks from pallet to handover
    mortar_bot.py    wheeled cart — drives along wall, extends nozzle, applies mortar
    workers.py       adapts task-graph tasks to robot action generators

  tasks/
    task_graph.py    DAG data structure
    wall_dag.py      masonry dependency rules
    executor.py      concurrent tick scheduler (generators, no threads)

  brain/
    planner.py       NL command → wall spec (Gemini + regex fallback)
    supervisor.py    LLM tactical supervisor (failure recovery, milestones)
    tools.py         function-calling tools the LLM can invoke
    prompts.py       system prompts and few-shot examples
    world_state.py   shared state model (blackboard pattern)
```


## About the LLM integration

I'm using Google Gemini free tier (Gemini 2.0 Flash). It has 15 requests per
minute which is more than enough since the supervisor only calls the LLM when
something interesting happens (a failure, a row completion, the end of the
build) — not on every simulation tick.

If you don't have an API key or don't want to use the LLM, pass `--no-llm`
and everything still works with a regex-based parser and deterministic
supervisor logic. The robots and physics don't change at all — the LLM is
purely the decision-making layer on top.

To get a key: go to https://aistudio.google.com/apikey — it's free.


## Requirements

- Python 3.8+
- PyBullet
- NumPy
- OpenCV (opencv-python)
- google-genai (for Gemini API, optional)
- matplotlib (for Gantt charts, optional)

See `requirements.txt` for exact versions.
