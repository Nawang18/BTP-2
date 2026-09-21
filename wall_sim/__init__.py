"""Multi-robot task orchestration for construction assembly (BTP project).

Package layout:
    config      -- all physical/world parameters in one place
    wall_plan   -- turns a wall spec into exact target brick poses
    scene       -- pybullet world, bricks, pallet, outlines, pedestal
    robots      -- PandaArm (arm+gripper), Shuttle (transport), MortarBot
    tasks       -- task graph (DAG), wall DAG builder, executor
    brain       -- command -> plan parsing (LLM front-end stub for later)
    metrics     -- placement error + task timeline reports
"""
