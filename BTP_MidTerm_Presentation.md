# BTP Mid-Term Presentation
**Multi-Robot Task Orchestration for Construction Assembly**

---

## Slide 1: Project Overview
* **Final Goal:** Develop a multi-agent system where heterogeneous robots work collaboratively to construct a complex brick wall.
* **Key Objectives:**
  * Precise millimeter-level placement.
  * Scalable task orchestration framework.
  * Seamless, collision-free multi-robot coordination.
* **Significance:** Bridges the gap between high-level cognitive planning (LLM) and low-level control for automated masonry.

---

## Slide 2: Progress Till Now
* **Simulation Environment:** Fully functional PyBullet setup with realistic physics.
* **Robot Control:** 
  * 7-DOF Panda arm with grasp-error compensation.
  * Kinematic shuttle & mortar bots for material transport.
* **Vision Integration:** 3D CV pipeline for overhead camera detection of bricks.
* **Task Orchestration & LLM Brain:** Not started yet.

**Scene Placement Concept:**
![Simulation Scene Placement](C:\Users\Nawang Tashi\.gemini\antigravity-ide\brain\cbbac865-4056-4654-a0c0-6439bfbecf31\scene_placement_1790571874278.jpg)

---

## Slide 3: Quantified Current Progress (Benchmarking)
* **Vision Perception Accuracy (Overhead Camera):**
  * Achieved highly accurate detection using RGB-D Point Cloud PCA.
  * **Mean 3D Pos Error:** 3.96 mm.
  * **Mean Yaw Error:** 0.18 degrees.
* **Kinematic Placement Precision:**
  * Demonstrated robust grasp-error compensation during final placement.
  * **Typical Mean Placement Deviation:** ~4.0 mm.
* **System Reliability & Fault Recovery:**
  * Implemented an autonomous, closed-loop recovery state machine for dropped or slipped bricks.
  * **First-try Success Rate:** 65%.
  * **Ultimate Success Rate:** 100% (Arm safely detects slip, re-homes, and recovers).

---

## Slide 4: Current Challenges & Problems
* **Workspace Constraints:** 
  * Primary robotic arm operates within a fixed, static workspace.
  * Reaches limit the maximum size, length, and height of the wall.
* **Scalability:** 
  * Constructing larger walls requires dynamic base placement or a mobile base.
  * Managing kinematics and localization for a mobile manipulator introduces significant complexity.

**Vision & Planning Concept:**
![Vision Planning](C:\Users\Nawang Tashi\.gemini\antigravity-ide\brain\cbbac865-4056-4654-a0c0-6439bfbecf31\vision_planning_1790571889723.jpg)

---

## Slide 5: Future Goals (Next 2 Months)
* **Refine LLM Orchestration:** Transition to a prompt-engineered LLM brain for strategic planning based on natural language.
* **Improve Physics Realism:** Implement dynamic grasping interactions and accurate mortar fluid dynamics.
* **Comprehensive Benchmarking:** Conduct automated evaluations for placement accuracy, timeline efficiency, and failure recovery.
* **Final Project Deliverables:** Finalize documentation, prepare viva defense, and open-source the codebase.
