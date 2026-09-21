"""Franka Panda driver: analytic-free IK via pybullet's nullspace IK,
position-controlled joints, and pick/place behaviours written as *generators*
that yield once per simulation step. The caller (demo loop or executor) owns
`p.stepSimulation()`, so the same code works standalone or under orchestration.

Why generators: the executor advances several robots one step at a time so
all robots move concurrently without threads.
"""
import math

import numpy as np
import pybullet as p
import pybullet_data

from ..config import Config


class GripFailed(RuntimeError):
    pass


class PlaceFailed(RuntimeError):
    pass


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def slerp(q0, q1, t):
    q0 = np.array(q0, dtype=float)
    q1 = np.array(q1, dtype=float)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return tuple(q / np.linalg.norm(q))
    th0 = math.acos(min(1.0, d))
    s = math.sin(th0)
    return tuple((math.sin((1 - t) * th0) / s) * q0 + (math.sin(t * th0) / s) * q1)


def gripper_down_quaternion(yaw=0.0):
    """EE orientation: approach axis pointing down (-z), fingers closing
    along world +/-y rotated by yaw. Matches bricks whose length runs +x."""
    qx_pi = (1.0, 0.0, 0.0, 0.0)                       # 180 deg about x
    qz = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    return qmul(qz, qx_pi)


def mat2quat(m):
    """Converts a 3x3 rotation matrix to a normalized quaternion (x, y, z, w)."""
    m = np.asarray(m, dtype=float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return tuple(q)


def gripper_3d_orientation(normal, length_axis):
    """Computes a 3D gripper orientation quaternion that aligns the approach axis
    with -normal (approaching perpendicular to the tilted face) and finger axis
    with the brick's width axis."""
    normal = np.asarray(normal, dtype=float)
    norm_len = np.linalg.norm(normal)
    if norm_len > 1e-6:
        normal /= norm_len
    if normal[2] < 0:
        normal = -normal

    length_axis = np.asarray(length_axis, dtype=float)
    length_axis = length_axis - np.dot(length_axis, normal) * normal
    len_mag = np.linalg.norm(length_axis)
    if len_mag > 1e-6:
        length_axis /= len_mag
    else:
        length_axis = np.array([1.0, 0.0, 0.0])

    col0 = length_axis
    col2 = -normal
    col1 = np.cross(col2, col0)
    col1 /= np.linalg.norm(col1)
    R = np.column_stack([col0, col1, col2])
    return mat2quat(R)


class PandaArm:
    def __init__(self, cfg: Config, base_position=(0.0, 0.0, 0.0)):
        self.cfg = cfg
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.body = p.loadURDF("franka_panda/panda.urdf",
                               basePosition=base_position, useFixedBase=True)
        self._map_joints()
        for fj in self.finger_joints:
            p.changeDynamics(self.body, fj, jointLowerLimit=0.0, jointUpperLimit=0.05,
                             lateralFriction=1.2, spinningFriction=0.5)
        self.held = None
        self.grasp_offset = np.zeros(3)   # brick pos error measured after grasp
        self.grasp_yaw_err = 0.0          # brick yaw error after grasp

    # -- setup ---------------------------------------------------------------
    def _map_joints(self):
        self.arm_joints, self.finger_joints, self.movable = [], [], []
        self.ee_link = None
        n = p.getNumJoints(self.body)
        for i in range(n):
            info = p.getJointInfo(self.body, i)
            jtype = info[2]
            if jtype == p.JOINT_REVOLUTE:
                self.arm_joints.append(i)
                self.movable.append(i)
            elif jtype == p.JOINT_PRISMATIC:
                self.finger_joints.append(i)
                self.movable.append(i)
            if info[12].decode() == "panda_hand":
                self.ee_link = i
        assert self.ee_link is not None, "panda_hand link not found"

        # Seed at a comfortable elbow-up config. NOTE: we deliberately do NOT
        # use nullspace-biased IK -- pybullet's nullspace gain trades primary
        # task accuracy away (~2 mm steady-state error on the ee pose).
        # Seeding + per-step re-solve keeps the redundant posture continuous.
        self.rest_config = [0.0, -0.35, 0.0, -1.40, 0.0, 1.30, 0.79]
        for j, q in zip(self.arm_joints, self.rest_config):
            p.resetJointState(self.body, j, q)
        for j in self.finger_joints:
            p.resetJointState(self.body, j, self.cfg.gripper_open)

    # -- state ---------------------------------------------------------------
    def ee_pose(self):
        st = p.getLinkState(self.body, self.ee_link)
        return np.array(st[4]), np.array(st[5])

    def brick_center_from_ee(self, ee_z):
        """Brick centre z when gripper holds a brick at ee height ee_z."""
        return ee_z - self.cfg.grip_depth

    # -- IK / commands -------------------------------------------------------
    def solve_ik(self, pos, orn):
        sol = p.calculateInverseKinematics(
            self.body, self.ee_link, pos, orn,
            maxNumIterations=200, residualThreshold=1e-8)
        # solution covers all movable joints (7 arm + 2 fingers) in order
        if len(sol) == len(self.movable):
            return dict(zip(self.movable, sol))
        if len(sol) == len(self.arm_joints):
            return dict(zip(self.arm_joints, sol))
        raise RuntimeError(f"unexpected IK solution length {len(sol)}")

    def check_reachability(self, target_pos, yaw=0.0, orn=None, normal=None):
        """Tests whether a 3D target position can be reached with the arm.
        
        Returns:
            (is_reachable: bool, reason: str)
        """
        x, y, z = target_pos
        r_xy = math.hypot(x, y)

        # 1. Geometric workspace bounds check (Franka Emika Panda)
        if r_xy > 0.82:
            return False, f"radial distance {r_xy:.3f}m exceeds maximum arm reach (0.82m)"
        if r_xy < 0.20:
            return False, f"radial distance {r_xy:.3f}m is within the robot inner base singularity zone (<0.20m)"
        if z < -0.02 or z > 0.85:
            return False, f"Z height {z:.3f}m is outside vertical workspace limits [0.0m, 0.85m]"

        # 2. Kinematic solver verification (IK residual error)
        if orn is None:
            orn = gripper_down_quaternion(yaw)

        if normal is not None:
            n = np.asarray(normal, dtype=float)
            n_len = np.linalg.norm(n)
            if n_len > 1e-6:
                n /= n_len
            if n[2] < 0:
                n = -n
            grasp_target = tuple(np.array(target_pos) + n * self.cfg.grip_depth)
        else:
            grasp_target = (x, y, z + self.cfg.grip_depth)

        # Save active joint states
        saved_q = [p.getJointState(self.body, j)[0] for j in self.arm_joints]
        try:
            sol = self.solve_ik(grasp_target, orn)
            for j in self.arm_joints:
                p.resetJointState(self.body, j, sol[j])
            ee_p, _ = self.ee_pose()
            err = float(np.linalg.norm(ee_p - np.array(grasp_target)))
            if err > 0.035:
                return False, f"IK solver residual error {err*1000:.1f} mm exceeds tolerance (joint limits or singularity)"
            return True, "Reachable"
        except Exception as e:
            return False, f"IK calculation error: {e}"
        finally:
            for j, q in zip(self.arm_joints, saved_q):
                p.resetJointState(self.body, j, q)

    def _command_arm(self, ee_pos, ee_orn):
        targets = self.solve_ik(ee_pos, ee_orn)
        for j in self.arm_joints:
            p.setJointMotorControl2(
                self.body, j, p.POSITION_CONTROL,
                targetPosition=targets[j], force=self.cfg.arm_force,
                maxVelocity=self.cfg.arm_max_joint_vel)

    def _command_gripper(self, target):
        for j in self.finger_joints:
            p.setJointMotorControl2(
                self.body, j, p.POSITION_CONTROL,
                targetPosition=target, force=self.cfg.gripper_force)

    # -- motion primitives (generators: one yield == one sim step) -----------
    def move_gen(self, pos, orn, speed=None, tol=0.0005, hold_timeout_s=2.0):
        cfg = self.cfg
        speed = speed or cfg.arm_speed
        cur_p, cur_o = self.ee_pose()
        dist = float(np.linalg.norm(np.asarray(pos) - cur_p))
        steps = max(int(dist / speed * cfg.sim_hz), 40)
        for k in range(1, steps + 1):
            t = k / steps
            ip = (1 - t) * cur_p + t * np.asarray(pos)
            io = slerp(cur_o, orn, t)
            self._command_arm(ip, io)
            yield
        # hold: re-solve at the final target until the ee is inside tol,
        # so every waypoint ends sub-mm regardless of solver transients
        target = np.asarray(pos, dtype=float)
        for _ in range(int(hold_timeout_s * cfg.sim_hz)):
            self._command_arm(target, orn)
            ee_p, _ = self.ee_pose()
            yield
            if float(np.linalg.norm(ee_p - target)) < tol:
                break

    def gripper_gen(self, target, hold_s=None):
        self._command_gripper(target)
        hold = hold_s if hold_s is not None else self.cfg.gripper_hold_s
        for _ in range(int(hold * self.cfg.sim_hz)):
            yield

    def wait_gen(self, seconds):
        for _ in range(int(seconds * self.cfg.sim_hz)):
            yield

    def home_gen(self):
        yield from self.gripper_gen(self.cfg.gripper_open)
        yield from self.move_gen((0.35, 0.0, self.cfg.transit_z),
                                 gripper_down_quaternion(0.0))

    def transit_gen(self, x, y, z, yaw=0.0, orn=None):
        """Collision-safe route: rise vertically, move laterally at
        clearance height, then descend vertically at the target column.
        Maintains target orientation so descending fingers do not collide with tilted/rotated bricks."""
        cfg = self.cfg
        if orn is None:
            orn = gripper_down_quaternion(yaw)
        cur_p, _ = self.ee_pose()
        if cur_p[2] < cfg.transit_z:
            yield from self.move_gen((cur_p[0], cur_p[1], cfg.transit_z), orn)
        yield from self.move_gen((x, y, cfg.transit_z), orn)
        yield from self.move_gen((x, y, z), orn)

    # -- behaviours ----------------------------------------------------------
    def pick_gen(self, brick_id, brick_center, yaw=0.0, orn=None, normal=None):
        cfg = self.cfg
        cx, cy, cz = brick_center
        if orn is None:
            orn = gripper_down_quaternion(yaw)

        if normal is not None:
            n = np.asarray(normal, dtype=float)
            n_len = np.linalg.norm(n)
            if n_len > 1e-6:
                n /= n_len
            if n[2] < 0:
                n = -n
            hand_above = tuple(np.array(brick_center) + n * (cfg.grip_depth + cfg.approach_height))
            hand_grasp = tuple(np.array(brick_center) + n * cfg.grip_depth)
        else:
            hand_above = (cx, cy, cz + cfg.grip_depth + cfg.approach_height)
            hand_grasp = (cx, cy, cz + cfg.grip_depth)

        yield from self.transit_gen(hand_above[0], hand_above[1], hand_above[2], orn=orn)
        yield from self.move_gen(hand_grasp, orn, speed=cfg.arm_speed / 2.0)
        yield from self.gripper_gen(0.0, hold_s=cfg.gripper_hold_s)

        pre_z = p.getBasePositionAndOrientation(brick_id)[0][2]
        yield from self.move_gen((hand_grasp[0], hand_grasp[1],
                                  hand_grasp[2] + cfg.lift_height), orn)
        post_z = p.getBasePositionAndOrientation(brick_id)[0][2]
        if post_z < pre_z + 0.03:
            raise GripFailed(f"brick {brick_id} not held after lift "
                            f"({pre_z:.3f} -> {post_z:.3f}); check finger "
                            f"friction / grip_depth")
        self.held = brick_id
        # Grasp-error compensation: measure where the brick actually sits in
        # the gripper (stand-in for wrist-camera feedback) so the place step
        # can subtract it.
        bpos, bq = p.getBasePositionAndOrientation(brick_id)
        expected = np.array(brick_center) + (0.0, 0.0, cfg.lift_height)
        self.grasp_offset = np.array(bpos) - expected
        self.grasp_yaw_err = p.getEulerFromQuaternion(bq)[2] - yaw
        print(f"[grip] {brick_id} grasp offset "
              f"{np.round(self.grasp_offset*1000, 1)} mm, "
              f"yaw {np.degrees(self.grasp_yaw_err):.2f} deg")

        # Grasp-quality check: if the brick slipped down by > 35 mm, re-grasp required
        if self.grasp_offset[2] < -0.035:
            raise GripFailed(f"brick {brick_id} grasped with excessive slip "
                            f"(offset Z: {self.grasp_offset[2]*1000:.1f} mm); "
                            f"re-grasp required")

    def place_gen(self, target_center, yaw=0.0):
        cfg = self.cfg
        # subtract the measured grasp offset: command the hand so the BRICK
        # (not the hand) lands on target
        tgt = (np.asarray(target_center, dtype=float) - self.grasp_offset)
        cx, cy, cz = tgt
        orn = gripper_down_quaternion(yaw - self.grasp_yaw_err)
        hand_above = (cx, cy, cz + cfg.grip_depth + cfg.approach_height)
        hand_place = (cx, cy, cz + cfg.grip_depth)

        yield from self.transit_gen(cx, cy, hand_above[2], yaw=yaw - self.grasp_yaw_err)
        yield from self.move_gen(hand_place, orn, speed=cfg.arm_speed / 2.0)
        yield from self.gripper_gen(cfg.gripper_open, hold_s=0.5)
        yield from self.wait_gen(0.2)   # let the brick settle before verifying

        # Closed-loop release verification: check if brick actually detached from gripper
        placed_brick = self.held
        if placed_brick is not None:
            bpos, _ = p.getBasePositionAndOrientation(placed_brick)
            if bpos[2] > target_center[2] + 0.035:
                raise PlaceFailed(f"brick {placed_brick} remained stuck in gripper at Z={bpos[2]:.3f}m "
                                  f"(target Z={target_center[2]:.3f}m)")

        self.held = None
        yield from self.move_gen((hand_place[0], hand_place[1],
                                  hand_place[2] + cfg.lift_height), orn)
        yield from self.wait_gen(0.3)   # let the brick settle before measuring

    def dislodge_gen(self, target_center, yaw=0.0):
        """Recovery maneuver when a brick is stuck in the gripper during placement:
        lowers hand, cycles gripper fingers, and applies a micro-release wiggle."""
        cfg = self.cfg
        tgt = (np.asarray(target_center, dtype=float) - self.grasp_offset)
        cx, cy, cz = tgt
        orn = gripper_down_quaternion(yaw - self.grasp_yaw_err)
        hand_place = (cx, cy, cz + cfg.grip_depth)

        print("[Recovery] Executing dislodge maneuver (descend, cycle fingers, wiggle)...")
        yield from self.move_gen(hand_place, orn, speed=cfg.arm_speed / 2.0)
        # Cycle fingers: partially close and reopen fully to break friction stiction
        yield from self.gripper_gen(0.025, hold_s=0.2)
        yield from self.gripper_gen(cfg.gripper_open, hold_s=0.4)

        # Micro-wiggle in yaw to free pinched corners
        for d_yaw in [0.03, -0.03, 0.0]:
            wiggle_orn = gripper_down_quaternion(yaw - self.grasp_yaw_err + d_yaw)
            yield from self.move_gen(hand_place, wiggle_orn, speed=cfg.arm_speed / 2.0)

        yield from self.gripper_gen(cfg.gripper_open, hold_s=0.4)
        yield from self.wait_gen(0.2)

        # Check if detached
        if self.held is not None:
            bpos, _ = p.getBasePositionAndOrientation(self.held)
            if bpos[2] <= target_center[2] + 0.035:
                self.held = None
                print("[Recovery] Dislodge succeeded: brick detached onto target slot.")
                yield from self.move_gen((hand_place[0], hand_place[1],
                                          hand_place[2] + cfg.lift_height), orn)
            else:
                raise PlaceFailed(f"brick {self.held} still stuck after dislodge maneuver")

    # -- single-robot convenience -------------------------------------------
    def run(self, gen):
        """Blocking driver for demos without the executor."""
        for _ in gen:
            p.stepSimulation()
