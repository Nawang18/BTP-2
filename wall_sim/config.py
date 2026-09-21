"""Central configuration: every physical/layout parameter lives here.

Units: metres, kg, seconds. Brick width is deliberately < 0.08 m because the
Panda gripper has an 80 mm stroke -- the fingers must close past both faces
of the brick for a friction grip.
"""
from dataclasses import dataclass, field


@dataclass
class BrickDims:
    length: float = 0.20
    width: float = 0.075
    height: float = 0.06
    gap: float = 0.005      # mortar joint thickness
    mass: float = 1.5


@dataclass
class CameraConfig:
    """Overhead camera parameters for vision-based perception."""
    eye_pos: tuple = (0.48, 0.0, 1.15)
    target_pos: tuple = (0.48, 0.0, 0.0)
    up_vector: tuple = (0.0, 1.0, 0.0)
    img_width: int = 640
    img_height: int = 640
    fov: float = 55.0
    near_val: float = 0.1
    far_val: float = 2.0
    # HSV color segmentation bounds for terracotta brick
    # Range 1 covers low-hue reds (0-15), range 2 covers wrap-around reds (165-180)
    hsv_lower1: tuple = (0, 70, 50)
    hsv_upper1: tuple = (15, 255, 255)
    hsv_lower2: tuple = (165, 70, 50)
    hsv_upper2: tuple = (180, 255, 255)
    min_contour_area: int = 300    # minimum pixel area to accept a detection


@dataclass
class Config:
    brick: BrickDims = field(default_factory=BrickDims)
    camera: CameraConfig = field(default_factory=CameraConfig)

    # --- layout -------------------------------------------------------------
    wall_origin: tuple = (0.02, 0.30, 0.0)   # wall runs along +x from here
    wall_rows: int = 3
    wall_bricks_per_row: int = 3             # full bricks in even rows

    pallet_xy: tuple = (0.0, -0.30)         # single brick stack (shuttle+arm reach)
    handover_pos: tuple = (0.50, -0.05)      # pedestal where shuttle -> arm
    pedestal_size: tuple = (0.12, 0.12, 0.05)
    shuttle_park: tuple = (0.80, -0.45)
    shuttle_load_dock_offset: float = 0.18   # dock this far -y of the stack
    shuttle_lane: tuple = (0.86, -0.48)      # east/south transit lane: routes the
                                              # chassis AROUND the stack & pedestal
                                              # (diagonal sweeps knock the stack over)
    shuttle_deposit_stop: tuple = (0.74, -0.05)  # stop east of pedestal (>=40 mm
                                                  # clear of the brick edge), then
                                                  # teleport brick onto it
    mortar_park: tuple = (1.05, 0.55, 0.35)

    # --- workspace bounds (for random brick spawning) -----------------------
    spawn_x_range: tuple = (0.40, 0.56)     # safe arm-reachable X range
    spawn_y_range: tuple = (-0.20, 0.15)     # safe arm-reachable Y range

    # --- simulation ---------------------------------------------------------
    sim_hz: float = 240.0

    # --- arm motion ---------------------------------------------------------
    arm_speed: float = 0.15                  # m/s cartesian (stable, low-jerk motion)
    transit_z: float = 0.58                  # clearance height for lateral moves
                                              # (fingertips reach 0.112 below the
                                              #  hand; must clear stack top 0.42)
    approach_height: float = 0.08
    lift_height: float = 0.10
    grip_depth: float = 0.085                # panda_hand frame above brick centre when grasping
    gripper_open: float = 0.048              # per finger (96 mm stroke provides
                                              # safe clearance over 75 mm brick)
    gripper_force: float = 55.0              # firm friction hold on 1.5 kg brick
    gripper_hold_s: float = 0.6
    arm_force: float = 150.0                 # Nm on arm joints
    arm_max_joint_vel: float = 1.5           # rad/s joint velocity limit (prevents whipping)

    # --- shuttle / mortar ---------------------------------------------------
    shuttle_speed: float = 0.15              # m/s (kinematic)
    mortar_speed: float = 0.30

    # --- colours ------------------------------------------------------------
    brick_rgba: tuple = (0.72, 0.25, 0.15, 1.0)
    half_brick_rgba: tuple = (0.60, 0.20, 0.12, 1.0)
    mortar_rgba: tuple = (0.55, 0.58, 0.52, 1.0)
    shuttle_rgba: tuple = (0.20, 0.35, 0.70, 1.0)
    mortar_head_rgba: tuple = (0.85, 0.65, 0.10, 1.0)
    pedestal_rgba: tuple = (0.6, 0.6, 0.6, 1.0)

    # --- outputs ------------------------------------------------------------
    results_dir: str = "results"

    def pedestal_brick_center(self):
        """Where a brick rests on the handover pedestal."""
        return (self.handover_pos[0], self.handover_pos[1],
                self.pedestal_size[2] + self.brick.height / 2.0)
