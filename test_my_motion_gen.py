#### CHANGES START ####
from visualizer import start_visualizer, get_retract_config, set_robot_state, animate_robot, Object, create_urdf
import torch
import pathlib
from curobo.util_file import (
    get_assets_path,
    get_robot_configs_path,
    join_path,
    load_yaml,
)

#### CHANGES END ####

# cuRobo
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

world_config = {
    "mesh": {
        "base_scene": {
            "pose": [3.5, 0.080, 1.6, 0.043, -0.471, 0.284, 0.834],
            "file_path": "scene/nvblox/srl_ur10_bins.obj",
        },
    },
    "cuboid": {
        "table": {
            "dims": [5.0, 5.0, 0.2],  # x, y, z
            "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
        },
    },
}

motion_gen_config = MotionGenConfig.load_from_robot_config(
    "ur5e.yml",
    world_config,
    interpolation_dt=0.01,
)
motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup()

retract_cfg = motion_gen.get_retract_config()

state = motion_gen.rollout_fn.compute_kinematics(JointState.from_position(retract_cfg.view(1, -1)))

goal_pose = Pose.from_list([-0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0])  # x, y, z, qw, qx, qy, qz
start_state = JointState.from_position(
    torch.zeros(1, 6).cuda(),
    joint_names=[
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ],
)

result = motion_gen.plan_single(start_state, goal_pose, MotionGenPlanConfig(max_attempts=1))
traj = result.get_interpolated_plan()  # result.optimized_dt has the dt between timesteps
print("Trajectory Generated: ", result.success)

#### CHANGES START ####
dt = result.interpolation_dt
qs = traj.position.cpu().numpy()



object_urdf_path = create_urdf(pathlib.Path(join_path(get_assets_path(), "scene/nvblox/srl_ur10_bins.obj")))

objects = [
    Object(
        urdf_path=object_urdf_path,
        xyz=(10.5, 0.080, 1.6),
        quat_wxyz=(0.043, -0.471, 0.284, 0.834),
    ),
]
pb_robot = start_visualizer(objects=objects)
# retract_q = get_retract_config()
# set_robot_state(pb_robot, qs)
animate_robot(pb_robot, qs, dt=dt)
breakpoint()
#### CHANGES END ####
