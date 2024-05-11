#### CHANGES START ####
from visualizer import (
    start_visualizer,
    set_robot_state,
    animate_robot,
    Object,
    create_urdf,
)
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

object_obj_path = "scene/nvblox/mug.obj"
object_xyz = (0.65, 0.0, 0.0)
object_quat_wxyz = (1.0, 0.0, 0.0, 0.0)
table_obj_path = "scene/nvblox/table.obj"
table_xyz = (0.50165, 0.0, -0.01)
table_quat_wxyz = (1.0, 0.0, 0.0, 0.0)

world_config = {
    "mesh": {
        "base_scene": {
            "pose": [*object_xyz, *object_quat_wxyz],
            "file_path": object_obj_path,
        },
        "table": {
            "pose": [*table_xyz, *table_quat_wxyz],  # x, y, z, qw, qx, qy, qz
            # "pose": [10, 0,0, 1, 0, 0, 0],  # x, y, z, qw, qx, qy, qz
            "file_path": table_obj_path,
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

# goal_ee_pos = state.ee_pos_seq.squeeze() + torch.tensor([0.0, 0.0, 0.1]).to(state.ee_pos_seq.device)
goal_ee_pos = torch.tensor([0.65, 0.1, 0.06]).to(state.ee_pos_seq.device)
goal_ee_quat = state.ee_quat_seq.squeeze()
goal_pose = Pose(goal_ee_pos, quaternion=goal_ee_quat)
start_state = JointState.from_position(
    retract_cfg.view(1, -1).clone(),
    joint_names=[
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ],
)

result = motion_gen.plan_single(start_state, goal_pose, MotionGenPlanConfig(max_attempts=1, enable_graph=False, enable_opt=True))
print("Trajectory Generated: ", result.success)
if result.success.item():
    traj = result.get_interpolated_plan()  # result.optimized_dt has the dt between timesteps

    #### CHANGES START ####
    dt = result.interpolation_dt
    qs = traj.position.cpu().numpy()


    object_urdf_path = create_urdf(pathlib.Path(join_path(get_assets_path(), object_obj_path)))
    table_urdf_path = create_urdf(pathlib.Path(join_path(get_assets_path(), table_obj_path)))

    objects = [
        Object(
            urdf_path=object_urdf_path,
            xyz=object_xyz,
            quat_wxyz=object_quat_wxyz,
        ),
        Object(
            urdf_path=table_urdf_path,
            xyz=table_xyz,
            quat_wxyz=table_quat_wxyz,
        ),
    ]
    pb_robot = start_visualizer(objects=objects)
    # set_robot_state(pb_robot, qs)
    animate_robot(pb_robot, qs, dt=dt)
    breakpoint()
#### CHANGES END ####

from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
config = RobotWorldConfig.load_from_config("ur5e.yml", world_config,
                                          collision_activation_distance=0.0)
from curobo.types.base import TensorDeviceType
curobo_fn = RobotWorld(config)

import numpy as np
qs = np.array([
    [-5.2642569e-02, -7.6695609e-01,  1.7180086e+00, -2.6310635e+00,
        -1.5632669e+00, -5.2167010e-02],
    [-5.2642569e-02, -7.6695609e-01,  1.7180086e+00, -2.6310635e+00,
        -1.5632669e+00, -5.2167010e-02],
]
)


# create spheres with shape batch, horizon, n_spheres, 4.
tensor_args = TensorDeviceType()
d_world, d_self = curobo_fn.get_world_self_collision_distance_from_joints(torch.from_numpy(qs).to(tensor_args.device).to(tensor_args.dtype))
print(f"d_world: {d_world}")
print(f"d_self: {d_self}")
