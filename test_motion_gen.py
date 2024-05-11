import torch

# cuRobo
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


from curobo.geom.types import WorldConfig

world_config = {
    "mesh": {
        "new_scene": {
            "pose": [0, 0, 0, 1, 0, 0, 0],
            # "file_path": "scene/nvblox/srl_ur10_bins.obj",
            # "file_path": "/juno/u/tylerlum/Downloads/cube.obj",
            "file_path": "/juno/u/tylerlum/github_repos/DexGraspNet/data/rotated_meshdata/core-bottle-1071fa4cddb2da2fc8724d5673a063a6/coacd/decomposed.obj",
        },
    },
    "cuboid": {
        "table": {
            "dims": [5.0, 5.0, 0.2],  # x, y, z
            "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
        },
    },
}

world_config2 = {
    "mesh": {
        "base_scene": {
            "pose": [10.5, 0.080, 1.6, 0.043, -0.471, 0.284, 0.834],
            "file_path": "scene/nvblox/srl_ur10_bins.obj",
            # "file_path": "/juno/u/tylerlum/Downloads/cube.obj",
            # "file_path": "/juno/u/tylerlum/github_repos/DexGraspNet/data/rotated_meshdata/core-bottle-1071fa4cddb2da2fc8724d5673a063a6/coacd/decomposed.obj",
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
    world_config2,
    interpolation_dt=0.01,
)
motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup()

retract_cfg = motion_gen.get_retract_config()

state = motion_gen.rollout_fn.compute_kinematics(
    JointState.from_position(retract_cfg.view(1, -1))
)

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
motion_gen.update_world(WorldConfig.from_dict(world_config))

result = motion_gen.plan_single(start_state, goal_pose, MotionGenPlanConfig(max_attempts=1))
traj = result.get_interpolated_plan()  # result.optimized_dt has the dt between timesteps
print("Trajectory Generated: ", result.success)
