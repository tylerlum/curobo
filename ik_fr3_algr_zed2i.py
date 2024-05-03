from curobo.geom.sdf.world import CollisionCheckerType
import yaml
import trimesh
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
import pybullet as pb
from argparse import ArgumentParser
import torch
import numpy as np
import time
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from scipy.spatial.transform import Rotation

from typing import Optional
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
import transforms3d


def solve_ik(
    X_W_H: np.ndarray,
    q_algr_constraint: Optional[np.ndarray] = None,
    collision_check_object: bool = True,
) -> np.ndarray:
    assert X_W_H.shape == (4, 4)
    trans = X_W_H[:3, 3]
    rot_matrix = X_W_H[:3, :3]
    quat_wxyz = transforms3d.quaternions.mat2quat(rot_matrix)

    target_pose = Pose(
        torch.from_numpy(trans).float().cuda(),
        quaternion=torch.from_numpy(quat_wxyz).float().cuda(),
    )

    tensor_args = TensorDeviceType()
    robot_file = "fr3_algr_zed2i.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )

    # Apply joint limits
    if q_algr_constraint is not None:
        assert q_algr_constraint.shape == (16,)
        assert robot_cfg.kinematics.kinematics_config.joint_limits.position.shape == (2, 23)
        robot_cfg.kinematics.kinematics_config.joint_limits.position[0, 7:] = (
            torch.from_numpy(q_algr_constraint).float().cuda() - 0.01
        )
        robot_cfg.kinematics.kinematics_config.joint_limits.position[1, 7:] = (
            torch.from_numpy(q_algr_constraint).float().cuda() + 0.01
        )

    world_file = "TYLER_scene_with_object.yml" if collision_check_object else "TYLER_scene.yml"
    world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)

    result = ik_solver.solve_single(target_pose)
    if not result.success.item():
        raise RuntimeError("IK failed to find a solution.")

    q_solution = result.solution[result.success]
    assert q_solution.shape == (1, 23)
    return q_solution.squeeze(dim=0).detach().cpu().numpy()


def main() -> None:
    X_W_H_feasible = np.array(
        [
            [0, 0, 1, 0.4],
            [0, 1, 0, 0.0],
            [-1, 0, 0, 0.15],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    X_W_H_collide_object = np.array(
        [
            [0, 0, 1, 0.65],
            [0, 1, 0, 0.0],
            [-1, 0, 0, 0.15],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    X_W_H_collide_table = np.array(
        [
            [0, 0, 1, 0.4],
            [0, 1, 0, 0.0],
            [-1, 0, 0, 0.10],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    q_algr_pre = np.array(
        [
            0.29094562,
            0.7371094,
            0.5108592,
            0.12263706,
            0.12012535,
            0.5845135,
            0.34382993,
            0.605035,
            -0.2684319,
            0.8784579,
            0.8497135,
            0.8972184,
            1.3328283,
            0.34778783,
            0.20921567,
            -0.00650969,
        ]
    )
    q_feasible = solve_ik(
        X_W_H=X_W_H_feasible, q_algr_constraint=q_algr_pre, collision_check_object=False
    )
    q_feasible_2 = solve_ik(
        X_W_H=X_W_H_feasible, q_algr_constraint=q_algr_pre, collision_check_object=True
    )
    q_feasible_3 = solve_ik(
        X_W_H=X_W_H_collide_object, q_algr_constraint=q_algr_pre, collision_check_object=False
    )
    print(f"q_feasible: {q_feasible}")
    print(f"q_feasible_2: {q_feasible_2}")
    print(f"q_feasible_3: {q_feasible_3}")

    try:
        q_collide_object = solve_ik(
            X_W_H=X_W_H_collide_object, q_algr_constraint=q_algr_pre, collision_check_object=True
        )
        raise RuntimeError("Collision check failed to detect collision.")
    except RuntimeError:
        print("Collision check successfully detected collision.")

    try:
        q_collide_table = solve_ik(
            X_W_H=X_W_H_collide_table, q_algr_constraint=q_algr_pre, collision_check_object=False
        )
        raise RuntimeError("Collision check failed to detect collision.")
    except RuntimeError:
        print("Collision check successfully detected collision.")


if __name__ == "__main__":
    main()
