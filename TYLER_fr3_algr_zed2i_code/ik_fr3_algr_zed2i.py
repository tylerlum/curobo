import pathlib
from typing import Optional, Tuple

import numpy as np
import torch
import transforms3d
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from fr3_algr_zed2i_world import (
    get_dummy_collision_dict,
    get_object_collision_dict,
    get_table_collision_dict,
)


def solve_ik(
    X_W_H: np.ndarray,
    q_algr_constraint: Optional[np.ndarray] = None,
    collision_check_object: bool = True,
    obj_filepath: Optional[pathlib.Path] = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-05-02_16-19-22/nerf_to_mesh/mug_330/coacd/decomposed.obj"
    ),
    obj_xyz: Tuple[float, float, float] = (0.65, 0.0, 0.0),
    obj_quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    collision_check_table: bool = True,
    raise_if_no_solution: bool = True,
    warn_if_no_solution: bool = False,
) -> np.ndarray:
    assert X_W_H.shape == (4, 4), f"X_W_H.shape: {X_W_H.shape}"
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
        assert q_algr_constraint.shape == (
            16,
        ), f"q_algr_constraint.shape: {q_algr_constraint.shape}"
        assert robot_cfg.kinematics.kinematics_config.joint_limits.position.shape == (2, 23)
        robot_cfg.kinematics.kinematics_config.joint_limits.position[0, 7:] = (
            torch.from_numpy(q_algr_constraint).float().cuda() - 0.01
        )
        robot_cfg.kinematics.kinematics_config.joint_limits.position[1, 7:] = (
            torch.from_numpy(q_algr_constraint).float().cuda() + 0.01
        )

    world_dict = {}
    if collision_check_table:
        world_dict.update(get_table_collision_dict())
    if collision_check_object and obj_filepath is not None:
        world_dict.update(
            get_object_collision_dict(file_path=obj_filepath, xyz=obj_xyz, quat_wxyz=obj_quat_wxyz)
        )
    if len(world_dict) == 0:
        world_dict.update(
            get_dummy_collision_dict()
        )
    world_cfg = WorldConfig.from_dict(world_dict)
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
        if raise_if_no_solution:
            raise RuntimeError("IK failed to find a valid solution.")
        elif warn_if_no_solution:
            print("WARNING: IK failed to find a valid solution.")

    assert result.solution.shape == (1, 1, 23)
    return result.solution.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()


def max_penetration_from_q(
    q: np.ndarray,
    include_object: bool = True,
    obj_filepath: Optional[pathlib.Path] = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-05-02_16-19-22/nerf_to_mesh/mug_330/coacd/decomposed.obj"
    ),
    obj_xyz: Tuple[float, float, float] = (0.65, 0.0, 0.0),
    obj_quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    include_table: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    assert q.shape == (23,), f"q.shape: {q.shape}"

    robot_file = "fr3_algr_zed2i.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )

    world_dict = {}
    if include_table:
        world_dict.update(get_table_collision_dict())
    if include_object and obj_filepath is not None:
        world_dict.update(
            get_object_collision_dict(file_path=obj_filepath, xyz=obj_xyz, quat_wxyz=obj_quat_wxyz)
        )
    if len(world_dict) == 0:
        world_dict.update(
            get_dummy_collision_dict()
        )
    world_cfg = WorldConfig.from_dict(world_dict)
    config = RobotWorldConfig.load_from_config(
        robot_cfg, world_cfg, collision_activation_distance=0.0
    )
    curobo_fn = RobotWorld(config)
    d_world, d_self = curobo_fn.get_world_self_collision_distance_from_joints(
        torch.from_numpy(q[None, ...]).float().cuda()
    )
    return (
        d_world.squeeze(dim=0).detach().cpu().numpy(),
        d_self.squeeze(dim=0).detach().cpu().numpy(),
    )


def max_penetration_from_X_W_H(
    X_W_H: np.ndarray,
    q_algr_constraint: np.ndarray,
    include_object: bool = True,
    obj_filepath: Optional[pathlib.Path] = pathlib.Path(
        "/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-05-02_16-19-22/nerf_to_mesh/mug_330/coacd/decomposed.obj"
    ),
    obj_xyz: Tuple[float, float, float] = (0.65, 0.0, 0.0),
    obj_quat_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    include_table: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:

    q_solution = solve_ik(
        X_W_H=X_W_H,
        q_algr_constraint=q_algr_constraint,
        collision_check_object=True,
        obj_filepath=obj_filepath,
        obj_xyz=obj_xyz,
        obj_quat_wxyz=obj_quat_wxyz,
        collision_check_table=True,
        raise_if_no_solution=False,
    )
    return max_penetration_from_q(
        q=q_solution,
        include_object=include_object,
        obj_filepath=obj_filepath,
        obj_xyz=obj_xyz,
        obj_quat_wxyz=obj_quat_wxyz,
        include_table=include_table,
    )


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

    print("=" * 80)
    try:
        q_collide_object = solve_ik(
            X_W_H=X_W_H_collide_object, q_algr_constraint=q_algr_pre, collision_check_object=True
        )
        raise ValueError("Collision check failed to detect collision.")
    except RuntimeError:
        print("Collision check successfully detected collision.")
        max_penetration_collide_object = max_penetration_from_X_W_H(
            X_W_H=X_W_H_collide_object, q_algr_constraint=q_algr_pre, include_object=True
        )
        print(f"max_penetration_collide_object = {max_penetration_collide_object}")
        max_penetration_collide_object_turn_off_object = max_penetration_from_X_W_H(
            X_W_H=X_W_H_collide_object, q_algr_constraint=q_algr_pre, include_object=False
        )
        print(
            f"max_penetration_collide_object_turn_off_object = {max_penetration_collide_object_turn_off_object}"
        )
        max_penetration_collide_object_turn_off_table = max_penetration_from_X_W_H(
            X_W_H=X_W_H_collide_object, q_algr_constraint=q_algr_pre, include_table=False
        )
        print(
            f"max_penetration_collide_object_turn_off_table = {max_penetration_collide_object_turn_off_table}"
        )
    print("=" * 80 + "\n")

    print("=" * 80)
    try:
        q_collide_table = solve_ik(
            X_W_H=X_W_H_collide_table, q_algr_constraint=q_algr_pre, collision_check_object=False
        )
        raise ValueError("Collision check failed to detect collision.")
    except RuntimeError:
        print("Collision check successfully detected collision.")
        max_penetration_collide_table = max_penetration_from_X_W_H(
            X_W_H=X_W_H_collide_table, q_algr_constraint=q_algr_pre, include_table=True
        )
        print(f"max_penetration_collide_table = {max_penetration_collide_table}")
        max_penetration_collide_table_turn_off_table = max_penetration_from_X_W_H(
            X_W_H=X_W_H_collide_table, q_algr_constraint=q_algr_pre, include_table=False
        )
        print(
            f"max_penetration_collide_table_turn_off_table = {max_penetration_collide_table_turn_off_table}"
        )
        max_penetration_collide_table_turn_off_object = max_penetration_from_X_W_H(
            X_W_H=X_W_H_collide_table, q_algr_constraint=q_algr_pre, include_object=False
        )
        print(
            f"max_penetration_collide_table_turn_off_object = {max_penetration_collide_table_turn_off_object}"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
