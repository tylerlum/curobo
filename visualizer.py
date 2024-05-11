import pathlib
from tqdm import tqdm
import numpy as np
from typing import Optional, Tuple, List
import pybullet as pb
from curobo.util_file import (
    get_assets_path,
    get_robot_configs_path,
    join_path,
    load_yaml,
)

import yaml
import time

from dataclasses import dataclass


@dataclass
class Object:
    urdf_path: pathlib.Path
    xyz: Tuple[float, float, float]
    quat_wxyz: Tuple[float, float, float, float]


def start_visualizer(
    robot_file: str = "ur5e.yml",
    objects: Optional[List[Object]] = None,
):
    robot_urdf_path = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"][
        "kinematics"
    ]["urdf_path"]
    robot_urdf_path = pathlib.Path(join_path(get_assets_path(), robot_urdf_path))
    assert robot_urdf_path.exists()

    pb.connect(pb.GUI)
    robot = pb.loadURDF(
        str(robot_urdf_path),
        useFixedBase=True,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1],
    )
    num_total_joints = pb.getNumJoints(robot)

    if objects is not None:
        for object in objects:
            assert object.urdf_path.exists()
            obj_xyz = object.xyz
            obj_quat_xyzw = object.quat_wxyz[1:] + object.quat_wxyz[:1]
            obj = pb.loadURDF(
                str(object.urdf_path),
                useFixedBase=True,
                basePosition=obj_xyz,
                baseOrientation=obj_quat_xyzw,  # Must be xyzw
            )
    return robot


def get_retract_config(robot_file: str = "ur5e.yml") -> np.ndarray:
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    retract_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    return np.array(retract_config)


def set_robot_state(robot, q: np.ndarray) -> None:
    assert len(q.shape) == 1, q.shape

    num_total_joints = pb.getNumJoints(robot)
    actuatable_joint_idxs = [
        i for i in range(num_total_joints) if pb.getJointInfo(robot, i)[2] != pb.JOINT_FIXED
    ]
    num_actuatable_joints = len(actuatable_joint_idxs)
    assert (
        num_actuatable_joints == q.shape[0]
    ), f"Expected {num_actuatable_joints}, got {q.shape[0]}"

    for i, joint_idx in enumerate(actuatable_joint_idxs):
        pb.resetJointState(robot, joint_idx, q[i])


def animate_robot(robot, qs: np.ndarray, dt: float) -> None:
    N_pts, n_joints = qs.shape[:2]
    assert qs.shape == (N_pts, n_joints), qs.shape

    last_update_time = time.time()
    for i in tqdm(range(N_pts)):
        q = qs[i]

        set_robot_state(robot, q)
        time_since_last_update = time.time() - last_update_time
        if time_since_last_update <= dt:
            time.sleep(dt - time_since_last_update)
        last_update_time = time.time()


def create_urdf(obj_path: pathlib.Path) -> pathlib.Path:
    assert obj_path.suffix == ".obj"
    filename = obj_path.name
    parent_folder = obj_path.parent
    urdf_path = parent_folder / f"{obj_path.stem}.urdf"
    urdf_text = f"""<?xml version="0.0" ?>
<robot name="model.urdf">
  <link name="baseLink">
    <contact>
      <lateral_friction value="0.8"/>
      <rolling_friction value="0.001"/>g
      <contact_cfm value="0.0"/>
      <contact_erp value="1.0"/>
    </contact>
    <inertial>
    <origin rpy="0 0 0" xyz="0.01 0.0 0.01"/>
       <mass value=".066"/>
       <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="{filename}" scale="1 1 1"/>
      </geometry>
      <material name="white">
        <color rgba="1. 1. 1. 1."/>
      </material>
    </visual>
    <collision>
      <geometry>
    	 	<mesh filename="{filename}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>"""
    with urdf_path.open("w") as f:
        f.write(urdf_text)
    return urdf_path


def main() -> None:
    pb_robot = start_visualizer()
    retract_q = get_retract_config()
    set_robot_state(pb_robot, retract_q)
    breakpoint()


if __name__ == "__main__":
    main()
