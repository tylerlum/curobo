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


def get_link_com_xyz_orn(pb, body_id, link_id):
    # get the world transform (xyz and quaternion) of the Center of Mass of the link
    # We *assume* link CoM transform == link shape transform (the one you use to calculate fluid force on each shape)
    assert link_id >= -1
    if link_id == -1:
        link_com, link_quat = pb.getBasePositionAndOrientation(body_id)
    else:
        link_com, link_quat, *_ = pb.getLinkState(body_id, link_id, computeForwardKinematics=1)
    return list(link_com), list(link_quat)


def create_primitive_shape(
    pb,
    mass,
    shape,
    dim,
    color=(0.6, 0, 0, 1),
    collidable=True,
    init_xyz=(0, 0, 0),
    init_quat=(0, 0, 0, 1),
):
    # shape: p.GEOM_SPHERE or p.GEOM_BOX or p.GEOM_CYLINDER
    # dim: halfExtents (vec3) for box, (radius, length)vec2 for cylinder, (radius) for sphere
    # init_xyz vec3 being initial obj location, init_quat being initial obj orientation
    visual_shape_id = None
    collision_shape_id = -1
    if shape == pb.GEOM_BOX:
        visual_shape_id = pb.createVisualShape(shapeType=shape, halfExtents=dim, rgbaColor=color)
        if collidable:
            collision_shape_id = pb.createCollisionShape(shapeType=shape, halfExtents=dim)
    elif shape == pb.GEOM_CYLINDER:
        visual_shape_id = pb.createVisualShape(shape, dim[0], [1, 1, 1], dim[1], rgbaColor=color)
        if collidable:
            collision_shape_id = pb.createCollisionShape(shape, dim[0], [1, 1, 1], dim[1])
    elif shape == pb.GEOM_SPHERE:
        visual_shape_id = pb.createVisualShape(shape, radius=dim[0], rgbaColor=color)
        if collidable:
            collision_shape_id = pb.createCollisionShape(shape, radius=dim[0])

    sid = pb.createMultiBody(
        baseMass=mass,
        baseInertialFramePosition=[0, 0, 0],
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=init_xyz,
        baseOrientation=init_quat,
    )
    return sid


def draw_collision_spheres(robot, config):
    from collections import defaultdict

    link_names = {"world": -1}
    for i in range(pb.getNumJoints(robot)):
        link_names[pb.getJointInfo(robot, i)[12].decode("utf-8")] = i

    color_codes = [[1, 0, 0, 0.7], [0, 1, 0, 0.7]]

    if not hasattr(draw_collision_spheres, "cached_spheres"):
        draw_collision_spheres.cached_spheres = defaultdict(list)
        for i, link in enumerate(config["collision_spheres"].keys()):
            if link not in link_names:
                continue

            link_id = link_names[link]
            link_pos, link_ori = get_link_com_xyz_orn(pb, robot, link_id)
            for sphere in config["collision_spheres"][link]:
                s = create_primitive_shape(
                    pb,
                    0.0,
                    shape=pb.GEOM_SPHERE,
                    dim=(sphere["radius"],),
                    collidable=False,
                    color=color_codes[i % 2],
                )
                # Place the sphere relative to the link
                world_coord = list(
                    pb.multiplyTransforms(link_pos, link_ori, sphere["center"], [0, 0, 0, 1])[0]
                )
                world_coord[1] += 0.0
                pb.resetBasePositionAndOrientation(s, world_coord, [0, 0, 0, 1])
                draw_collision_spheres.cached_spheres[link].append(s)
    else:
        cached_spheres = draw_collision_spheres.cached_spheres
        for i, link in enumerate(config["collision_spheres"].keys()):
            if link not in link_names:
                continue

            link_id = link_names[link]
            link_pos, link_ori = get_link_com_xyz_orn(pb, robot, link_id)
            for j, sphere in enumerate(config["collision_spheres"][link]):
                s = cached_spheres[link][j]

                # Place the sphere relative to the link
                world_coord = list(
                    pb.multiplyTransforms(link_pos, link_ori, sphere["center"], [0, 0, 0, 1])[0]
                )
                world_coord[1] += 0.0
                pb.resetBasePositionAndOrientation(s, world_coord, [0, 0, 0, 1])

def remove_collision_spheres():
    if hasattr(draw_collision_spheres, "cached_spheres"):
        for link, spheres in draw_collision_spheres.cached_spheres.items():
            for s in spheres:
                pb.resetBasePositionAndOrientation(s, [100, 0, 0], [0, 0, 0, 1])


DEFAULT_Q_FR3 = np.array(
    [
        1.76261055e-06,
        -1.29018439e00,
        0.00000000e00,
        -2.69272642e00,
        0.00000000e00,
        1.35254201e00,
        7.85400000e-01,
    ]
)
DEFAULT_Q_ALGR = np.array(
    [
        2.90945620e-01,
        7.37109400e-01,
        5.10859200e-01,
        1.22637060e-01,
        1.20125350e-01,
        5.84513500e-01,
        3.43829930e-01,
        6.05035000e-01,
        -2.68431900e-01,
        8.78457900e-01,
        8.49713500e-01,
        8.97218400e-01,
        1.33282830e00,
        3.47787830e-01,
        2.09215670e-01,
        -6.50969000e-03,
    ]
)
DEFAULT_Q = np.concatenate([DEFAULT_Q_FR3, DEFAULT_Q_ALGR])

grasp_config_dict = np.load(
    "/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-05-02_16-19-22/optimized_grasp_config_dicts/mug_330_0_9999.npy",
    allow_pickle=True,
).item()
breakpoint()
BEST_IDX = 4
GOOD_IDX = 0
GOOD_IDX_2 = 1

trans = grasp_config_dict["trans"][GOOD_IDX]
rot = grasp_config_dict["rot"][GOOD_IDX]
joint_angles = grasp_config_dict["joint_angles"][GOOD_IDX]
X_Oy_H = np.eye(4)
X_Oy_H[:3, :3] = rot
X_Oy_H[:3, 3] = trans

# DEFAULT_Q_ALGR = joint_angles

X_W_N = trimesh.transformations.translation_matrix([0.65, 0, 0])
X_O_Oy = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
obj_centroid = trimesh.load(
    "/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-05-02_16-19-22/nerf_to_mesh/mug_330/coacd/decomposed.obj"
).centroid
print(f"obj_centroid = {obj_centroid}")
X_N_O = trimesh.transformations.translation_matrix(obj_centroid)
X_W_Oy = X_W_N @ X_N_O @ X_O_Oy

parser = ArgumentParser()
parser.add_argument("--grasp_idx", type=int, default=0)
parser.add_argument("--traj_len", type=int, default=6)
parser.add_argument("--pause", action="store_true", default=False)
args = parser.parse_args()


tensor_args = TensorDeviceType()
world_file = "TYLER_scene.yml"
robot_file = "fr3_algr_zed2i.yml"
robot_cfg_dict = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
robot_cfg = RobotConfig.from_dict(robot_cfg_dict, tensor_args)
robot_cfg.kinematics.kinematics_config.joint_limits.position[0, 7:] = (
    torch.from_numpy(DEFAULT_Q_ALGR).float().cuda() - 0.01
)
robot_cfg.kinematics.kinematics_config.joint_limits.position[1, 7:] = (
    torch.from_numpy(DEFAULT_Q_ALGR).float().cuda() + 0.01
)
motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_cfg,
    world_file,
    tensor_args,
    trajopt_tsteps=args.traj_len + 3,
    collision_checker_type=CollisionCheckerType.MESH,
    use_cuda_graph=True,
)
motion_gen = MotionGen(motion_gen_config)
robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
retract_cfg = motion_gen.get_retract_config()


# #Should be forward kinematics
# state = motion_gen.rollout_fn.compute_kinematics(
#     JointState.from_position(retract_cfg.view(1, -1))
# )

pb.connect(pb.GUI)

# r = pb.loadURDF("/juno/u/tylerlum/github_repos/curobo/src/curobo/content/assets/robot/franka_description/franka_panda.urdf", useFixedBase=True, basePosition=[-0.14134081,  0.50142033, -0.15], baseOrientation=[0, 0, -0.3826834, 0.9238795])
r = pb.loadURDF(
    "/juno/u/tylerlum/github_repos/nerf_grasping/nerf_grasping/fr3_algr_ik/allegro_ros2/models/fr3_algr_zed2i.urdf",
    useFixedBase=True,
    basePosition=[0, 0, 0],
    baseOrientation=[0, 0, 0, 1],
)
num_total_joints = pb.getNumJoints(r)
assert num_total_joints == 39

# obj = pb.loadURDF("/juno/u/tylerlum/github_repos/DexGraspNet/data/rotated_meshdata/core-bottle-109d55a137c042f5760315ac3bf2c13e/coacd/coacd.urdf", useFixedBase=True, basePosition=[1, 1, 1], baseOrientation=[0, 0, 0, 1])
# obj = pb.loadURDF("/juno/u/tylerlum/github_repos/pybullet-object-models/pybullet_object_models/ycb_objects/YcbBanana/model.urdf", useFixedBase=True, basePosition=[0.65, 0, 0,], baseOrientation=[0, 0, 0, 1])
obj = pb.loadURDF(
    "/juno/u/tylerlum/github_repos/nerf_grasping/experiments/2024-05-02_16-19-22/nerf_to_mesh/mug_330/coacd/coacd.urdf",
    useFixedBase=True,
    basePosition=[
        0.65,
        0,
        0,
    ],
    baseOrientation=[0, 0, 0, 1],
)

joint_names = [
    pb.getJointInfo(r, i)[1].decode("utf-8")
    for i in range(num_total_joints)
    if pb.getJointInfo(r, i)[2] != pb.JOINT_FIXED
]
link_names = [
    pb.getJointInfo(r, i)[12].decode("utf-8")
    for i in range(num_total_joints)
    if pb.getJointInfo(r, i)[2] != pb.JOINT_FIXED
]
print(f"joint_names = {joint_names}")
print(f"link_names = {link_names}")
actuatable_joint_idxs = [
    i for i in range(num_total_joints) if pb.getJointInfo(r, i)[2] != pb.JOINT_FIXED
]
num_actuatable_joints = len(actuatable_joint_idxs)
assert num_actuatable_joints == 23
arm_actuatable_joint_idxs = actuatable_joint_idxs[:7]
hand_actuatable_joint_idxs = actuatable_joint_idxs[7:]

for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
    pb.resetJointState(r, joint_idx, DEFAULT_Q_FR3[i])

for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
    pb.resetJointState(r, joint_idx, DEFAULT_Q_ALGR[i])

collision_config = yaml.safe_load(
    open(
        "/juno/u/tylerlum/github_repos/curobo/src/curobo/content/configs/robot/spheres/fr3_algr_zed2i.yml",
        "r",
    )
)
draw_collision_spheres(
    robot=r,
    config=collision_config,
)
trajs = []
successes = []

# X_W_H = np.array(
#     [
#         # [-0.40069854, 0.06362686, 0.91399777, 0.66515265],
#         # [-0.367964, 0.90242159, -0.22413731, 0.02321906],
#         # [-0.83907259, -0.4261297, -0.33818674, 0.29229766],
#         # [0.0, 0.0, 0.0, 1.0],
#         [0, 0, 1, 0.4],
#         [0, 1, 0, 0.0],
#         [-1, 0, 0, 0.15],
#         [0.0, 0.0, 0.0, 1.0],
#     ]
# )
# q_algr_pre = np.array(
#     [
#         0.29094562,
#         0.7371094,
#         0.5108592,
#         0.12263706,
#         0.12012535,
#         0.5845135,
#         0.34382993,
#         0.605035,
#         -0.2684319,
#         0.8784579,
#         0.8497135,
#         0.8972184,
#         1.3328283,
#         0.34778783,
#         0.20921567,
#         -0.00650969,
#     ]
# )

X_W_H = X_W_Oy @ X_Oy_H
q_algr_pre = DEFAULT_Q_ALGR

trans = X_W_H[:3, 3]
rot_matrix = X_W_H[:3, :3]

import transforms3d

quat_wxyz = transforms3d.quaternions.mat2quat(rot_matrix)

target_pose = Pose(
    torch.from_numpy(trans).float().cuda(), quaternion=torch.from_numpy(quat_wxyz).float().cuda()
)
start_state = JointState.from_position(torch.from_numpy(DEFAULT_Q).float().cuda().view(1, -1))
t_start = time.time()
result = motion_gen.plan_single(
    start_state=start_state,
    goal_pose=target_pose,
    plan_config=MotionGenPlanConfig(
        enable_graph=True,
        enable_opt=False,
        max_attempts=10,
        num_trajopt_seeds=10,
        num_graph_seeds=10,
    ),
)
breakpoint()
remove_collision_spheres()
print(result)
if result is None:
    print("IK Failed!")
    successes.append(False)
    trajs.append(DEFAULT_Q_FR3.reshape(1, -1).repeat(args.traj_len, axis=0))
    raise ValueError()
print("Time taken: ", time.time() - t_start)
print("Trajectory Generated: ", result.success)
if not result.success:
    print(f"Traj failed")
    successes.append(False)
    trajs.append(DEFAULT_Q_FR3.reshape(1, -1).repeat(args.traj_len, axis=0))
    raise ValueError()
traj = result.get_interpolated_plan()

if args.pause:
    input()
traj = result.interpolated_plan

for t in range(len(traj.position)):
    position = traj.position[t].cpu().numpy()
    assert position.shape == (DEFAULT_Q.shape[0],)
    print(f"{t} / {len(traj.position)} {position}")
    # print(position)

    for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
        pb.resetJointState(r, joint_idx, position[i])
    for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
        pb.resetJointState(r, joint_idx, position[i + 7])
    if args.pause:
        input()
    time.sleep(0.001)

draw_collision_spheres(
    robot=r,
    config=collision_config,
)

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

tensor_args = TensorDeviceType()
robot_cfg = RobotConfig.from_dict(
    load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
)

robot_cfg.kinematics.kinematics_config.joint_limits.position[0, 7:] = (
    torch.from_numpy(DEFAULT_Q_ALGR).float().cuda() - 0.01
)
robot_cfg.kinematics.kinematics_config.joint_limits.position[1, 7:] = (
    torch.from_numpy(DEFAULT_Q_ALGR).float().cuda() + 0.01
)
world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), "TYLER_scene.yml")))
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
breakpoint()
q_solution = result.solution[result.success]
print(f"q_solution = {q_solution}")
for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
    pb.resetJointState(r, joint_idx, q_solution[0, i])
for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
    pb.resetJointState(r, joint_idx, q_solution[0, i + 7])

from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

config = RobotWorldConfig.load_from_config(
    robot_cfg, "TYLER_scene.yml", collision_activation_distance=0.0
)
curobo_fn = RobotWorld(config)
d_world, d_self = curobo_fn.get_world_self_collision_distance_from_joints(q_solution)
state = curobo_fn.get_kinematics(q_solution)
print(f"d_world = {d_world}")
print(f"d_self = {d_self}")
breakpoint()

config = RobotWorldConfig.load_from_config(
    robot_cfg, "TYLER_scene_with_object.yml", collision_activation_distance=0.0
)
curobo_fn = RobotWorld(config)
d_world, d_self = curobo_fn.get_world_self_collision_distance_from_joints(q_solution)
state = curobo_fn.get_kinematics(q_solution)
print(f"d_world = {d_world}")
print(f"d_self = {d_self}")
breakpoint()
# print(f"state = {state}")

if args.pause:
    input()
breakpoint()
successes.append(True)
trajs.append(np.vstack([traj.position.cpu().numpy(), position.cpu().numpy()]))

trajs = np.stack(trajs)
print("Number of feasible trajectories:", len(trajs))
np.savez("traj.npz", trajs=trajs, successes=successes)
