from curobo.geom.sdf.world import CollisionCheckerType
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
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from scipy.spatial.transform import Rotation


parser = ArgumentParser()
parser.add_argument("--grasp_idx", type=int, default=0)
parser.add_argument("--traj_len", type=int, default=6)
parser.add_argument("--pause", action="store_true", default=False)
args = parser.parse_args()


tensor_args = TensorDeviceType()
world_file = "my_scene.yml"
robot_file = "fr3_algr_zed2i.yml"
motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_file,
    world_file,
    tensor_args,
    trajopt_tsteps=args.traj_len+3,
    collision_checker_type=CollisionCheckerType.MESH,
    use_cuda_graph=True,
)
motion_gen = MotionGen(motion_gen_config)
robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
retract_cfg = motion_gen.get_retract_config()


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
# #Should be forward kinematics
# state = motion_gen.rollout_fn.compute_kinematics(
#     JointState.from_position(retract_cfg.view(1, -1))
# )

pb.connect(pb.GUI)

# r = pb.loadURDF("/juno/u/tylerlum/github_repos/curobo/src/curobo/content/assets/robot/franka_description/franka_panda.urdf", useFixedBase=True, basePosition=[-0.14134081,  0.50142033, -0.15], baseOrientation=[0, 0, -0.3826834, 0.9238795])
r = pb.loadURDF("/juno/u/tylerlum/github_repos/nerf_grasping/nerf_grasping/fr3_algr_ik/allegro_ros2/models/fr3_algr_zed2i.urdf", useFixedBase=True, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1])
num_total_joints = pb.getNumJoints(r)
assert num_total_joints == 39

# obj = pb.loadURDF("/juno/u/tylerlum/github_repos/DexGraspNet/data/rotated_meshdata/core-bottle-109d55a137c042f5760315ac3bf2c13e/coacd/coacd.urdf", useFixedBase=True, basePosition=[1, 1, 1], baseOrientation=[0, 0, 0, 1])
obj = pb.loadURDF("/juno/u/tylerlum/github_repos/pybullet-object-models/pybullet_object_models/ycb_objects/YcbBanana/model.urdf", useFixedBase=True, basePosition=[0.65, 0, 0,], baseOrientation=[0, 0, 0, 1])

joint_names = [pb.getJointInfo(r, i)[1].decode("utf-8") for i in range(num_total_joints) if pb.getJointInfo(r, i)[2] != pb.JOINT_FIXED]
link_names = [pb.getJointInfo(r, i)[12].decode("utf-8") for i in range(num_total_joints) if pb.getJointInfo(r, i)[2] != pb.JOINT_FIXED]
print(f"joint_names = {joint_names}")
print(f"link_names = {link_names}")
actuatable_joint_idxs = [i for i in range(num_total_joints) if pb.getJointInfo(r, i)[2] != pb.JOINT_FIXED]
num_actuatable_joints = len(actuatable_joint_idxs)
assert num_actuatable_joints == 23
arm_actuatable_joint_idxs = actuatable_joint_idxs[:7]
hand_actuatable_joint_idxs = actuatable_joint_idxs[7:]

for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
    pb.resetJointState(r, joint_idx, DEFAULT_Q_FR3[i])

for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
    pb.resetJointState(r, joint_idx, DEFAULT_Q_ALGR[i])

trajs = []
successes = []

X_W_H = np.array(
    [
        # [-0.40069854, 0.06362686, 0.91399777, 0.66515265],
        # [-0.367964, 0.90242159, -0.22413731, 0.02321906],
        # [-0.83907259, -0.4261297, -0.33818674, 0.29229766],
        # [0.0, 0.0, 0.0, 1.0],
        [0, 0, 1, 0.4],
        [0, 1, 0, 0.0],
        [-1, 0, 0, 0.3],
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

trans = X_W_H[:3, 3]
rot_matrix = X_W_H[:3, :3]

import transforms3d
quat_wxyz = transforms3d.quaternions.mat2quat(rot_matrix)

target_pose = Pose(torch.from_numpy(trans).float().cuda(), quaternion=torch.from_numpy(quat_wxyz).float().cuda())
start_state = JointState.from_position(torch.from_numpy(DEFAULT_Q).float().cuda().view(1, -1))
t_start = time.time()
result = motion_gen.plan(
        start_state,
        target_pose,
        enable_graph=True,
        enable_opt=False,
        max_attempts=10,
        num_trajopt_seeds=10,
        num_graph_seeds=10)
breakpoint()
if result is None:
    print("IK Failed!")
    successes.append(False)
    trajs.append(DEFAULT_Q_FR3.reshape(1, -1).repeat(args.traj_len, axis=0))
    raise ValueError()
print("Time taken: ", time.time()-t_start)
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
    assert position.shape == (7,)
    print(f"{t} / {len(traj.position)} {position}")
    #print(position)

    for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
        pb.resetJointState(r, joint_idx, position[i])
    if args.pause: 
        input()
    time.sleep(0.001)
if args.pause:
    input()
breakpoint()
successes.append(True)
trajs.append(np.vstack([traj.position.cpu().numpy(), position.cpu().numpy()]))

trajs = np.stack(trajs)
print("Number of feasible trajectories:", len(trajs))
np.savez("traj.npz", trajs = trajs, successes = successes)
