import yaml
import numpy as np
import pybullet as pb
def get_link_com_xyz_orn(pb, body_id, link_id):
    # get the world transform (xyz and quaternion) of the Center of Mass of the link
    # We *assume* link CoM transform == link shape transform (the one you use to calculate fluid force on each shape)
    assert link_id >= -1
    if link_id == -1:
        link_com, link_quat = pb.getBasePositionAndOrientation(body_id)
    else:
        link_com, link_quat, *_ = pb.getLinkState(body_id, link_id, computeForwardKinematics=1)
    return list(link_com), list(link_quat)


def create_primitive_shape(pb, mass, shape, dim, color=(0.6, 0, 0, 1), 
                           collidable=True, init_xyz=(0, 0, 0),
                           init_quat=(0, 0, 0, 1)):
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

    sid = pb.createMultiBody(baseMass=mass, baseInertialFramePosition=[0, 0, 0],
                             baseCollisionShapeIndex=collision_shape_id,
                             baseVisualShapeIndex=visual_shape_id,
                             basePosition=init_xyz, baseOrientation=init_quat)
    return sid


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

c = pb.connect(pb.GUI)

configs = yaml.safe_load(open("/juno/u/tylerlum/github_repos/curobo/src/curobo/content/configs/robot/spheres/fr3_algr_zed2i.yml","r"))

robot = pb.loadURDF("/juno/u/tylerlum/github_repos/nerf_grasping/nerf_grasping/fr3_algr_ik/allegro_ros2/models/fr3_algr_zed2i.urdf", useFixedBase=True)

num_total_joints = pb.getNumJoints(robot)
assert num_total_joints == 39
actuatable_joint_idxs = [i for i in range(num_total_joints) if pb.getJointInfo(robot, i)[2] != pb.JOINT_FIXED]
num_actuatable_joints = len(actuatable_joint_idxs)
assert num_actuatable_joints == 23
arm_actuatable_joint_idxs = actuatable_joint_idxs[:7]
hand_actuatable_joint_idxs = actuatable_joint_idxs[7:]

for i, joint_idx in enumerate(arm_actuatable_joint_idxs):
    pb.resetJointState(robot, joint_idx, DEFAULT_Q_FR3[i])

for i, joint_idx in enumerate(hand_actuatable_joint_idxs):
    pb.resetJointState(robot, joint_idx, DEFAULT_Q_ALGR[i])

link_names = {"world": -1}
for i in range(pb.getNumJoints(robot)):
    link_names[pb.getJointInfo(robot, i)[12].decode("utf-8")] = i

color_codes = [[1,0,0,0.7],[0,1,0,0.7]]

for i, link in enumerate(configs["collision_spheres"].keys()):
    if link not in link_names:
        continue

    link_id = link_names[link]
    link_pos, link_ori = get_link_com_xyz_orn(pb, robot, link_id)
    for sphere in configs["collision_spheres"][link]:
        s = create_primitive_shape(pb, 0.0, shape=pb.GEOM_SPHERE, dim=(sphere["radius"],), collidable=False, color=color_codes[i%2])
        # Place the sphere relative to the link
        world_coord = list(pb.multiplyTransforms(link_pos, link_ori, sphere["center"], [0,0,0,1])[0])
        world_coord[1] += 0.
        pb.resetBasePositionAndOrientation(s, world_coord, [0,0,0,1])


input()

