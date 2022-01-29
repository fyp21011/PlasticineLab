import pickle
from typing import List, Dict

import numpy as np
from pytorch3d import transforms
import torch

from plb.urdfpy import *
from plb.urdfpy.diff_fk import (
    TIME_INTERVAL,
    DiffJoint,
    DiffLink,
    _matrix_2_xyz_quat, 
    _rotation_2_matrix, 
    _translation_2_matrix,
    _xyz_rpy_2_matrix,
    _tensor_creator, 
    DEVICE
)

RANDOM_ROUNDS = 10

def test_xyz_rpy_2_matrix():
    for _ in range(RANDOM_ROUNDS):
        xyz_rpy = torch.rand((6,), device=DEVICE, dtype=torch.float64, requires_grad=True)
        matrix  = _xyz_rpy_2_matrix(xyz_rpy)
        matrix_np = xyz_rpy_to_matrix(xyz_rpy.detach().cpu().numpy())
        assert np.all(np.isclose(matrix_np, matrix.detach().cpu().numpy(), rtol=1e-3)), \
            f'{xyz_rpy} -> expecting \n{matrix_np},\ngot{matrix}'
        
        xyz_rpy.retain_grad()
        matrix.backward(torch.ones_like(matrix, device=DEVICE, dtype=torch.float64))
        assert xyz_rpy.grad != None, f"NONE GRAD from {xyz_rpy}"

def test_translation_2_matrix():
    for _ in range(RANDOM_ROUNDS):
        xyz    = torch.rand((3,), device=DEVICE, dtype=torch.float64, requires_grad=True)
        matrix = _translation_2_matrix(xyz)
        assert matrix.shape == (4, 4)
        assert torch.allclose(matrix[:3, 3], xyz, rtol=1e-3), \
            f'expecting \n{xyz},\ngot {matrix[:3, 3]}'
        xyz.retain_grad()
        matrix.backward(torch.ones_like(matrix, device=DEVICE, dtype=torch.float64))
        assert xyz.grad != None, f"NONE GRAD from {xyz}" 
        assert torch.all(xyz.grad == 1), f"incorrect gradient: {xyz.grad}"

def test_rotation_2_matrix():
    for _ in range(RANDOM_ROUNDS):
        rot3by3 = torch.rand((3,3), device=DEVICE, dtype=torch.float64, requires_grad=True)
        matrix = _rotation_2_matrix(rot3by3)
        assert matrix.shape == (4, 4)
        assert torch.allclose(matrix[:3,:3], rot3by3, rtol=1e-3), \
            f'expecting \n{rot3by3},\ngot {matrix[:3,:3]}'
        rot3by3.retain_grad()
        matrix.backward(torch.ones_like(matrix, device=DEVICE, dtype=torch.float64))
        assert rot3by3.grad != None, f"NONE GRAD from {rot3by3}" 
        assert torch.all(rot3by3.grad == 1), f"incorrect gradient: {rot3by3.grad}"
        

def test_matrix_2_xyz_quat():
    for _ in range(RANDOM_ROUNDS):
        rotate3  = torch.rand((3,), device=DEVICE, dtype=torch.float64, requires_grad=True)
        trans3   = torch.rand((3,), device=DEVICE, dtype=torch.float64, requires_grad=True)
        matrix   = _xyz_rpy_2_matrix(torch.cat((trans3, rotate3), dim = 0))
        xyz_quat = _matrix_2_xyz_quat(matrix)
        assert xyz_quat.shape == (7,)
        assert torch.allclose(xyz_quat[:3], trans3, rtol=1e-3), \
            f"XYZ = {trans3} -> matrix = \n{matrix}\n -> XYZ = {xyz_quat[:3]}"
        trueQuat = transforms.matrix_to_quaternion(matrix[:3, :3])
        assert torch.allclose(xyz_quat[3:], trueQuat, rtol=1e-3), \
            f"RPY = {rotate3} -> matrix = \n{matrix}\n -> XYZ = {xyz_quat[3:]}"
        
        rotate3.retain_grad()
        trans3.retain_grad()
        xyz_quat.backward(torch.ones_like(xyz_quat, device=DEVICE, dtype=torch.float64))
        assert rotate3.grad != None, f"NONE GRAD FROM {rotate3}"
        assert trans3.grad != None, f"NONE GRAD FROM {trans3}"
        assert torch.all(trans3.grad == 1), f"incorrect gradient: {trans3.grad}"

robot = Robot.load('tests/data/ur5/ur5.urdf')

def test_diff_joint():
    for idx, jointType in \
    enumerate(['prismatic', 'revolute', 'continuous', 'planar', 'floating']):
        joint = Joint(
            name       = f'test_joint_{idx}', 
            joint_type = jointType, 
            parent     = None, 
            child      = None, 
            axis       = np.random.random((3,)), 
            limit      = JointLimit(1.0, 1.0, -24.0, 24.0)
        )
        diffJoint = DiffJoint(joint)
        assert np.allclose(diffJoint.diff_axis.detach().cpu().numpy(), diffJoint.axis, rtol=1e-3), \
            f"unmatched axes: {diffJoint.diff_axis}, {diffJoint.axis}"
        assert np.allclose(diffJoint.diff_origin.detach().cpu().numpy(), diffJoint.origin, rtol=1e-3), \
            f"unmatched origin: \n{diffJoint.diff_origin}, \n{diffJoint.origin}"

        # test action
        action1 = _tensor_creator(torch.rand, (diffJoint.action_dim, )) * 0.01
        diffJoint.apply_velocity(action1)
        with torch.no_grad():
            assert len(diffJoint.velocities) == 2 and torch.allclose(diffJoint.angle, \
                action1 * TIME_INTERVAL, rtol=1e-3), \
                f"applying {action1} to joint, expecting {action1 * TIME_INTERVAL}, get {diffJoint.angle}"
        action2 = _tensor_creator(torch.rand, (diffJoint.action_dim, )) * 0.01
        diffJoint.apply_velocity(action2)
        with torch.no_grad():
            assert len(diffJoint.velocities) == 3 and torch.allclose(diffJoint.angle, (action1 * TIME_INTERVAL + action2 * TIME_INTERVAL), rtol=1e-3), f"applying {action1}, "+\
                f"{action2} to joint, expecting {action1 * TIME_INTERVAL + action2 * TIME_INTERVAL}, get {diffJoint.angle}"

        # test backward
        action1.retain_grad()
        action2.retain_grad()
        diffJoint.velocities[1].backward(torch.ones_like(diffJoint.velocities[1]), retain_graph=True)
        assert action1.grad is not None, f"NONE action1({action1}).grad"
        diffJoint.velocities[2].backward(torch.ones_like(diffJoint.velocities[2]), retain_graph=True)
        assert action2.grad is not None, f"NONE action2({action2}).grad"
    
def test_diff_link():
    initFk = robot.link_fk()
    for link, pose in initFk.items():
        diffLink = DiffLink(link, pose)
        for _ in range(RANDOM_ROUNDS):
            diffLink.move_link(_tensor_creator(torch.rand, (7,)))
            diffLink.move_link(_tensor_creator(torch.rand, (4,4)))
        assert 2 * RANDOM_ROUNDS + 1 == len(diffLink.trajectory) == len(diffLink.velocities), \
            f"unmatched: {2 * RANDOM_ROUNDS} {len(diffLink.trajectory)} {len(diffLink.velocities)}"

        for i in range(1 + 2 * RANDOM_ROUNDS):
            diffLink.trajectory[i].retain_grad()
        for i in range(2 * RANDOM_ROUNDS, -1, -1):
            diffLink.velocities[i].backward(torch.ones_like(diffLink.velocities[i]), retain_graph=True)
            assert diffLink.trajectory[i].grad != None, \
                f"backward from velocity {i} to pose {i}, {i-1}: pose{i} got NONE grad"
            assert diffLink.trajectory[i - 1].grad != None, \
                f"backward from velocity {i} to pose {i}, {i-1}: pose{i-1} got NONE grad"

def test_diff_robot():
    robotDiff = DiffRobot.load('tests/data/ur5/ur5.urdf')
    robotRaw  = robot.copy()

    # first check the init_pose
    fk = robotRaw.link_fk(use_names = True)
    for linkName in robotRaw.link_map:
        assert linkName in robotDiff._link_map, \
            f"{linkName} NOT FOUND in diff robot"
        diffLink = robotDiff._link_map[linkName]
        assert isinstance(diffLink, DiffLink), \
            f"{linkName} in diff robot is not type DiffLink, but {type(diffLink)}"
        
        translation = fk[linkName][:3, 3].reshape((3,))
        translationDiff = diffLink.init_pose.detach()[:3].cpu().numpy()
        assert np.allclose(translation, translationDiff, rtol=1e-3), \
            f"Init XYZ of {linkName} in the raw robot:{translation} does not match that in the diff robot: {translationDiff}"
        rotation = diffLink.init_pose.detach()[3:]
        rotation = transforms.quaternion_to_matrix(rotation).cpu().numpy()
        assert np.allclose(rotation, fk[linkName][:3,:3], rtol=1e-3), \
            f"Init rotation of {linkName} in the raw robot:\n{fk[linkName][:3,:3]}\ndoes not match " + \
            f"that in the diff robot:\n{rotation}"

def test_diff_fk():
    robotDiff = DiffRobot.load('tests/data/ur5/ur5.urdf')
    # 1) load the action from the tests/data
    with open('tests/data/ur5/action_swap.pkl', 'rb') as istream:
        actions: List[Dict[str, float]] = pickle.load(istream)
        assert len(actions) == 42, "ERROR in loading `tests/data/ur5/action_swap.pkl`"
    
    # 2) Apply the action to robotDiff
    for eachAction in actions:
        joinVelocities = []
        for jointName in robotDiff.actuated_joint_names:
            if jointName in eachAction:
                joinVelocities.append(
                    _tensor_creator(torch.tensor, eachAction[jointName])
                )
            else:
                joinVelocities.append(
                    _tensor_creator(torch.zeros, (robotDiff._joint_map[jointName].action_dim,))
                )
        assert len(joinVelocities) == len(robotDiff._actuated_joints)
        linkVelCnt = sum(1 for _ in robotDiff.link_fk_diff(joinVelocities))
        assert linkVelCnt == len(robotDiff._links), \
            f"expecting {len(robotDiff._links)} link velocities, got {len(linkVelCnt[-1])}"

    # 3) Tries to backpropagte along the time
    for timeStep in range(41, 0, -1):
        linkGrad = {
            linkName: torch.ones((7,), device=DEVICE)
            for linkName in robotDiff._link_map.keys()
        }
        for jointVelGrad in robotDiff.backward(timeStep, linkGrad):
            assert jointVelGrad != None, f"NONE Gradient at time={timeStep}"         
    
def test_diff_fk_random_actions():
    rounds = 10
    diffRobot = DiffRobot.load('tests/data/ur5/ur5.urdf')
    robotRaw  = Robot.load('tests/data/ur5/ur5.urdf')
    jointN = len(diffRobot._actuated_joints)
    actions = [
        [
            0.01 * torch.rand((1,), dtype=torch.float64, device = 'cuda', requires_grad=True)
            for _ in range(jointN)
        ] for timeStep in range(rounds)
    ]
    cfgs = []
    for eachAction in actions:
        cfgs.append(
            {
                jointName: 24 * eachAction[idx].detach().cpu().numpy()
                for idx, jointName in enumerate(diffRobot.actuated_joint_names)
            }
        )
    for action in actions:
        for _ in diffRobot.link_fk_diff(action): pass

    groundTruthPos = None
    for cfg in cfgs:
        groundTruthPos = robotRaw.link_fk(cfg, cfgType=FK_CFG_Type.velocity, use_names=True)
    for diffJoint in diffRobot._actuated_joints:
        angle = diffJoint.angle.detach().cpu().numpy()
        jointInRaw = robotRaw._joint_map[diffJoint.name]
        assert np.allclose(
            robotRaw._current_cfg[jointInRaw],
            angle,
            rtol=1e-3
        ), f"MISSMATCH: {diffJoint.name}'s angle, TORCH is {angle}, while NUMPY gives {robotRaw._current_cfg[jointInRaw]}"
    for diffLink in diffRobot._links:
        testResult = diffLink.trajectory[-1]
        assert diffLink.name in groundTruthPos, f"{diffLink.name} not in robotRaw"
        trueResult = torch.tensor(groundTruthPos[diffLink.name], dtype=torch.float64, device=DEVICE)
        trueResult = _matrix_2_xyz_quat(trueResult)
        assert torch.allclose(testResult, trueResult, rtol=1e-3), \
            f"{diffLink}'s final position is different in diffRobot and robotRaw" +\
            f"\nexpecting {trueResult.cpu().numpy()}" + \
            f"got {testResult.detach().cpu().numpy()}"