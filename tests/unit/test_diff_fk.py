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
