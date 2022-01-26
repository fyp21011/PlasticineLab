import numpy as np
from pytorch3d import transforms
import torch

from plb.urdfpy import DiffRobot, xyz_rpy_to_matrix
from plb.urdfpy.diff_fk import (
    _matrix_2_xyz_quat, 
    _rotation_2_matrix, 
    _translation_2_matrix,
    _xyz_rpy_2_matrix
)

RANDOM_ROUNDS = 10

def test_xyz_rpy_2_matrix():
    for _ in range(RANDOM_ROUNDS):
        xyz_rpy = torch.rand((6,), requires_grad=True)
        matrix  = _xyz_rpy_2_matrix(xyz_rpy)
        matrix_np = xyz_rpy_to_matrix(xyz_rpy.detach().numpy())
        assert np.all(np.isclose(matrix_np, matrix.detach().numpy(), rtol=1e-3)), \
            f'{xyz_rpy} -> expecting \n{matrix_np},\ngot{matrix}'
        
        xyz_rpy.retain_grad()
        matrix.backward(torch.ones_like(matrix))
        assert xyz_rpy.grad != None, f"NONE GRAD from {xyz_rpy}"

def test_translation_2_matrix():
    for _ in range(RANDOM_ROUNDS):
        xyz    = torch.rand((3,), requires_grad=True)
        matrix = _translation_2_matrix(xyz)
        assert matrix.shape == (4, 4)
        assert torch.allclose(matrix[:3, 3], xyz, rtol=1e-3), \
            f'expecting \n{xyz},\ngot {matrix[:3, 3]}'
        xyz.retain_grad()
        matrix.backward(torch.ones_like(matrix))
        assert xyz.grad != None, f"NONE GRAD from {xyz}" 
        assert torch.all(xyz.grad == 1), f"incorrect gradient: {xyz.grad}"

def test_rotation_2_matrix():
    for _ in range(RANDOM_ROUNDS):
        rot3by3 = torch.rand((3,3), requires_grad=True)
        matrix = _rotation_2_matrix(rot3by3)
        assert matrix.shape == (4, 4)
        assert torch.allclose(matrix[:3,:3], rot3by3, rtol=1e-3), \
            f'expecting \n{rot3by3},\ngot {matrix[:3,:3]}'
        rot3by3.retain_grad()
        matrix.backward(torch.ones_like(matrix))
        assert rot3by3.grad != None, f"NONE GRAD from {rot3by3}" 
        assert torch.all(rot3by3.grad == 1), f"incorrect gradient: {rot3by3.grad}"
        

def test_matrix_2_xyz_quat():
    for _ in range(RANDOM_ROUNDS):
        rotate3  = torch.rand((3,), requires_grad=True)
        trans3   = torch.rand((3,), requires_grad=True)
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
        xyz_quat.backward(torch.ones_like(xyz_quat))
        assert rotate3.grad != None, f"NONE GRAD FROM {rotate3}"
        assert trans3.grad != None, f"NONE GRAD FROM {trans3}"
        assert torch.all(trans3.grad == 1), f"incorrect gradient: {trans3.grad}"


