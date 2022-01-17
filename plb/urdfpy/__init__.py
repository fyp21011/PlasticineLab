import os
if 'DISPLAY' not in os.environ or len(os.environ['DISPLAY']) == 0:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

from .base  import URDFType
from .geometry import Box, Cylinder, Sphere, Mesh, Geometry, Collision
from .link import Texture, Material, Visual, Link
from .joint import JointCalibration, JointDynamics, JointLimit, JointMimic, SafetyController, Joint
from .transmission import Transmission, TransmissionJoint, Actuator
from .manipulation import Robot, FK_CFG_Type
from .utils import (rpy_to_matrix, matrix_to_rpy, xyz_rpy_to_matrix,
                    matrix_to_xyz_rpy)

__all__ = [
    'URDFType', 'Box', 'Cylinder', 'Sphere', 'Mesh', 'Geometry',
    'Texture', 'Material', 'Collision', 'Visual',
    'JointCalibration', 'JointDynamics', 'JointLimit', 'JointMimic',
    'SafetyController', 'Actuator', 'TransmissionJoint',
    'Transmission', 'Joint', 'Link', 'Robot',
    'rpy_to_matrix', 'matrix_to_rpy', 'xyz_rpy_to_matrix', 'matrix_to_xyz_rpy',
    '__version__'
]
