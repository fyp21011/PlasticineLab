import os
if 'DISPLAY' not in os.environ or len(os.environ['DISPLAY']) == 0:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

from urdfpy.urdf import (URDFType,
                   Box, Cylinder, Sphere, Mesh, Geometry,
                   Texture, Material,
                   Collision, Visual,
                   JointCalibration, JointDynamics, JointLimit, JointMimic,
                   SafetyController, Actuator, TransmissionJoint,
                   Transmission, Joint, Link, Robot)
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
