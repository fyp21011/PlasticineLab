from typing import Any, Callable, Dict, List, Union
import warnings

from pytorch3d import transforms
import torch

from plb.urdfpy import Robot, Joint, Link

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE)


def _tensor_creator(creator: Callable[[Any], torch.Tensor], *value, **kwargs) -> torch.Tensor:
    """ Create tensor with argument: 
        dtype = torch.float64
        device = DEVICE
        requires_grad = True

    Params
    ------
    creator: the method to create a tensor, such as torch.tensor, torch.zeros,
        torch.zeros_like, torch.ones, torch.ones_like, torch.eye, etc.
    value: the arguments to the creator, except for the dtype, device and requires_grad
    """
    kwargs['device'] = DEVICE
    kwargs['dtype'] = torch.float64
    kwargs['requires_grad'] = True
    return creator(*value, **kwargs)

def _matrix_2_xyz_quat(matrix: torch.Tensor) -> torch.Tensor:
    """ Convert a (4, 4) matrix to a 6-dim XYZ-RPY transformation

    Params
    ------
    matrix: a torch.Tensor of shape (4,4)

    Return
    ------
    a torch.Tensor of shape (7,)
    """
    eularAngle  = transforms.matrix_to_quaternion(matrix[:3, :3]) # shape(4,)
    translation = matrix[:3, 3].reshape((3,)) # shape(3,)
    return torch.cat((translation, eularAngle), dim = 0)

def _xyz_rpy_2_matrix(xyzrpy: torch.Tensor) -> torch.Tensor:
    """ Convert a 6-dim XYZ-RPY transformation to a (4, 4) matrix

    Params
    ------
    xyzrpy: a 6-dim torch.Tensor, consisting of [X-trans, Y-trans,
        Z-trans, Roll (X-rot), Pitch (Y-rot), Yaw (Z-rot)]

    Return
    ------
    The corresponding 4-by-4 matrix corresponding to the
        xyzrpy input. 
    """
    translation = xyzrpy[:3].reshape((3,1)) # X-Y-Z (3,1)
    eularAngle  = xyzrpy[3:] # R-P-Y (3,)

    c3, c2, c1 = torch.cos(eularAngle)
    s3, s2, s1 = torch.sin(eularAngle)

    rot3by3 = torch.stack((
        c1 * c2, (c1 * s2 * s3) - (c3 * s1), (s1 * s3) + (c1 * c3 * s2),
        c2 * s1, (c1 * c3) + (s1 * s2 * s3), (c3 * s1 * s2) - (c1 * s3),
        -s2,     c2 * s3,                    c2 * c3
    )).reshape((3,-1))

    return torch.cat((
        torch.cat((rot3by3, translation), dim = 1),
        _tensor_creator(torch.tensor, [[0, 0, 0, 1]])
    ), dim = 0)

def _translation_2_matrix(translation: torch.Tensor) -> torch.Tensor:
    """ Convert a translation vector to transform matrix

    Params
    ------
    translation: a Tensor of shape (3,), i.e.
        [X-trans, Y-trans, Z-trans], respectively

    Return
    ------
    The (4, 4) transform matrix, in the following form:
    ```
        0 0 0 X-trans
        0 0 0 Y-trans
        0 0 0 Z-trans
        0 0 0    1
    ```
    """
    return torch.cat((
        torch.cat((_tensor_creator(torch.eye, 3), translation.reshape((3, -1))), dim = 1),
        _tensor_creator(torch.tensor, [[0, 0, 0, 1]])
    ), dim = 0)

def _rotation_2_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """ Convert a rotation matrix to transform matrix

    Params
    ------
    rotation: a Tensor of shape (3, 3)

    Return
    ------
    The (4, 4) transform matrix, in the following form:
    ```
          3 by 3  | 0
         Rotation | 0
          Matrix  | 0
        [ 0  0  0 ] 1
    ```
    """
    return torch.cat((
        torch.cat((rotation, _tensor_creator(torch.zeros, (3,1))), dim=1),
        _tensor_creator(torch.tensor, [[0, 0, 0, 1]])
    ), dim = 0)

class DiffRobot(Robot):
    """ The robot with differentiable forward kinematics
    """
    def __init__(self, name, links, joints=None, transmissions=None, materials=None, other_xml=None):
        super().__init__(name, links, joints, transmissions, materials, other_xml)

        if self._joint_map is None or len(self._joint_map) == 0:
            warnings.warn("try to init torch on empty robot. SKIP")
            return
        
        self.actuated_joint_2_idx: Dict[Joint, int] = {}
        """ A map from Joint to its index in the self._actuated_joints """
        self.joint_pos: List[torch.Tensor] = []
        """ The differentiable joints' origins """
        self.joint_axis: List[torch.Tensor] = []
        """ the differentiable joints' axes """
        # static origins and axes to differentiable variables
        for idx, joint in enumerate(self._actuated_joints): 
            self.joint_pos.append(
                torch.tensor(joint.origin, dtype=torch.float64, device=DEVICE, requires_grad=True)
            )
            self.joint_axis.append(
                torch.tensor(joint.axis, dtype=torch.float64, device=DEVICE, requires_grad=True)
            )
            self.actuated_joint_2_idx[joint] = idx

        self.link_2_idx = {link: idx for idx, link in enumerate(self._links)}
        """ A map from Link to its index in self._links """
        _initPoses = sorted([(self.link_2_idx[link], pose4by4) for link, pose4by4 in self.link_fk().items()])
        self.link_pos: List[List[torch.Tensor]] = [
            list(
                _matrix_2_xyz_quat(torch.tensor(eachPose, device=DEVICE, requires_grad=True) )
                for _, eachPose in _initPoses
            ) # time 0 poses
        ]
        """ Shape of [TIMESTEP, LINK_NUMBER, (7, )]"""


    def link_fk_diff(self, jointActions: Union[None, List[torch.Tensor]]) -> torch.Tensor:
        """ Differetiable version of robot.link_fk
        """
        explored_link_2_id: Dict[Link, int] = {}
        for lnk in self._reverse_topo:
            pose = torch.eye(4, dtype=torch.float64) # constant, no grad needed
            path = self._paths_to_base[lnk]
            for i in range(len(path) - 1):
                child, parent = path[i], path[i + 1]
                joint = self._G.get_edge_data(child, parent)['joint']
                jointIdx = self.actuated_joint_2_idx[joint]

                if joint.mimic is not None:
                    mimic_joint = self._joint_map[joint.mimic.joint]
                    if mimic_joint in self.actuated_joint_2_idx[mimic_joint]:
                        jointVel = jointActions[self.actuated_joint_2_idx[mimic_joint]]
                        jointVel = joint.mimic.multiplier * jointVel + joint.mimic.offset
                elif joint in self.actuated_joint_2_idx and self._current_cfg[joint] != 0:
                    jointVel = jointActions[jointIdx]
                else:
                    jointVel = None
                
                pose = self.get_child_pose(
                    joint_type   = joint.joint_type,
                    joint_pos    = self.joint_pos[jointIdx], 
                    joint_axis   = self.joint_axis[jointIdx], 
                    joint_action = jointVel
                ).dot(pose)

                self.joint_pos[jointIdx] = self.joint_pos[jointIdx] + jointVel

                # Check existing FK to see if we can exit early
                # TODO: store the pose (converted to the xyz+quat) into the torch
                if parent in explored_link_2_id:
                    pose = self.link_pos[explored_link_2_id[parent]].dot(pose)
                    break
            linkIdx = self.link_2_idx[lnk]
    
    def backward(self, jointAction: torch.Tensor, linkGrad: torch.Tensor):
        """
        TODO
        """
        jointAction.retain_grad()
        self.link_vel.backward(linkGrad, retain_graph=True)
        return jointAction.grad

    @staticmethod
    def get_child_pose(
            joint_type:   str,
            joint_pos:    torch.Tensor,
            joint_axis:   torch.Tensor,
            joint_action: torch.Tensor
        ) -> torch.Tensor:
        """ Differentiable version of `plb.urdfpy.Joint.get_child_pose`

        Params
        ------
        joint_type: returned value of the joint.type
        joint_pos:  a 4-by-4 tensor, describing the joint's position,
            initialized from joint.origin
        joint_axis: a tensor of shape (3,), initialized from joint.axis
        joint_action: the action to be applied on the joint:

            * (1,) for revolute, continuous or prismatic joint
            * (2,) for planar joint
            * (6,) for floating joint

        Return
        ------
        A 4-by-4 transformation matrix to be applied on the joint's 
        child link
        """
        if joint_action == None or torch.all(joint_action == 0) or joint_type == 'fixed':
            return joint_pos
        elif joint_type in ['revolute', 'continuous']:
            # angle = cfg[0], cfg.shape == (1,)
            rot3by3 = transforms.axis_angle_to_matrix(joint_action * joint_axis)
            return joint_pos.dot(_rotation_2_matrix(rot3by3))
        elif joint_type == 'prismatic':
            translation = torch.reshape(joint_action * joint_axis, (3,1))
            return joint_pos.dot(_translation_2_matrix(translation))
        elif joint_type == 'planar':
            if joint_action.shape != (2,):
                raise ValueError(
                    '(2,) float configuration required for planar joints'
                )
            translation = torch.reshape(joint_pos[:3, :2].dot(joint_action),(3,1))
            return joint_pos.dot(_translation_2_matrix(translation))
        elif joint_type == 'floating':
            joint_action = _xyz_rpy_2_matrix(joint_action)
            return joint_pos.dot(joint_action)
        else:
            raise ValueError('Invalid configuration')
