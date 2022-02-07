from typing import Any, Callable, Dict, Generator, Iterable, List, Set, Union
import warnings

from pytorch3d import transforms
import torch

from plb.urdfpy import Robot, Joint, Link

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE)

TIME_INTERVAL = 24


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
    """ Convert a (4, 4) matrix to a 7-dim XYZ-Quat transformation

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

class DiffJoint:
    """ The differentiable wrapper of the Joint

    Params
    ------
    joint: the Joint to be wrapped
    """
    def __init__(self, joint: Joint):
        self._joint = joint
        self.angle = _tensor_creator(torch.zeros, (self._joint.action_dim, ))
        """ The current angle of this joint: shape(joint.action_dim, 1)"""
        self.velocities: List[torch.Tensor] = [
            _tensor_creator(torch.zeros_like, self.angle)
        ]
        """A list of joint's velocities along the time"""
        self.diff_axis = _tensor_creator(torch.tensor, joint.axis)
        """Differentiable axis"""
        self.diff_origin = _tensor_creator(torch.tensor, joint.origin)
        """Differentiable origin"""
    
    def __getattr__(self, name):
        return getattr(self._joint, name)
    
    def apply_velocity(self, vel: torch.Tensor) -> None:
        """ Apply a new velocity

        Params
        ------
        vel: the velocity to be applied, which MUST be of
            the same DoF as the wrapped joint's action_dim
        """
        velShape = vel.shape
        if (not velShape and self._joint.action_dim != 1) \
        or (velShape and velShape[-1] != self._joint.action_dim):
            raise ValueError(f"Joint: {self._joint.name} expects {self._joint.action_dim}"+\
                f"-dim velocity, but got {vel.shape if vel.shape else 1}-d velocity")
        
        self.angle = self.angle + vel * TIME_INTERVAL
        self.velocities.append(vel)

    def get_child_pose_diff(self) -> torch.Tensor:
        """ Differentiable version of `plb.urdfpy.Joint.get_child_pose`

        Return
        ------
        A 4-by-4 transformation matrix to be applied on the joint's 
        child link, given this joint's current angle
        """
        if torch.all(self.angle == 0) or self._joint.joint_type == 'fixed':
            return self.diff_origin
        elif self._joint.joint_type in ['revolute', 'continuous']:
            # angle = cfg[0], cfg.shape == (1,)
            rot3by3 = transforms.axis_angle_to_matrix(self.angle * self.diff_axis)
            return self.diff_origin.mm(_rotation_2_matrix(rot3by3))
        elif self._joint.joint_type == 'prismatic':
            translation = torch.reshape(self.angle * self.diff_axis, (3,1))
            return self.diff_origin.mm(_translation_2_matrix(translation))
        elif self._joint.joint_type == 'planar':
            if self.angle.shape != (2,):
                raise ValueError(
                    '(2,) float configuration required for planar joints'
                )
            translation = torch.reshape(self.diff_origin[:3, :2].mm(self.angle),(3,1))
            return self.diff_origin.mm(_translation_2_matrix(translation))
        elif self._joint.joint_type == 'floating':
            joint_action = _xyz_rpy_2_matrix(self.angle)
            return self.diff_origin.mm(joint_action)
        else:
            raise ValueError('Invalid configuration')

class DiffLink:
    """ The differentiable wrapper of the Link

    Params
    ------
    link: the Link to be wrapped
    pose: the initial pose of the link, which is a 4-by-4 matrix
    """
    def __init__(self, link: Link, pose = None) -> None:
        self._link = link
        if pose is not None: self.init_pose = pose

    @property
    def init_pose(self) -> Union[torch.Tensor, None]:
        if len(self.trajectory) == 0:
            warnings.warn(f"the link: {self._link.name}'s initial pose has not been initialized!")
            return None
        return self.trajectory[0]
    
    @init_pose.setter
    def init_pose(self, pose: Union[torch.Tensor, Iterable]) -> None:
        if not isinstance(pose, torch.autograd.Variable):
            pose = _tensor_creator(torch.tensor, pose)
        if pose.shape == (4,4):
            pose = _matrix_2_xyz_quat(pose)
        if pose.shape != (7,):
            raise ValueError(f"initial pose of a link MUST be a 4-by-4 matrix or a 7-dim vector, got {pose.shape}")
        self.trajectory: List[torch.Tensor] = [pose]
        """A list of link's 7-D, i.e., xyz+quat, position along the time"""
        self.velocities: List[torch.Tensor] = [
            _tensor_creator(torch.zeros_like, self.trajectory[0])
        ]

    
    def __getattr__(self, name):
        return getattr(self._link, name)

    def move_link(self, pose: torch.Tensor) -> torch.Tensor:
        """ Add a new pose to the link's trajectory

        Params
        ------
        pose: a tensor of shape (4,4) --- transformation matrix
            or a tensor of shape (7,) --- XYZ + Quat
        
        Return
        ------
        The velocity necessary to move the link toward the target
        """
        if pose.shape == (4, 4):
            pose = _matrix_2_xyz_quat(pose)
        if pose.shape[-1] != 7:
            raise ValueError(f'pose must either be of shape 4,4 or 7, but got {pose.shape}')
        self.trajectory.append(pose)
        self.velocities.append((self.trajectory[-1] - self.trajectory[-2]) / TIME_INTERVAL)
        return self.velocities[-1]


class DiffRobot(Robot):
    """ The robot with differentiable forward kinematics
    """
    def __init__(self, name, links, joints=None, transmissions=None, materials=None, other_xml=None):
        diffJoints = (DiffJoint(joint) for joint in joints) if joints is not None else None
        diffLinks  = (DiffLink(link) for link in links) if links is not None else None
        super().__init__(name, diffLinks, diffJoints, transmissions, materials, other_xml)
        for diffLink, pose in self.link_fk().items():
            diffLink.init_pose = pose

    def link_fk_diff(
        self,
        jointActions: Union[None, List[torch.Tensor]],
        link_names: List[str] = None
    ) -> Dict[str, torch.Tensor]:
        """ Differetiable version of robot.link_fk

        Params
        ------
        jointActions: It is expected to be a list of 
            torch.Tensor, each of which specifying a
            joint's velocity. The sequence is the same
            as that of self._actuated_joints
        link_names: the concerned links' names, if set
            to NONE, all the links' velocities will be
            returned
        Return
        ------
        A mapping from link names to their poses, each
        being a 7-dim vector
        """
        if link_names == None:
            link_names = self._link_map.keys()
        for idx, joint in enumerate(self._actuated_joints):
            joint.apply_velocity(jointActions[idx])
        fk = {}
        for currentLink in self._reverse_topo:
            pose4by4 = torch.eye(4, dtype=torch.float64, device=DEVICE, requires_grad=True)
            path = self._paths_to_base[currentLink]
            for i in range(len(path) - 1):
                child, parent = path[i], path[i + 1]
                joint = self._G.get_edge_data(child, parent)['joint']
                
                pose4by4 = joint.get_child_pose_diff().mm(pose4by4)

                # Check existing FK to see if we can exit early
                if parent in fk:
                    pose4by4 = fk[parent].mm(pose4by4)
                    break
            fk[currentLink] = pose4by4
        for link, pose4by4 in fk.items():
            link.move_link(pose4by4)
        return {
            linkName: self._link_map[linkName].trajectory[-1]
            for linkName in link_names
        }
    
    
    def fk_gradient(self, timeStep: int, linkGrad: Dict[str, torch.Tensor]) -> Generator[torch.Tensor, None, None]:
        """ Backward the gradients from links' velocity at one moment to
        the actuated joints' velocities, i.e. reverse the FK path

        Params
        ------
        timeStep: determine at which moment the gradient is propagated to
        linkGrad: from link's name to its velocity gradient

        Returns
        -------
        A sequence of gradients of the joints' velocities
        """
        for diffJoint in self._actuated_joints:
            if len(diffJoint.velocities) < timeStep:
                raise ValueError(f"{diffJoint.name} does not have {timeStep} time steps")
            diffJoint.velocities[timeStep].retain_grad()

        for linkName, gradient in linkGrad.items():
            diffLink = self._link_map[linkName]
            if len(diffLink.velocities) < timeStep:
                raise ValueError(f"{diffLink.name} does not have {timeStep} time steps")
            diffLink.velocities[timeStep].backward(gradient, retain_graph=True)
        
        for diffJoint in self._actuated_joints:
            yield diffJoint.velocities[timeStep].grad
