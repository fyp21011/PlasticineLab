from typing import List, Dict, Iterable, Union, Generator
import warnings
import yaml

import numpy as np
import torch
from yacs.config import CfgNode as CN

from plb.config.utils import make_cls_config
from .controller import Controller
from plb.urdfpy import DiffRobot, Robot, Collision, DEVICE
from plb.engine.primitive.primitive import Box, Sphere, Cylinder, Primitive
from .primitive_controller import PrimitivesController

ROBOT_LINK_DOF = 7
ROBOT_LINK_DOF_SCALE = tuple((0.01 for _ in range(ROBOT_LINK_DOF)))
ROBOT_COLLISION_COLOR = '(0.8, 0.8, 0.8)'

def _generate_primitive_config(rawPose: torch.Tensor, offset:Iterable[float], shapeName: str, **kwargs) -> CN:
    """ Generate a CfgNode for primitive mapping

    Based on the URDF's primtives, generate the configuration for PLB's
    primitives, to establish the mapping in between

    Params
    ------
    rawPose: a 7-dim tensor specifying the 
    offset: the offset of the robot to which the primitve belongs
    shapeName: 'Sphere', 'Cylinder' or 'Box'
    **kwargs: other primitive-type specific parameters, such as
        the `size` description for `Box`, `r` and `h` for cylinders
    
    Return
    ------
    A CfgNode for the PLB's primitive instantiation
    """
    if rawPose.shape != (7,):
        raise ValueError(f"expecting a 7-dim Tensor as the initial position, got {rawPose}")
    actionCN = CN(init_dict={'dim': ROBOT_LINK_DOF, 'scale': f'{ROBOT_LINK_DOF_SCALE}'})
    configDict = {
        'action': actionCN, 
        'color':  ROBOT_COLLISION_COLOR, 
        'init_pos': f'({rawPose[0] + offset[0]}, {rawPose[1] + offset[1]}, {rawPose[2] + offset[2]})',
        'init_rot': f'({rawPose[3]}, {rawPose[4]}, {rawPose[5]}, {rawPose[6]})',
        'shape': shapeName,
    }
    for key, value in kwargs.items():
        if isinstance(value, CN): configDict[key] = value
        else:                     configDict[key] = str(value)
    return CN(init_dict=configDict)

class RobotsController(Controller):
    """ A controller that maps robot to primitive links
    """
    def __init__(self) -> None:
        super().__init__()
        self.robots: List[DiffRobot] = []
        self.robot_action_dims: List[int] = []
        self.link_2_primitives: List[Dict[str, Primitive]] = []
        """ `link_2_primitives[RobotIdx][LinkName]` returns the primivite
        corresponds to the collision geometry of the link named as `LinkName`
        in `RobotIdx`'s robot
        """
        self.current_step = 0
        """pointer to the current step

        Example
        -------
        `self.current_step = 3` ==> 
        * the first tree, i.e., `self.flatten_actions[:3]` are executed
        * `self.flatten_actions[3]` are the one to be executed next
        """

    @property
    def not_empty(self) -> bool:
        return len(self.robots) > 0

    @classmethod
    def parse_config(cls, cfgs: List[Union[CN, str]], primitive_controller: PrimitivesController) -> "RobotsController":
        """ Parse the YAML configuration node for `Robots`
        
        Load the robots from the URDF files specified by the `Robots` config node
        and insert the collision shapes into the primitive_controller as primitive
        shapes. 

        Params
        ------
        cfgs: the YAML list titled `Robots` in the env configuration file
        primitive_controller: the Primitives from TaichiEnv, which is pared
            from the `Primitives` list in the env configuration file
        """
        outs = []
        for eachCfg in cfgs:
            if isinstance(eachCfg, CN):
                cfg = eachCfg
            else:
                cfg = CN(new_allowed=True)
                cfg = cfg._load_cfg_from_yaml_str(yaml.safe_dump(eachCfg))
            outs.append(make_cls_config(cls, cfg))
        rc = RobotsController()
        for eachOutCfg in outs:
            diffRobot = DiffRobot.load(eachOutCfg.path)
            for shape in rc.append_robot(diffRobot, eachOutCfg.offset):
                primitive_controller.primitives.append(shape)
        return rc

    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.shape = 'Robot'
        cfg.path = ''
        cfg.offset = (0.0, 0.0, 0.0)
        return cfg

    @classmethod
    def _urdf_collision_to_primitive(cls, collision: Collision, offset: np.ndarray, pose: torch.Tensor) -> Union[Primitive, None]:
        """ Converting the URDF's collision geometry to primitive

        Params
        ------
        collision: the URDF robot's collision geometry
        offset: the offset of the robot's position
        pose: the pose matrix (4 * 4) of this geometry, which
            records the initial position of the geometry
        """
        if collision.geometry.box is not None:
            linkPrimitive = Box(cfg = _generate_primitive_config(
                rawPose   = pose, 
                offset    = offset,
                shapeName = 'Box',
                size      = tuple(collision.geometry.box.size)
            ))
        elif collision.geometry.sphere is not None:
            linkPrimitive = Sphere(cfg = _generate_primitive_config(
                rawPose   = pose,
                offset    = offset,
                shapeName = 'Sphere',
                radius    = collision.geometry.sphere.radius
            ))
        elif collision.geometry.cylinder is not None:
            linkPrimitive = Cylinder(cfg = _generate_primitive_config(
                rawPose   = pose,
                offset    = offset,
                shapeName = 'Cylinder',
                r         = collision.geometry.cylinder.radius,
                h         = collision.geometry.cylinder.length
            ))
        else:
            warnings.warn(f"Not supported type: {type(collision.geometry.geometry)}")
            linkPrimitive = None
        return linkPrimitive

    def append_robot(self, robot: DiffRobot, offset_: Iterable[float] = torch.zeros(3)) -> Generator[Primitive, None, None]:
        """ Append a new URDF-loaded robot to the controller

        Params
        ------
        robot: the newly loaded robot
        offset: the offset of the robot's position, which will be added
            to each link of this robot's Cartesian primitives

        Returns
        -----
        A sequence of primitives that are derived from the robot's links
        """
        self.robots.append(robot)
        self.robot_action_dims.append(sum(
            joint.action_dim
            for joint in robot.actuated_joints
        ))
        self.link_2_primitives.append({})
        for linkName, link in robot.link_map.items():
            if len(link.collisions) > 1:
                raise ValueError(f"{linkName} has multiple collision")
            if len(link.collisions) == 0:
                warnings.warn(f"{linkName} has no collision")
            else:
                # converting
                linkPrimitive = self._urdf_collision_to_primitive(link.collisions[0], offset_, link.init_pose)
                if linkPrimitive is not None:
                    self.link_2_primitives[-1][linkName] = linkPrimitive
                    yield linkPrimitive
    
    def export_action_dims(self, to: List[int] = [0]): 
        """ Append the robots' action spaces to the actionDims list

        For example, plb.engine.primitive.Primitives maintains an action_dims list
        like [0, 3, 6, 8] which indicates the action[0:3] is for primitive[0], 
        action[3:6] for primitive[1], action[6:8] for primitive[2]. 
        
        Assuming we now have two robots, one of 10 DoFs the other of 5, then, 
        the action_dims should be appended to [0, 3, 6, 8, 18, 23]. 

        Params
        ------
        to: the `action_dims` in `plb.engine.primitive.Primitives` where
            the numbers of action dimensions of robots in this controller will
            be appended.
        """
        if not self.robots:
            return # no robots, so no effect on actionDims
        assert len(to) > 0, "Cannot append to empty action dims list"
        for dims in self.robot_action_dims:
            to.append(to[-1] + dims)
        return to

    @staticmethod
    def _deflatten_robot_actions(robot: Robot, robotActions: torch.Tensor) -> List:
        """ Deflatten a robot's action list according to its actuatable joints'
        action dimenstions. 

        For example, if a robot has four joints, whose action spaces are 
        (1,1), (2,1), (6,1) and (1,1). Assume the input action list is
        ```
        [0.121, 0.272, 0.336, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.57]
        ``` 
        It will therefore be deflattened to
        ```
        [
            0.121, 
            [0.272, 0.336], 
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            1.57
        ]
        ```
        Params
        ------
        robot: the urdf-loaded robot, for whom the flatten actions will be recovered
        robotActions: a flatten action lists

        Return
        ------
        A list of action configuration for EACH actuatable joint in the given robot
        """
        jointActions = []
        counter = 0
        for joint in robot.actuated_joints:
            if joint.action_dim == 1:
                jointActions.append(robotActions[counter])
                counter += 1
            else:
                jointActions.append(robotActions[counter : counter + joint.action_dim])
                counter += joint.action_dim

        return jointActions
    
    def set_action(self, step_idx: int, n_substep: int, actions: torch.Tensor):
        # inherited from Controller
        while self.current_step <= (step_idx + 1) * n_substep:
            if isinstance(actions, torch.Tensor):
                substep_actions = actions.clone().detach().requires_grad_(True).to(DEVICE)
            elif isinstance(actions, np.ndarray):
                substep_actions = torch.from_numpy(actions).requires_grad_(True).to(DEVICE)
            else:
                substep_actions = torch.tensor(actions,
                    dtype=torch.float64, requires_grad=True, device=DEVICE)
            dim_counter = 0
            for robot, action_dim in zip(self.robots, self.robot_action_dims):
                jointVelocity = self._deflatten_robot_actions(robot, 
                    substep_actions[dim_counter : dim_counter + action_dim])
                dim_counter += action_dim
                robot.link_fk_diff(jointVelocity)
            self.current_step += 1

    def _forward_kinematics(self, step_idx: int):
        # inherited from Controller
        for robot, primitive_dict in zip(self.robots, self.link_2_primitives):
            for name, link in robot._link_map.items():
                if name not in primitive_dict: continue
                primitive_dict[name].apply_robot_forward_kinemtaics(
                    step_idx, 
                    link.trajectory[step_idx]
                )

    def _forward_kinematics_grad(self, step_idx: int):
        # inherited from Controller
        for robot, primitive_dict in zip(reversed(self.robots), reversed(self.link_2_primitives)):
            for name, link in reversed(robot._link_map.items()):
                if name not in primitive_dict: continue
                primitive_dict[name].apply_robot_forward_kinemtaics.grad(
                    self     = primitive_dict[name],
                    frameIdx = step_idx, 
                    xyz_quat = link.trajectory[step_idx]
                )

    def get_step_grad(self, s: int) -> torch.Tensor:
        # inherited from Controller
        actionGrad = []
        for robot in self.robots:
            for grad in robot.fk_gradient(s + 1):
                actionGrad.append(grad)
        return torch.cat(actionGrad)
