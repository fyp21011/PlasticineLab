from typing import Iterable, List, Dict, Generator, Union
import warnings

import torch
from yacs.config import CfgNode as CN

from plb.engine.primitive.primitives.shapes import Box, Cylinder, Sphere
from plb.engine.primitive.primive_base import Primitive
from plb.urdfpy import Robot, matrix_to_xyz_rpy
from plb.urdfpy.diff_fk import DiffRobot

ROBOT_LINK_DOF = 7
ROBOT_LINK_DOF_SCALE = tuple((0.01 for _ in range(ROBOT_LINK_DOF)))
ROBOT_COLLISION_COLOR = '(0.8, 0.8, 0.8)'

def _generate_primitive_config(rawPose: Iterable[float], offset:Iterable[float], shapeName: str, **kwargs) -> CN:
    """ Generate a CfgNode for primitive mapping

    Based on the URDF's primtives, generate the configuration for PLB's
    primitives, to establish the mapping in between

    Params
    ------
    rawPose: the pose matrix of primitive's initial position
    offset: the offset of the robot to which the primitve belongs
    shapeName: 'Sphere', 'Cylinder' or 'Box'
    **kwargs: other primitive-type specific parameters, such as
        the `size` description for `Box`, `r` and `h` for cylinders
    
    Return
    ------
    A CfgNode for the PLB's primitive instantiation
    """
    position = matrix_to_xyz_rpy(rawPose)
    actionCN = CN(init_dict={'dim': ROBOT_LINK_DOF, 'scale': f'{ROBOT_LINK_DOF_SCALE}'})
    configDict = {
        'action': actionCN, 
        'color':  ROBOT_COLLISION_COLOR, 
        'init_pos': f'({position[0] + offset[0]}, {position[1] + offset[1]}, {position[2] + offset[2]})',
        'init_rot': f'({1.0}, {0.1}, {0.1}, {0.1})',
        'shape': shapeName,
    }
    for key, value in kwargs.items():
        if isinstance(value, CN): configDict[key] = value
        else:                     configDict[key] = str(value)
    return CN(init_dict=configDict)


class RobotsControllers:
    """ A controller for tracking URDF loaded robots' actions
    """
    def __init__(self) -> None:
        self.robots: List[DiffRobot] = []
        self.robot_action_dims: List[int] = []
        self.link_2_primtives: List[Dict[str, List[Primitive]]] = []

    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.shape = 'Robot'
        cfg.path = ''
        cfg.offset = (0.0, 0.0, 0.0)
        return cfg

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
        self.link_2_primtives.append({})
        initPose = robot.link_fk(use_names=True)
        for linkName, link in robot.link_map.items():
            for collision in link.collisions:
                if collision.geometry.box is not None:
                    linkPrimitive = Box(cfg = _generate_primitive_config(
                        rawPose   = initPose[linkName], 
                        offset    = offset_,
                        shapeName = 'Box',
                        size      = tuple(collision.geometry.box.size)
                    ))
                elif collision.geometry.sphere is not None:
                    linkPrimitive = Sphere(cfg = _generate_primitive_config(
                        rawPose   = initPose[linkName],
                        offset    = offset_,
                        shapeName = 'Sphere',
                        radius    = collision.geometry.sphere.radius
                    ))
                elif collision.geometry.cylinder is not None:
                    linkPrimitive = Cylinder(cfg = _generate_primitive_config(
                        rawPose   = initPose[linkName],
                        offset    = offset_,
                        shapeName = 'Cylinder',
                        r         = collision.geometry.cylinder.radius,
                        h         = collision.geometry.cylinder.length
                    ))
                else:
                    warnings.warn(f"Not supported type: {type(collision.geometry.geometry)}")
                    linkPrimitive = None

                if linkPrimitive is not None:
                    if linkName not in self.link_2_primtives:
                        self.link_2_primtives[-1][linkName] = [linkPrimitive]
                    else:
                        self.link_2_primtives[-1].append(linkPrimitive)
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
        [0.121, 0.272, 0.336, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.57]. It will therefore
        be deflattened to
        [
            0.121, 
            [0.272, 0.336], 
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            1.57
        ]

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
    
    def set_robot_actions(self, envActions: torch.Tensor, primitiveCnt: int = 0):
        """ Set the actions for the robot.

        Params
        ------
        envActions: a list of all the actions in this environment, the first 
            several elements of which might be irrelevant to the environment
        primitiveCnt: the number of the irrelevant actions
        """
        totalActions = len(envActions)
        assert primitiveCnt <= totalActions
        dimCounter = primitiveCnt
        for robot, actionDims, primitiveDict in zip(self.robots, self.robot_action_dims, self.link_2_primtives):
            jointVelocity = self._deflatten_robot_actions(
                robot,
                envActions[dimCounter : dimCounter + actionDims]
            )
            dimCounter += actionDims
            linkPose = robot.link_fk_diff(jointVelocity) # shape (LINK_CNT, 7)
            for eachLink, rowIdx in robot.link_2_idx.items():
                primitiveLst  = primitiveDict[eachLink.name]
                commonActions = linkPose[rowIdx, :] #shape (7,)
                for each in primitiveLst:
                    each.set_action() #TODO: put the linkPose into simulator

    def get_robot_action_grad(self, s, n): 
        pass

    def get_robot_action_step_grad(self, s, n):
        pass
