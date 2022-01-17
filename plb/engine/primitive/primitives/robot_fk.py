from typing import List, Dict, Generator

import numpy as np

from plb.engine.primitive.primive_base import Primitive
from plb.urdfpy import Robot, FK_CFG_Type

class RobotsControllers:
    """ A controller for tracking URDF loaded robots' actions
    """
    def __init__(self) -> None:
        self.robots: List[Robot]                         = []
        self.link_2_primtive: List[Dict[str, Primitive]] = []
        self.robot_action_dims: List[int]                = []

    def append_robot(self, robot: Robot, offset: np.ndarray = None) -> Generator[Primitive, None, None]:
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
        self.link_2_primtive.append({})
        for linkName in robot.link_map.keys():
            linkPrimitive = Primitive()
            self.link_2_primtive[-1][linkName] = linkPrimitive
            yield linkPrimitive #TODO: create primitive & offsets
    
    def append_action_dims(self, actionDims: List[int]): 
        """ Append the robots' action spaces to the actionDims list

        For example, plb.engine.primitive.Primitives maintains an action_dims list
        like [0, 3, 6, 8] which indicates the action[0:3] is for primitive[0], 
        action[3:6] for primitive[1], action[6:8] for primitive[2]. 
        
        Assuming we now have two robots, one of 10 DoFs the other of 5, then, 
        the action_dims should be appended to [0, 3, 6, 8, 18, 23]. 

        Params
        ------
        actionDims: the `action_dims` in `plb.engine.primitive.Primitives` where
            the numbers of action dimensions of robots in this controller will
            be appended.
        """
        if not self.robots:
            return # no robots, so no effect on actionDims
        assert len(actionDims) > 0, "Cannot append to empty action dims list"
        for dims in self.robot_action_dims:
            actionDims.append(actionDims[-1] + dims)

    @staticmethod
    def _deflatten_robot_actions(robot: Robot, robotActions: np.ndarray) -> List:
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
    
    def set_robot_actions(self, envActions: np.ndarray, primitiveCnt: int = 0):
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
        for robot, actionDims in zip(self.robots, self.robot_action_dims):
            jointVelocity = self._deflatten_robot_actions(
                robot,
                envActions[dimCounter : dimCounter + actionDims]
            )
            dimCounter += actionDims
        linkPose = robot.link_fk(jointVelocity, cfgType=FK_CFG_Type.velocity)
        #TODO: 

    def get_robot_action_grad(self, s, n): 
        pass

    def get_robot_action_step_grad(self, s, n):
        pass
