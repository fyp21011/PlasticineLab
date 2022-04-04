from typing import Any, Dict

import numpy as np
import taichi
import torch

from plb.engine.controller.robot_controller import RobotsController
from plb.urdfpy import DiffRobot, Link, FK_CFG_Type, Mesh

taichi.init()

def test_deflatten_robot_actions():
    robot = DiffRobot.load('tests/data/ur5/ur5.urdf')
    robotActionDim = sum((
        joint.action_dim for joint in robot.actuated_joints
    ))
    envAction = [np.random.random() for _ in range(robotActionDim)]
    jointActions = RobotsController._deflatten_robot_actions(robot, envAction)
    assert np.all(np.equal(jointActions, envAction))

    envAction = [0.33, 0.59, 0.01] + envAction
    jointActions = RobotsController._deflatten_robot_actions(robot, envAction[3:])
    assert np.all(np.equal(jointActions, envAction[3:]))

    class FakeJoint:
        def __init__(self, n) -> None:
            self.action_dim = n

    class FakeRobot:
        def __init__(self) -> None:
            self.actuated_joints = [
                FakeJoint(1), 
                FakeJoint(2), 
                FakeJoint(3)
            ]
    
    envAction = [0.33, 0.59, 0.01, 0.1, 0.22, 0.33, 0.44, 0.55, 0.66]
    jointActions = RobotsController._deflatten_robot_actions(FakeRobot(), envAction[3:])
    assert len(jointActions) == 3 and \
        jointActions[0] == 0.1 and \
        len(jointActions[1]) == 2 and jointActions[1][0] == 0.22 and jointActions[1][1] == 0.33 and\
        len(jointActions[2]) == 3 and jointActions[2][0] == 0.44 and jointActions[2][1] == 0.55 and\
        jointActions[2][2] == 0.66, f"{envAction} => {jointActions}"

def test_single_robot():
    rc = RobotsController()
    robot = DiffRobot.load('tests/data/ur5/ur5_primitive.urdf')
    rc.append_robot(robot)

    # test append robot
    linkCnt = sum(len(mapping) for mapping in rc.link_2_primitives)
    truePrimitiveCnt = sum((
        sum((1 for c in link.collisions if not isinstance(c.geometry.geometry, Mesh))) for link in robot.links
    ))
    assert linkCnt == truePrimitiveCnt,\
        f"got {linkCnt} Primitives, but there are {truePrimitiveCnt} in the RC"
    assert 1 == len(rc.robots),\
        f"got {len(rc.robots)} robots in the RC, but expecting 1"
    assert 1 == len(rc.link_2_primitives),\
        f"got {len(rc.link_2_primitives)} link_2_primitive in the RC, but expecting 1"
    for linkName, link in robot.link_map.items():
        if any((
            not isinstance(collision.geometry.geometry, Mesh)
            for collision in link.collisions
        )):
            assert linkName in rc.link_2_primitives[0],\
                f"{linkName} of the loaded robot not in rc.link_2_primitive"

    # test append action dims
    # action_dims = [0]
    robotActionDim = sum((
        joint.action_dim for joint in robot.actuated_joints
    ))
    # rc.export_action_dims(action_dims)
    # assert len(action_dims) == 2,\
    #     f"after appending robot's action dims, the action_dims become {action_dims},"+\
    #     f" but expecting [0, {robotActionDim}]"
    # action_dims = [0,3,6,9] # pretending there are 3 3-DoF primitives already
    # rc.export_action_dims(action_dims)
    # assert len(action_dims) == 5,\
    #     f"after appending robot's action dims, the action_dims become {action_dims},"+\
    #     f" but expecting [0, 3, 6, 9, {robotActionDim}]"
    
    # test set robot actions
    envAction = [
        torch.rand((1,), device='cuda', dtype=torch.float64, requires_grad=True)
        for _ in range(robotActionDim)
    ]
    rc.set_action(0, 1, envAction)
    poseA: Dict[Link, Any] = robot._current_cfg
    robot.link_fk(envAction, cfgType=FK_CFG_Type.angle)
    poseB: Dict[Link, Any] = robot._current_cfg
    poseA = {
        link.name : pose
        for link, pose in poseA.items()
    }
    poseB = {
        link.name : pose
        for link, pose in poseB.items()
    }
    for linkName in poseA:
        if linkName in poseB:
            assert np.all(np.isclose(poseA[linkName], poseB[linkName], rtol=1e-3)), \
                f"Incorrect FK for {linkName}, expecting {poseB[linkName]}, got {poseA[linkName]}"


def test_dual_robot():
    rc = RobotsController()
    robotA = DiffRobot.load('tests/data/ur5/ur5_primitive.urdf')
    rc.append_robot(robotA)
    robotB = DiffRobot.load('tests/data/ur5/ur5_primitive.urdf')
    rc.append_robot(robotB)
    
    linkCnt = sum(len(mapping) for mapping in rc.link_2_primitives)
    truePrimitiveCnt = sum((
        sum((1 for c in link.collisions if not isinstance(c.geometry.geometry, Mesh))) for link in robotA.links
    )) + sum((
        sum((1 for c in link.collisions if not isinstance(c.geometry.geometry, Mesh))) for link in robotB.links
    ))
    assert linkCnt == truePrimitiveCnt,\
        f"got {linkCnt} Primitives, but there are {truePrimitiveCnt} in the RC"
    assert 2 == len(rc.robots),\
        f"got {len(rc.robots)} robots in the RC, but expecting 2"
    assert 2 == len(rc.link_2_primitives),\
        f"got {len(rc.link_2_primitives)} link_2_primitive in the RC, but expecting 2"
    for linkName, link in robotA.link_map.items():
        if any((
            not isinstance(collision.geometry.geometry, Mesh)
            for collision in link.collisions
        )):
            assert linkName in rc.link_2_primitives[0],\
                f"{linkName} of the loaded robot not in rc.link_2_primitive"
    for linkName, link in robotB.link_map.items():
        if any((
            not isinstance(collision.geometry.geometry, Mesh)
            for collision in link.collisions
        )):
            assert linkName in rc.link_2_primitives[1],\
                f"{linkName} of the loaded robot not in rc.link_2_primitive"

    # action_dims = [0]
    robotActionDim = sum((
        joint.action_dim for joint in robotA.actuated_joints
    ))
    # rc.export_action_dims(action_dims)
    # assert len(action_dims) == 3
    # assert action_dims[1] - action_dims[0] == action_dims[2] - action_dims[1] == robotActionDim,\
    #     f"after appending robot's action dims, the action_dims become {action_dims},"+\
    #     f" but expecting [0, {robotActionDim, robotActionDim}]"
    # action_dims = [0,3,6,9] # pretending there are 3 3-DoF primitives already
    # rc.export_action_dims(action_dims)
    # assert len(action_dims) == 6 and \
    #     action_dims[5] - action_dims[4] == action_dims[4] - action_dims[3] == robotActionDim,\
    #     f"after appending robot's action dims, the action_dims become {action_dims},"+\
    #     f" but expecting [0, 3, 6, 9, {robotActionDim}, {robotActionDim}]"
    
    envAction = [
        torch.rand((1,), device='cuda', dtype=torch.float64, requires_grad=True)
        for _ in range(robotActionDim * 2)
    ]
    rc.set_action(0, 1, envAction)
    poseA: Dict[Link, Any] = robotB._current_cfg
    robotB.link_fk(envAction[robotActionDim:], cfgType=FK_CFG_Type.angle)
    poseB: Dict[Link, Any] = robotB._current_cfg
    poseA = {
        link.name : pose
        for link, pose in poseA.items()
    }
    poseB = {
        link.name : pose
        for link, pose in poseB.items()
    }
    for linkName in poseA:
        if linkName in poseB:
            assert np.all(np.isclose(poseA[linkName], poseB[linkName], rtol=1e-3)), \
                f"Incorrect FK for {linkName}, expecting {poseB[linkName]}, got {poseA[linkName]}"
