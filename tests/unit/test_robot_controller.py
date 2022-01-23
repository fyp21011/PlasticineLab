from typing import Any, Dict

import numpy as np
import taichi

from plb.engine.primitive.primitives import RobotsControllers
from plb.urdfpy import Robot, Link, FK_CFG_Type, Mesh

taichi.init()

def test_deflatten_robot_actions():
    robot = Robot.load('tests/data/ur5/ur5.urdf')
    robotActionDim = sum((
        joint.action_dim for joint in robot.actuated_joints
    ))
    envAction = [np.random.random() for _ in range(robotActionDim)]
    jointActions = RobotsControllers._deflatten_robot_actions(robot, envAction)
    assert np.all(np.equal(jointActions, envAction))

    envAction = [0.33, 0.59, 0.01] + envAction
    jointActions = RobotsControllers._deflatten_robot_actions(robot, envAction[3:])
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
    jointActions = RobotsControllers._deflatten_robot_actions(FakeRobot(), envAction[3:])
    assert len(jointActions) == 3 and \
        jointActions[0] == 0.1 and \
        len(jointActions[1]) == 2 and jointActions[1][0] == 0.22 and jointActions[1][1] == 0.33 and\
        len(jointActions[2]) == 3 and jointActions[2][0] == 0.44 and jointActions[2][1] == 0.55 and\
        jointActions[2][2] == 0.66, f"{envAction} => {jointActions}"

def test_single_robot():
    rc = RobotsControllers()
    robot = Robot.load('tests/data/ur5/ur5.urdf')


    # test append robot
    linkCnt = sum((
        1 for _ in rc.append_robot(robot)
    ))
    truePrimitiveCnt = sum((
        sum((1 for c in link.collisions if not isinstance(c.geometry.geometry, Mesh))) for link in robot.links
    ))
    assert linkCnt == truePrimitiveCnt,\
        f"got {linkCnt} Primitives, but there are {truePrimitiveCnt} in the RC"
    assert 1 == len(rc.robots),\
        f"got {len(rc.robots)} robots in the RC, but expecting 1"
    assert 1 == len(rc.link_2_primtives),\
        f"got {len(rc.link_2_primtives)} link_2_primitive in the RC, but expecting 1"
    for linkName, link in robot.link_map.items():
        if any((
            not isinstance(collision.geometry.geometry, Mesh)
            for collision in link.collisions
        )):
            assert linkName in rc.link_2_primtives[0],\
                f"{linkName} of the loaded robot not in rc.link_2_primitive"

    # test append action dims
    action_dims = [0]
    robotActionDim = sum((
        joint.action_dim for joint in robot.actuated_joints
    ))
    rc.export_action_dims(action_dims)
    assert len(action_dims) == 2,\
        f"after appending robot's action dims, the action_dims become {action_dims},"+\
        f" but expecting [0, {robotActionDim}]"
    action_dims = [0,3,6,9] # pretending there are 3 3-DoF primitives already
    rc.export_action_dims(action_dims)
    assert len(action_dims) == 5,\
        f"after appending robot's action dims, the action_dims become {action_dims},"+\
        f" but expecting [0, 3, 6, 9, {robotActionDim}]"
    
    # test set robot actions
    envAction = [0.33, 0.66, 0.72] + [np.random.random() for _ in range(robotActionDim)]
    rc.set_robot_actions(envAction, 3)
    poseA: Dict[Link, Any] = robot._current_cfg
    robot.link_fk(envAction[3:], cfgType=FK_CFG_Type.angle)
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
    rc = RobotsControllers()
    robotA = Robot.load('tests/data/ur5/ur5.urdf')
    robotB = Robot.load('tests/data/ur5/ur5.urdf')
    
    linkCnt = sum((
        1 for _ in rc.append_robot(robotA)
    )) + sum ((
        1 for _ in rc.append_robot(robotB)
    ))
    truePrimitiveCnt = sum((
        sum((1 for c in link.collisions if not isinstance(c.geometry.geometry, Mesh))) for link in robotA.links
    )) + sum((
        sum((1 for c in link.collisions if not isinstance(c.geometry.geometry, Mesh))) for link in robotB.links
    ))
    assert linkCnt == truePrimitiveCnt,\
        f"got {linkCnt} Primitives, but there are {truePrimitiveCnt} in the RC"
    assert 2 == len(rc.robots),\
        f"got {len(rc.robots)} robots in the RC, but expecting 2"
    assert 2 == len(rc.link_2_primtives),\
        f"got {len(rc.link_2_primtives)} link_2_primitive in the RC, but expecting 2"
    for linkName, link in robotA.link_map.items():
        if any((
            not isinstance(collision.geometry.geometry, Mesh)
            for collision in link.collisions
        )):
            assert linkName in rc.link_2_primtives[0],\
                f"{linkName} of the loaded robot not in rc.link_2_primitive"
    for linkName, link in robotB.link_map.items():
        if any((
            not isinstance(collision.geometry.geometry, Mesh)
            for collision in link.collisions
        )):
            assert linkName in rc.link_2_primtives[1],\
                f"{linkName} of the loaded robot not in rc.link_2_primitive"

    action_dims = [0]
    robotActionDim = sum((
        joint.action_dim for joint in robotA.actuated_joints
    ))
    rc.export_action_dims(action_dims)
    assert len(action_dims) == 3 and \
        action_dims[1] - action_dims[0] == action_dims[2] - action_dims[1] == robotActionDim,\
        f"after appending robot's action dims, the action_dims become {action_dims},"+\
        f" but expecting [0, {robotActionDim, robotActionDim}]"
    action_dims = [0,3,6,9] # pretending there are 3 3-DoF primitives already
    rc.export_action_dims(action_dims)
    assert len(action_dims) == 6 and \
        action_dims[5] - action_dims[4] == action_dims[4] - action_dims[3] == robotActionDim,\
        f"after appending robot's action dims, the action_dims become {action_dims},"+\
        f" but expecting [0, 3, 6, 9, {robotActionDim}, {robotActionDim}]"
    
    envAction = [0.33, 0.66, 0.72] + [np.random.random() for _ in range(robotActionDim * 2)]
    rc.set_robot_actions(envAction, 3)
    poseA: Dict[Link, Any] = robotB._current_cfg
    robotB.link_fk(envAction[3 + robotActionDim:], cfgType=FK_CFG_Type.angle)
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
