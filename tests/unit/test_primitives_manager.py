import taichi

from plb.engine.primitive.primitives_manager import PrimitivesManager
from plb.engine.controller.robot_controller import RobotsController
from plb.urdfpy import DiffRobot, Mesh

taichi.init()


def test_primitive_management_w_free_primitives():
    pass #TODO



def test_primitive_management_w_single_robot():
    pm = PrimitivesManager()
    rc = RobotsController()
    robot = DiffRobot.load('tests/data/ur5/ur5_primitive.urdf')
    list(rc.append_robot(robot))

    pm.register_robot_primitives(rc)

    # test number of primitives
    linkCnt = len(pm)
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

    # test action dims
    robotActionDim = sum((
        joint.action_dim for joint in robot.actuated_joints
    ))
    assert len(pm.action_dims) == 2,\
        f"after appending robot's action dims, the action_dims become {pm.action_dims},"+\
        f" but expecting [0, {robotActionDim}]"
    
    # pretending there are 3 3-DoF primitives already
    pm = PrimitivesManager()
    pm.action_dims = [0,3,6,9] 
    pm.register_robot_primitives(rc)
    assert len(pm.action_dims) == 5,\
        f"after appending robot's action dims, the action_dims become {pm.action_dims},"+\
        f" but expecting [0, 3, 6, 9, {robotActionDim}]"



def test_primitive_management_w_dual_robot():
    pm = PrimitivesManager()
    rc = RobotsController()
    robotA = DiffRobot.load('tests/data/ur5/ur5_primitive.urdf')
    robotB = DiffRobot.load('tests/data/ur5/ur5_primitive.urdf')
    list(rc.append_robot(robotA))
    list(rc.append_robot(robotB))
    pm.register_robot_primitives(rc)
    
    # test number of primitives
    linkCnt = len(pm)
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

    # test action dims
    robotActionDim = sum((
        joint.action_dim for joint in robotA.actuated_joints
    ))
    assert len(pm.action_dims) == 3 and \
        pm.action_dims[1] - pm.action_dims[0] == pm.action_dims[2] - pm.action_dims[1] == robotActionDim,\
        f"after appending robot's action dims, the action_dims become {pm.action_dims},"+\
        f" but expecting [0, {robotActionDim, robotActionDim}]"
    
    # pretending there are 3 3-DoF primitives already
    pm = PrimitivesManager()
    pm.action_dims = [0,3,6,9] 
    pm.register_robot_primitives(rc)
    assert len(pm.action_dims) == 6 and \
        pm.action_dims[5] - pm.action_dims[4] == pm.action_dims[4] - pm.action_dims[3] == robotActionDim,\
        f"after appending robot's action dims, the action_dims become {pm.action_dims},"+\
        f" but expecting [0, 3, 6, 9, {robotActionDim}, {robotActionDim}]"



def test_primitive_management_w_free_primitives_and_robot():
    pass #TODO