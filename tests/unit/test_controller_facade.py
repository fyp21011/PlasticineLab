import taichi as ti
import numpy as np

from plb.engine.controller_facade import ControllersFacade
from plb.engine.controller.primitive_controller import PrimitivesController
from plb.engine.controller.robot_controller import RobotsController
from plb.engine.primitive.primitive import Sphere, Cylinder
from plb.urdfpy import DiffRobot

ti.init()

def test_controller_facade_single_primitive():
    sphere_cfg = Sphere.default_config()
    sphere_cfg.action.dim = 3
    fpc = PrimitivesController([sphere_cfg])
    rc = None

    # test action_dim
    facade = ControllersFacade()
    facade.register_controllers(fpc, rc)
    assert facade.action_dim == 3, f"expect action_dim to be 3, but got {facade.action_dim}"

    # test set_action
    action = np.ones(facade.action_dim)
    facade.set_action(0, 20, action)
    assert fpc.primitives[0].action_buffer[0][0] == 1


def test_controller_facade_single_robot():
    fpc = None
    rc = RobotsController()
    robot = DiffRobot.load('tests/data/ur5/ur5_primitive.urdf')
    rc.append_robot(robot)

    # test action_dim
    facade = ControllersFacade()
    facade.register_controllers(fpc, rc)
    assert facade.action_dim == 6, f"expect action_dim to be 6, but got {facade.action_dim}"
    assert facade.accu_action_dims == [0, 6], f"expect action_dim to be 6, but got {facade.accu_action_dims}"

    #TODO: test set_action


def test_controller_facade_fpc_and_rc():
    sphere_cfg = Sphere.default_config()
    sphere_cfg.action.dim = 3
    fpc = PrimitivesController([Cylinder.default_config(), sphere_cfg, sphere_cfg], max_timesteps=2)
    rc = RobotsController()
    robotA = DiffRobot.load('tests/data/ur5/ur5_primitive.urdf')
    rc.append_robot(robotA)
    # Having the second robot causes assertion failure: 
    # (int)snodes.size() <= taichi_max_num_snodes
    # robotB = DiffRobot.load('tests/data/ur5/ur5_primitive.urdf')
    # rc.append_robot(robotB)

    facade = ControllersFacade()
    facade.register_controllers(fpc, rc)

    # test action_dim
    expected_action_dim = 0+3+3+6 # +6
    assert facade.action_dim == expected_action_dim, f"expect action_dim to be {expected_action_dim}, but got {facade.action_dim}"
    assert len(facade.accu_action_dims) == 5 # 6

    # test set_action
    action = np.ones(facade.action_dim)
    facade.set_action(0, 20, action)
    assert fpc.primitives[1].action_buffer[0][0] == 1
    assert fpc.primitives[2].action_buffer[0][0] == 1
    # TODO: test robot action
