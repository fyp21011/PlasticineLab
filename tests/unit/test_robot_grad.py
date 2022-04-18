import taichi as ti
import torch

from plb.engine.primitives_facade import PrimitivesFacade
from plb.engine.controller.robot_controller import RobotsController
from plb.urdfpy import DiffRobot


def test_robot_and_primitives():
    ti.init()
    primitives = PrimitivesFacade()
    robot = DiffRobot.load("tests/data/ur5/ur5_primitive.urdf")
    rc = RobotsController()
    rc.append_robot(robot)
    primitives.register_robot_primitives(rc)

    assert len(primitives) == 8, \
        f"8 primitives from the robot is expected, but got {len(primitives)}"

    action_dim = rc.robot_action_dims[0]
    assert action_dim == 6, f"action_dim is expected to be 6, but instead is {action_dim}"

    primitives.initialize()

    # forward pass
    actions = torch.rand((7,))
    rc.set_action(0, 5, actions)
    actions = torch.rand((7,))
    rc.set_action(1, 5, actions)
    for i in range(10):
        rc.forward_kinematics(i)

    # backward pass
    for i in range(9, -1, -1):
        rc.forward_kinematics.grad(i)

    grad = rc.get_step_grad(0)
    assert len(grad) != 0 and all((each_grad != None for each_grad in grad)), \
        f"Gradient backpropagation fails: step[0] gradient = {grad}"
    grad = rc.get_step_grad(1)
    assert len(grad) != 0 and all((each_grad != None for each_grad in grad)), \
        f"Gradient backpropagation fails: step[1] gradient = {grad}"

    ti.reset()