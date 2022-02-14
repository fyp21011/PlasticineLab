import taichi
import torch

from plb.engine.primitive import Primitives
from plb.engine.robots import RobotsController
from plb.urdfpy import DiffRobot

taichi.init()

def test_robot_and_primitives():
    primitives = Primitives([]) # empty primitives
    robot = DiffRobot.load("tests/data/ur5/ur5_primitive.urdf")
    rc = RobotsController()
    for shape in rc.append_robot(robot, (0.0, 0.0, 0.0)):
        primitives.primitives.append(shape)
    assert primitives.n == 0, f"No free primitives expected, but got {primitives.n}"
    assert len(primitives.primitives) == 8, \
        f"8 primitives from the robot is expected, but got {len(primitives.primitives)}"

    action_dims = primitives.action_dims.copy()
    assert len(action_dims) == 1 and action_dims[0] == 0, \
        f"primitives.action_dims is expected to be [0], but got {action_dims}"
    rc.export_action_dims(to = action_dims)
    assert len(action_dims) == 2 and action_dims[0] == 0 and action_dims[1] == 6, \
        f"action_dims after the robot's exporting is expected to be [0,6], but got {action_dims}"

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