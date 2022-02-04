from typing import Any, Callable

import numpy as np
import torch

from plb.engine.robots import ROBOT_LINK_DOF
from plb.envs import PlasticineEnv
from plb.engine.taichi_env import TaichiEnv
from plb.urdfpy.diff_fk import DEVICE

def test_load_robot_env():
    cfg = PlasticineEnv.load_varaints('rope_robot.yml', 1)
    tcEnv = TaichiEnv(cfg, False)
    assert len(tcEnv.action_dims) == 5 # [0, 3, 6, 6, 12]
    assert tcEnv.action_dim == 12 # 3 of one sphere, 3 of another, 6 for robot
    assert len(tcEnv.robots_controller.robots) == 1
    assert len(tcEnv.robots_controller.robots[0].link_map.keys()) == 11

class MockedSimulator:
    def __init__(self, checker:Callable[[Any], bool]):
        self.checker = checker
    
    def step(self, *args, **kwargs):
        allArgs = []
        for posArg in args:
            allArgs.append(str(posArg))
        for key, arg in kwargs.items():
            allArgs.append(f'{key}={arg}')
        assert self.checker(*args, **kwargs), f'{self}.step({", ".join(allArgs)})'


def test_robot_env_step():
    cfg = PlasticineEnv.load_varaints('rope_robot.yml', 1)
    tcEnv = TaichiEnv(cfg, False, False)

    def simulator_step_checker(**kwargs):
        actions = kwargs['action']
        return isinstance(actions, np.ndarray) and \
               actions.shape == (tcEnv.primitives.action_dim + sum(
                   ROBOT_LINK_DOF * len(eachRobot.values())
                   for eachRobot in tcEnv.robots_controller.link_2_primitives
               ), )
    
    tcEnv.simulator = MockedSimulator(simulator_step_checker)
    tcEnv.step(torch.rand((tcEnv.action_dim, ), device=DEVICE, dtype=torch.float64, requires_grad=True))
