import os
from typing import List

import numpy as np
import taichi as ti
import yaml
from yacs.config import CfgNode as CN

from plb.config.utils import make_cls_config
from plb.engine.primitive.primive_base import Primitive
from plb.urdfpy import Robot

from .robot_fk import RobotsControllers
from .shapes import Sphere, Capsule, RollingPin, Chopsticks, Cylinder, Box

class Primitives:
    def __init__(self, cfgs, max_timesteps=1024):
        self.primitives: List[Primitive] = []
        self.action_dims = [0]

        outs = (
            each if isinstance(each, CN) else CN(new_allowed=True)._load_cfg_from_yaml_str(yaml.safe_dump(each))
            for each in cfgs
        )
        robotList = []
        for eachOutCfg in outs:
            if eachOutCfg.shape == 'Robot':
                robotList.append(eachOutCfg)
            else:
                primitive = eval(eachOutCfg.shape)(cfg=eachOutCfg, max_timesteps=max_timesteps)
                self.primitives.append(primitive)
                self.action_dims.append(self.action_dims[-1] + primitive.action_dim)
        self.n = len(self.primitives)
        """ Number of the non-robot primitives"""
        self._robots = RobotsControllers()
        for eachRobotConfig in robotList:
            self._add_robot(eachRobotConfig)
        self._robots.export_action_dims(to = self.action_dims)

    def _add_robot(self, cfg: CN):
        """ Load an articulated robot into the env

        Retrieve the robot's links from the environment
        and insert them as the primitives into the Env

        Params
        ------
        cfg: the YAML CfgNode, whose ROBOT element, if exists,
            will be understood as a path to the URDF file describing
            the expected 
        """
        robotCfg = make_cls_config(self._robots, cfg)
        assert isinstance(robotCfg.path, str), \
            f"invalid ROBOT configuration in {cfg}"
        assert os.path.exists(robotCfg.path), \
            f"no such robot @ {robotCfg}"
        newRobot = Robot.load(robotCfg.path)
        robotPos = robotCfg.offset
        for robotPrimitive in self._robots.append_robot(newRobot, robotPos):
            self.primitives.append(robotPrimitive)


    @property
    def action_dim(self):
        return self.action_dims[-1]

    @property
    def state_dim(self):
        return sum([i.state_dim for i in self.primitives])

    def set_action(self, s, n_substeps, action):
        action = np.asarray(action).reshape(-1).clip(-1, 1)
        assert len(action) == self.action_dims[-1]
        for i in range(self.n):
            self.primitives[i].set_action(s, n_substeps, action[self.action_dims[i]:self.action_dims[i+1]])
        self._robots.set_robot_actions(envAction = action, primitiveCnt=self.n)
                

    def get_grad(self, n):
        grads = []
        for i in range(self.n):
            grad = self.primitives[i].get_action_grad(0, n)
            if grad is not None:
                grads.append(grad)
        for robotActionGrad in self._robots.get_robot_action_grad(0, n):
            grads.append(robotActionGrad)
        return np.concatenate(grads, axis=1)

    def get_step_grad(self,n):
        grads = []
        for i in range(self.n):
            grad = self.primitives[i].get_step_action_grad(n)
            if grad is not None:
                grads.append(grad)
        for robotActionGrad in self._robots.get_robot_action_step_grad(0, n):
            grads.append(robotActionGrad)
        return np.concatenate(grads,axis=0)

    def set_softness(self, softness=666.):
        for i in self.primitives:
            i.softness[None] = softness

    def get_softness(self):
        return self.primitives[0].softness[None]

    def __getitem__(self, item):
        if isinstance(item, tuple):
            item = item[0]
        return self.primitives[item]

    def __len__(self):
        return len(self.primitives)

    def initialize(self):
        for i in self.primitives:
            i.initialize()
