from .controller import Controller
from plb.engine.primitive.primitive import Box, Sphere, Cylinder
# DO NOT PRUNE THIS LINE, otherwise the `eval` fails

import numpy as np
import yaml
from yacs.config import CfgNode as CN

class PrimitivesController(Controller):
    def __init__(self, cfgs, max_timesteps=1024):
        super().__init__()
        outs = []
        self.primitives = []
        for i in cfgs:
            if isinstance(i, CN):
                cfg = i
            else:
                cfg = CN(new_allowed=True)
                cfg = cfg._load_cfg_from_yaml_str(yaml.safe_dump(i))
            outs.append(cfg)

        self.action_dims = [0]
        for i in outs:
            primitive = eval(i.shape)(cfg=i, max_timesteps=max_timesteps)
            self.primitives.append(primitive)
            self.action_dims.append(self.action_dims[-1] + primitive.action_dim)
        self.n = len(self.primitives)
        """freeze number of free primitives
        
        Those primitives created & controlled by robots
        will not be counted here since they are added after self.n is initialized
        """

    def _forward_kinematics(self, s):
        for i in range(self.n):
            self.primitives[i].forward_kinematics(s)

    def _forward_kinematics_grad(self, s):
        for i in range(self.n-1, -1, -1):
            self.primitives[i].forward_kinematics.grad(s)

    @property
    def not_empty(self) -> bool:
        return len(self.primitives) > 0

    @property
    def action_dim(self):
        return self.action_dims[-1]

    def set_action(self, s, n_substeps, action):
        action = np.asarray(action).reshape(-1).clip(-1, 1)
        assert len(action) == self.action_dims[-1]
        for i in range(self.n):
            self.primitives[i].set_action(s, n_substeps, action[self.action_dims[i]:self.action_dims[i+1]])

    def get_grad(self, n):
        grads = []
        for i in range(self.n):
            grad = self.primitives[i].get_action_grad(0, n)
            if grad is not None:
                grads.append(grad)
        return np.concatenate(grads, axis=1)

    def get_step_grad(self,n):
        grads = []
        for i in range(self.n): # only the FREE PRIMITIVES
            grad = self.primitives[i].get_step_action_grad(n)
            if grad is not None:
                grads.append(grad)
        return np.concatenate(grads,axis=0)
