import numpy as np
import cv2
import taichi as ti
import torch

from plb.utils import VisRecordable
from protocol.message import DeformableMeshesMessage
from .mpm_simulator import MPMSimulator
from .renderer import Renderer
from .shapes import Shapes
from .losses import Loss
from .nn.mlp import MLP

# TODO: run on GPU, fast_math will cause error on float64's sqrt; removing it cuases compile error..
ti.init(arch=ti.gpu, debug=False, fast_math=True)

#TODO: merge TaichiEnv with PlasticineEnv
class TaichiEnv(VisRecordable):
    def __init__(self, cfg, nn=False, loss=True):
        """
        A taichi env builds scene according the configuration and the set of manipulators
        """
        self.shapes = Shapes(cfg.SHAPES)
        self.init_particles, self.particle_colors = self.shapes.get()

        cfg.SIMULATOR.defrost()
        self.n_particles = cfg.SIMULATOR.n_particles = len(self.init_particles)

        self.simulator = MPMSimulator(cfg)

        self.primitives_facade = self.simulator.primitives_facade
        self.renderer = Renderer(cfg.RENDERER, self.primitives_facade)

        if nn:
            self.nn = MLP(self.simulator, self.primitives_facade, (256, 256))

        if loss:
            self.loss = Loss(cfg.ENV.loss, self.simulator)
        else:
            self.loss = None
        self._is_copy = True
        self.step_cnt = 0
    
    @property
    def action_dim(self) -> int:
        return self.simulator.controllers_facade.action_dim

    def set_copy(self, is_copy: bool):
        self._is_copy = is_copy

    def initialize(self):
        # initialize all taichi variable according to configurations..
        self.primitives_facade.initialize()
        self.simulator.initialize()
        self.renderer.initialize()
        if self.loss:
            self.loss.initialize()
            self.renderer.set_target_density(
                self.loss.target_density.to_numpy()/self.simulator.p_mass)

        # call set_state instead of reset..
        self.simulator.reset(self.init_particles)
        if self.loss:
            self.loss.clear()
        self.step_cnt = 0

    def render(self, mode='human', **kwargs):
        assert self._is_copy, "The environment must be in the copy mode for render ..."
        if self.n_particles > 0:
            x = self.simulator.get_x(0)
            self.renderer.set_particles(x, self.particle_colors)
        img = self.renderer.render_frame(shape=1, primitive=1, **kwargs)
        img = np.uint8(img.clip(0, 1) * 255)

        if mode == 'human':
            cv2.imshow('x', img[..., ::-1])
            cv2.waitKey(1)
        elif mode == 'plt':
            import matplotlib.pyplot as plt
            plt.imshow(img)
            plt.show()
        else:
            return img

    def get_obs(self, n_observed_particles):
        if self._is_copy:
            t = 0
        else:
            t = self.simulator.cur

        x = self.simulator.get_x(t, needs_grad=False)
        v = self.simulator.get_v(t, needs_grad=False)
        outs = []
        for i in self.primitives_facade: 
            outs.append(i.get_state(t, needs_grad=False))
        s = np.concatenate(outs)
        step_size = len(x) // n_observed_particles
        return np.concatenate((np.concatenate((x[::step_size], v[::step_size]), axis=-1).reshape(-1), s.reshape(-1)))

    def step(self, action=None):
        if isinstance(action, torch.Tensor):
            action = action.detach()
        self.simulator.step(is_copy=self._is_copy, action=action)
        if self.is_recording:
            x = self.simulator.get_x(0, False)
            DeformableMeshesMessage.Factory(
                "plasticine",
                self.step_cnt * self.STEP_INTERVAL * self.simulator.substeps,
                pcd = self.y_up_2_z_up(x)
            ).message.send()
        self.step_cnt += 1

    def compute_loss(self):
        assert self.loss is not None
        if self._is_copy:
            self.loss.clear()
            return self.loss.compute_loss(0)
        else:
            return self.loss.compute_loss(self.simulator.cur)

    def get_state(self):
        assert self.simulator.cur == 0
        return {
            'state': self.simulator.get_state(0),
            'softness': self.primitives_facade.get_softness(),
            'is_copy': self._is_copy
        }

    def set_state(self, state, softness, is_copy):
        self.simulator.cur = 0
        self.simulator.set_state(0, state)
        self.primitives_facade.set_softness(softness)
        self._is_copy = is_copy
        if self.loss:
            self.loss.reset()
            self.loss.clear()
    
    def set_torch_nn(self,nn):
        self.simulator.set_nn(nn)
    # obs will be an numpy array
    # obs will be the last step cur
    def act(self,obs):
        action = np.zeros(self.action_dim)
        self.simulator.act(obs,self.simulator.cur,action)
        return action
