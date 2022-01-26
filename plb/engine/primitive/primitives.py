import os
from typing import List, Iterable, List, Dict, Generator
import warnings

import numpy as np
import taichi as ti
import torch
import yaml
from yacs.config import CfgNode as CN

from plb.config.utils import make_cls_config
from plb.engine.primitive.primive_base import Primitive
from plb.engine.primitive.utils import qrot, qmul, w2quat
from plb.urdfpy import Robot, matrix_to_xyz_rpy, DiffRobot

ROBOT_LINK_DOF = 7
ROBOT_LINK_DOF_SCALE = tuple((0.01 for _ in range(ROBOT_LINK_DOF)))
ROBOT_COLLISION_COLOR = '(0.8, 0.8, 0.8)'

@ti.func
def length(x):
    return ti.sqrt(x.dot(x) + 1e-14)

@ti.func
def normalize(n):
    return n/length(n)


class Sphere(Primitive):
    def __init__(self, **kwargs):
        super(Sphere, self).__init__(**kwargs)
        self.radius = self.cfg.radius

    @ti.func
    def sdf(self, f, grid_pos):
        return length(grid_pos-self.position[f]) - self.radius

    @ti.func
    def normal(self, f, grid_pos):
        return normalize(grid_pos-self.position[f])

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.radius = 1.
        return cfg

class Capsule(Primitive):
    def __init__(self, **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.h = self.cfg.h
        self.r = self.cfg.r

    @ti.func
    def _sdf(self, f, grid_pos):
        p2 = grid_pos
        p2[1] += self.h / 2
        p2[1] -= min(max(p2[1], 0.0), self.h)
        return length(p2) - self.r

    @ti.func
    def _normal(self, f, grid_pos):
        p2 = grid_pos
        p2[1] += self.h / 2
        p2[1] -= min(max(p2[1], 0.0), self.h)
        return normalize(p2)

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.h = 0.06
        cfg.r = 0.03
        return cfg

class RollingPin(Capsule):
    # rollingpin's capsule...
    @ti.kernel
    def forward_kinematics(self, f: ti.i32):
        vel = self.v[f]
        dw = vel[0]  # rotate about object y
        dth = vel[1]  # rotate about the world w
        dy = vel[2]  # decrease in y coord...
        y_dir = qrot(self.rotation[f], ti.Vector([0., -1., 0.]))
        x_dir = ti.Vector([0., 1., 0.]).cross(y_dir) * dw * 0.03  # move toward x, R=0.03 is hand crafted...
        x_dir[1] = dy  # direction
        self.rotation[f+1] = qmul(
            w2quat(ti.Vector([0., -dth, 0.]), self.dtype),
            qmul(self.rotation[f], w2quat(ti.Vector([0., dw, 0.]), self.dtype))
        )
        #print(self.rotation[f+1], self.rotation[f+1].dot(self.rotation[f+1]))
        self.position[f+1] = max(min(self.position[f] + x_dir, self.xyz_limit[1]), self.xyz_limit[0])

class Chopsticks(Capsule):
    state_dim = 8
    def __init__(self, **kwargs):
        super(Chopsticks, self).__init__(**kwargs)
        self.gap = ti.field(self.dtype, needs_grad=True, shape=(self.max_timesteps,))
        self.gap_vel = ti.field(self.dtype, needs_grad=True, shape=(self.max_timesteps,))
        self.h = self.cfg.h
        self.r = self.cfg.r
        self.minimal_gap = self.cfg.minimal_gap
        assert self.action_dim == 7 # 3 linear, 3 angle, 1 for grasp ..

    @ti.kernel
    def forward_kinematics(self, f: ti.i32):
        self.gap[f+1] = max(self.gap[f] - self.gap_vel[f], self.minimal_gap)
        self.position[f+1] = max(min(self.position[f] + self.v[f], self.xyz_limit[1]), self.xyz_limit[0])
        self.rotation[f+1] = qmul(self.rotation[f], w2quat(self.w[f], self.dtype))
        #print(self.rotation[f+1])

    @ti.kernel
    def set_velocity(self, s: ti.i32, n_substeps:ti.i32):
        # rewrite set velocity for different
        for j in range(s*n_substeps, (s+1)*n_substeps):
            for k in ti.static(range(3)):
                self.v[j][k] = self.action_buffer[s][k] * self.action_scale[None][k]/n_substeps
            for k in ti.static(range(3)):
                self.w[j][k] = self.action_buffer[s][k+3] * self.action_scale[None][k+3]/n_substeps
            self.gap_vel[j] = self.action_buffer[s][6] * self.action_scale[None][6]/n_substeps

    @ti.func
    def _sdf(self, f, grid_pos):
        delta = ti.Vector([self.gap[f] / 2, 0., 0.])
        p = grid_pos - ti.Vector([0., -self.h/2, 0.])
        a = super(Chopsticks, self)._sdf(f, p-delta) # grid_pos - (mid + delta)
        b = super(Chopsticks, self)._sdf(f, p+delta) # grid_pos - (mid - delta)
        return ti.min(a, b)

    @ti.func
    def _normal(self, f, grid_pos):
        delta = ti.Vector([self.gap[f] / 2, 0., 0.])
        p = grid_pos - ti.Vector([0., -self.h/2, 0.])
        a = super(Chopsticks, self)._sdf(f, p-delta) # grid_pos - (mid + delta)
        b = super(Chopsticks, self)._sdf(f, p+delta) # grid_pos - (mid - delta)
        a_n = super(Chopsticks, self)._normal(f, p-delta) # grid_pos - (mid + delta)
        b_n = super(Chopsticks, self)._normal(f, p+delta) # grid_pos - (mid + delta)
        m = ti.cast(a <= b, self.dtype)
        return m * a_n + (1-m) * b_n

    @property
    def init_state(self):
        return self.cfg.init_pos + self.cfg.init_rot + (self.cfg.init_gap,)

    def get_state(self, f):
        return np.append(super(Chopsticks, self).get_state(f), self.gap[f])

    @ti.func
    def copy_frame(self, source, target):
        super(Chopsticks, self).copy_frame(source, target)
        self.gap[target] = self.gap[source]

    def set_state(self, f, state):
        assert len(state) == 8
        super(Chopsticks, self).set_state(f, state[:7])
        self.gap[f] = state[7]

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.h = 0.06
        cfg.r = 0.03
        cfg.minimal_gap = 0.06
        cfg.init_gap = 0.06
        return cfg

class Cylinder(Primitive):
    def __init__(self, **kwargs):
        super(Cylinder, self).__init__(**kwargs)
        self.h = self.cfg.h
        self.r = self.cfg.r

    @ti.func
    def _sdf(self, f, grid_pos):
        # convert it to a 2D box .. and then call the sdf of the 2d box..
        d = ti.abs(ti.Vector([length(ti.Vector([grid_pos[0], grid_pos[2]])), grid_pos[1]])) - ti.Vector([self.h, self.r])
        return min(max(d[0], d[1]), 0.0) + length(max(d, 0.0)) # if max(d, 0) < 0 or if max(d, 0) > 0

    @ti.func
    def _normal(self, f, grid_pos):
        p = ti.Vector([grid_pos[0], grid_pos[2]])
        l = length(p)
        d = ti.Vector([l, ti.abs(grid_pos[1])]) - ti.Vector([self.h, self.r])

        # if max(d) > 0, normal direction is just d
        # other wise it's 1 if d[1]>d[0] else -d0
        # return min(max(d[0], d[1]), 0.0) + length(max(d, 0.0))
        f = ti.cast(d[0] > d[1], self.dtype)
        n2 = max(d, 0.0) + ti.cast(max(d[0], d[1]) <= 0., self.dtype) * ti.Vector([f, 1-f]) # normal should be always outside ..
        n2_ = n2/length(n2)
        p2 = p/l
        n3 = ti.Vector([p2[0] * n2_[0], n2_[1] * (ti.cast(grid_pos[1]>=0, self.dtype) * 2 - 1), p2[1] * n2_[0]])
        return normalize(n3)

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.h = 0.2
        cfg.r = 0.1
        return cfg

class Torus(Primitive):
    def __init__(self, **kwargs):
        super(Torus, self).__init__(**kwargs)
        self.tx = self.cfg.tx
        self.ty = self.cfg.ty

    @ti.func
    def _sdf(self, f, grid_pos):
        q = ti.Vector([length(ti.Vector([grid_pos[0], grid_pos[2]])) - self.tx, grid_pos[1]])
        return length(q) - self.ty

    @ti.func
    def _normal(self, f, grid_pos):
        x = ti.Vector([grid_pos[0], grid_pos[2]])
        l = length(x)
        q = ti.Vector([length(x) - self.tx, grid_pos[1]])

        n2 = q/length(q)
        x2 = x/l
        n3 = ti.Vector([x2[0] * n2[0], n2[1], x2[1] * n2[0]])
        return normalize(n3)

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.tx = 0.2
        cfg.ty = 0.1
        return cfg

class Box(Primitive):
    def __init__(self, **kwargs):
        super(Box, self).__init__(**kwargs)
        self.size = ti.Vector.field(3, self.dtype, shape=())

    def initialize(self):
        super(Box, self).initialize()
        self.size[None] = self.cfg.size

    @ti.func
    def _sdf(self, f, grid_pos):
        # p: vec3,b: vec3
        q = ti.abs(grid_pos) - self.size[None]
        out = length(max(q, 0.0))
        out += min(max(q[0], max(q[1], q[2])), 0.0)
        return out

    @ti.func
    def _normal(self, f, grid_pos):
        #TODO: replace it with analytical normal later..
        d = ti.cast(1e-4, ti.float64)
        n = ti.Vector.zero(self.dtype, self.dim)
        for i in ti.static(range(self.dim)):
            inc = grid_pos
            dec = grid_pos
            inc[i] += d
            dec[i] -= d
            n[i] = (0.5 / d) * (self._sdf(f, inc) - self._sdf(f, dec))
        return n / length(n)

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.size = (0.1, 0.1, 0.1)
        return cfg


def _generate_primitive_config(rawPose: Iterable[float], offset:Iterable[float], shapeName: str, **kwargs) -> CN:
    """ Generate a CfgNode for primitive mapping

    Based on the URDF's primtives, generate the configuration for PLB's
    primitives, to establish the mapping in between

    Params
    ------
    rawPose: the pose matrix of primitive's initial position
    offset: the offset of the robot to which the primitve belongs
    shapeName: 'Sphere', 'Cylinder' or 'Box'
    **kwargs: other primitive-type specific parameters, such as
        the `size` description for `Box`, `r` and `h` for cylinders
    
    Return
    ------
    A CfgNode for the PLB's primitive instantiation
    """
    position = matrix_to_xyz_rpy(rawPose)
    actionCN = CN(init_dict={'dim': ROBOT_LINK_DOF, 'scale': f'{ROBOT_LINK_DOF_SCALE}'})
    configDict = {
        'action': actionCN, 
        'color':  ROBOT_COLLISION_COLOR, 
        'init_pos': f'({position[0] + offset[0]}, {position[1] + offset[1]}, {position[2] + offset[2]})',
        'init_rot': f'({1.0}, {0.1}, {0.1}, {0.1})',
        'shape': shapeName,
    }
    for key, value in kwargs.items():
        if isinstance(value, CN): configDict[key] = value
        else:                     configDict[key] = str(value)
    return CN(init_dict=configDict)


class RobotsControllers:
    """ A controller that maps robot to primitive links
    """
    def __init__(self) -> None:
        self.robots: List[DiffRobot] = []
        self.robot_action_dims: List[int] = []
        self.link_2_primtives: List[Dict[str, List[Primitive]]] = []

    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.shape = 'Robot'
        cfg.path = ''
        cfg.offset = (0.0, 0.0, 0.0)
        return cfg

    def append_robot(self, robot: DiffRobot, offset_: Iterable[float] = torch.zeros(3)) -> Generator[Primitive, None, None]:
        """ Append a new URDF-loaded robot to the controller

        Params
        ------
        robot: the newly loaded robot
        offset: the offset of the robot's position, which will be added
            to each link of this robot's Cartesian primitives

        Returns
        -----
        A sequence of primitives that are derived from the robot's links
        """
        self.robots.append(robot)
        self.robot_action_dims.append(sum(
            joint.action_dim
            for joint in robot.actuated_joints
        ))
        self.link_2_primtives.append({})
        initPose = robot.link_fk(use_names=True)
        for linkName, link in robot.link_map.items():
            for collision in link.collisions:
                if collision.geometry.box is not None:
                    linkPrimitive = Box(cfg = _generate_primitive_config(
                        rawPose   = initPose[linkName], 
                        offset    = offset_,
                        shapeName = 'Box',
                        size      = tuple(collision.geometry.box.size)
                    ))
                elif collision.geometry.sphere is not None:
                    linkPrimitive = Sphere(cfg = _generate_primitive_config(
                        rawPose   = initPose[linkName],
                        offset    = offset_,
                        shapeName = 'Sphere',
                        radius    = collision.geometry.sphere.radius
                    ))
                elif collision.geometry.cylinder is not None:
                    linkPrimitive = Cylinder(cfg = _generate_primitive_config(
                        rawPose   = initPose[linkName],
                        offset    = offset_,
                        shapeName = 'Cylinder',
                        r         = collision.geometry.cylinder.radius,
                        h         = collision.geometry.cylinder.length
                    ))
                else:
                    warnings.warn(f"Not supported type: {type(collision.geometry.geometry)}")
                    linkPrimitive = None

                if linkPrimitive is not None:
                    if linkName not in self.link_2_primtives:
                        self.link_2_primtives[-1][linkName] = [linkPrimitive]
                    else:
                        self.link_2_primtives[-1].append(linkPrimitive)
                    yield linkPrimitive
    
    def export_action_dims(self, to: List[int] = [0]): 
        """ Append the robots' action spaces to the actionDims list

        For example, plb.engine.primitive.Primitives maintains an action_dims list
        like [0, 3, 6, 8] which indicates the action[0:3] is for primitive[0], 
        action[3:6] for primitive[1], action[6:8] for primitive[2]. 
        
        Assuming we now have two robots, one of 10 DoFs the other of 5, then, 
        the action_dims should be appended to [0, 3, 6, 8, 18, 23]. 

        Params
        ------
        to: the `action_dims` in `plb.engine.primitive.Primitives` where
            the numbers of action dimensions of robots in this controller will
            be appended.
        """
        if not self.robots:
            return # no robots, so no effect on actionDims
        assert len(to) > 0, "Cannot append to empty action dims list"
        for dims in self.robot_action_dims:
            to.append(to[-1] + dims)
        return to

    @staticmethod
    def _deflatten_robot_actions(robot: Robot, robotActions: torch.Tensor) -> List:
        """ Deflatten a robot's action list according to its actuatable joints'
        action dimenstions. 

        For example, if a robot has four joints, whose action spaces are 
        (1,1), (2,1), (6,1) and (1,1). Assume the input action list is
        [0.121, 0.272, 0.336, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.57]. It will therefore
        be deflattened to
        [
            0.121, 
            [0.272, 0.336], 
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            1.57
        ]

        Params
        ------
        robot: the urdf-loaded robot, for whom the flatten actions will be recovered
        robotActions: a flatten action lists

        Return
        ------
        A list of action configuration for EACH actuatable joint in the given robot
        """
        jointActions = []
        counter = 0
        for joint in robot.actuated_joints:
            if joint.action_dim == 1:
                jointActions.append(robotActions[counter])
                counter += 1
            else:
                jointActions.append(robotActions[counter : counter + joint.action_dim])
                counter += joint.action_dim

        return jointActions
    
    def set_robot_actions(self, envActions: torch.Tensor, primitiveCnt: int = 0):
        """ Set the actions for the robot.

        Params
        ------
        envActions: a list of all the actions in this environment, the first 
            several elements of which might be irrelevant to the environment
        primitiveCnt: the number of the irrelevant actions
        """
        totalActions = len(envActions)
        assert primitiveCnt <= totalActions
        dimCounter = primitiveCnt
        for robot, actionDims, primitiveDict in zip(self.robots, self.robot_action_dims, self.link_2_primtives):
            jointVelocity = self._deflatten_robot_actions(
                robot,
                envActions[dimCounter : dimCounter + actionDims]
            )
            dimCounter += actionDims
            linkPose = robot.link_fk_diff(jointVelocity) # shape (LINK_CNT, 7)
            for eachLink, rowIdx in robot.link_2_idx.items():
                primitiveLst  = primitiveDict[eachLink.name]
                commonActions = linkPose[rowIdx, :] #shape (7,)
                for each in primitiveLst:
                    each.set_action() #TODO: put the linkPose into simulator

    def get_robot_action_grad(self, s, n): 
        pass

    def get_robot_action_step_grad(self, s, n):
        pass

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
        newRobot = DiffRobot.load(robotCfg.path)
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
