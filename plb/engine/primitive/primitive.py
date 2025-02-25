from typing import Iterable
import torch
import numpy as np
import fcl
import taichi as ti
from open3d import geometry
from yacs.config import CfgNode as CN

from plb.utils.vis_recorder import VisRecordable
from protocol import AddRigidBodyPrimitiveMessage, DeformableMeshesMessage, UpdateRigidBodyPoseMessage

from ...config.utils import make_cls_config
from .utils import qrot, qmul, w2quat, inv_trans


@ti.func
def length(x):
    return ti.sqrt(x.dot(x) + 1e-14)

@ti.func
def normalize(n):
    return n/length(n)

def _bottom_center_2_volume_center(pose: np.ndarray, h: float) -> np.ndarray:
    """ Switch the local cylinder origin from its
    bottom surface center to its gravity center

    The formula is: o' = o + R(h / 2)
    where o is the cylinder center, R is the rotation
    matrix, and the h is the vector [0, 0, h], i.e.
    the axis of the cylinder
    """
    R = geometry.get_rotation_matrix_from_quaternion(pose[3:])
    h = np.array([0, 0, h / 2])
    return pose[:3] + np.dot(R, h)

def _volume_center_2_bottom_center(pose: np.ndarray, h: float) -> np.ndarray:
    """ The reverse function of 
    `self.bottom_center_2_volume_center`
    """
    R = geometry.get_rotation_matrix_from_quaternion(pose[3:])
    h = np.array([0, 0, h / 2])
    return pose[:3] - np.dot(R, h)

def plb_pose_2_z_up_rhs(pose, is_cylinder = False, cylinder_h = 0):
    z_up_pose = VisRecordable.y_up_2_z_up(pose)
    if is_cylinder:
        z_up_pose[:3] = _bottom_center_2_volume_center(z_up_pose, cylinder_h)
    return z_up_pose

def z_up_rhs_pose_2_plb(pose, is_cylinder = False, cylinder_h = 0):
    if is_cylinder:
        pose[:3] = _volume_center_2_bottom_center(pose, cylinder_h)
    return pose[[1, 2, 0, 3, 5, 6, 4]]

ROBOT_LINK_DOF = 7
ROBOT_LINK_DOF_SCALE = tuple((0.01 for _ in range(ROBOT_LINK_DOF)))
ROBOT_COLLISION_COLOR = '(0.8, 0.8, 0.8)'

def primitive_cfg_in_mem(rawPose: torch.Tensor, shapeName: str, **kwargs) -> CN:
    """ Generate a CfgNode for primitive in memory 
    instead of loading from `yml` files

    Based on the URDF's primtives, generate the configuration for PLB's
    primitives, to establish the mapping in between

    Params
    ------
    rawPose: a 7-dim tensor specifying the 
    offset: the offset of the robot to which the primitve belongs
    shapeName: 'Sphere', 'Cylinder' or 'Box'
    **kwargs: other primitive-type specific parameters, such as
        the `size` description for `Box`, `r` and `h` for cylinders
    
    Return
    ------
    A CfgNode for the PLB's primitive instantiation
    """
    if rawPose.shape != (7,):
        raise ValueError(f"expecting a 7-dim Tensor as the initial position, got {rawPose}")
    rawPose = z_up_rhs_pose_2_plb(
        pose = rawPose.detach().cpu().numpy(),
        is_cylinder = shapeName == "Cylinder",
        cylinder_h = kwargs.get('h', 0)
    )
    actionCN = CN(init_dict={'dim': ROBOT_LINK_DOF, 'scale': f'{ROBOT_LINK_DOF_SCALE}'})
    configDict = {
        'action': actionCN, 
        'color':  ROBOT_COLLISION_COLOR, 
        'init_pos': f'({rawPose[0]}, {rawPose[1]}, {rawPose[2]})',
        'init_rot': f'({rawPose[3]}, {rawPose[4]}, {rawPose[5]}, {rawPose[6]})',
        'shape': shapeName
    }
    for key, value in kwargs.items():
        if isinstance(value, CN): configDict[key] = value
        else:                     configDict[key] = str(value)
    return CN(init_dict=configDict)


@ti.data_oriented
class Primitive(VisRecordable):
    # single primitive ..
    state_dim = 7

    def __init__(self, cfg=None, dim=3, max_timesteps=1024, dtype=ti.f64, **kwargs):
        """
        The primitive has the following functions ...
        """
        self.cfg = make_cls_config(self, cfg, **kwargs)
        print('Building primitive')
        print(self.cfg)

        self.dim = dim
        self.max_timesteps = max_timesteps
        self.dtype = dtype

        self.rotation_dim = 4
        self.angular_velocity_dim = 3

        self.friction = ti.field(dtype, shape=())
        self.softness = ti.field(dtype, shape=())
        # positon of the primitive
        self.color = ti.Vector.field(3, ti.f32, shape=())
        self.position = ti.Vector.field(
            3, dtype, needs_grad=True)  # positon of the primitive
        # quaternion for storing rotation
        self.rotation = ti.Vector.field(4, dtype, needs_grad=True)

        self.v = ti.Vector.field(3, dtype, needs_grad=True)  # velocity
        self.w = ti.Vector.field(3, dtype, needs_grad=True)  # angular velocity

        ti.root.dense(ti.i, (self.max_timesteps,)).place(self.position, self.position.grad, self.rotation, self.rotation.grad,
                                                         self.v, self.v.grad, self.w, self.w.grad)
        self.xyz_limit = ti.Vector.field(
            3, ti.f64, shape=(2,))  # positon of the primitive

        self.action_dim = self.cfg.action.dim

        if self.action_dim > 0:
            self.action_buffer = ti.Vector.field(
                self.action_dim, dtype, needs_grad=True, shape=(max_timesteps,))
            self.action_scale = ti.Vector.field(
                self.action_dim, dtype, shape=())
            # record min distance to the point cloud..
            self.min_dist = ti.field(dtype, shape=(), needs_grad=True)
            # record min distance to the point cloud..
            self.dist_norm = ti.field(dtype, shape=(), needs_grad=True)

        self.register_scene_init_callback(self._init_object_in_visualizer)

    def _init_object_in_visualizer(self):
        """ Callback when the visualizer is turned on

        Send a message to the visualizer to initialize the primitive
        object in the scene. 

        NOTE: desipte rigid-body primitives, the default representation
        of their spacial shape is as well the SDF values. Hence, the
        initialization borrows the factory method from the 
        `DeformableMeshesMessage` to transfer SDF to meshes.
        """
        sdf_ = np.zeros((256, 256, 256))
        self.get_global_sdf_kernel(0, sdf_)
        sdf_ = self.plb_sdf_2_z_up_rhs(sdf_)
        DeformableMeshesMessage.Factory(
            self.unique_name,
            0,
            sdf = sdf_, 
            scale = (256, 256, 256) 
        ).message.send()

    @ti.func
    def _sdf(self, f, grid_pos):
        raise NotImplementedError

    @ti.func
    def _normal(self, f, grid_pos):
        raise NotImplementedError

    @ti.func
    def sdf(self, f, grid_pos):
        grid_pos = inv_trans(grid_pos, self.position[f], self.rotation[f])
        return self._sdf(f, grid_pos)

    @ti.kernel
    def get_global_sdf_kernel(self, f: ti.i32, out: ti.ext_arr()):
        for I in ti.grouped(ti.ndrange(256, 256, 256)):
            I_scaled = I / 256
            out[I] = self.sdf(f, I_scaled)

    # @ti.complex_kernel_grad(get_global_sdf_kernel)
    # def get_global_sdf_kernel_grad(self, f: ti.i32, out: ti.ext_arr()):
    #     return

    @ti.func
    def normal2(self, f, p):
        d = ti.cast(1e-8, ti.float64)
        n = ti.Vector.zero(self.dtype, self.dim)
        for i in ti.static(range(self.dim)):
            inc = p
            dec = p
            inc[i] += d
            dec[i] -= d

            n[i] = (0.5 / d) * (self.sdf(f, inc) - self.sdf(f, dec))
        return n/length(n)

    @ti.func
    def normal(self, f, grid_pos):
        #n2 = self.normal2(f, grid_pos)
        #xx = grid_pos
        grid_pos = inv_trans(grid_pos, self.position[f], self.rotation[f])
        return qrot(self.rotation[f], self._normal(f, grid_pos))

    @ti.func
    def collider_v(self, f, grid_pos, dt):
        inv_quat = ti.Vector(
            [self.rotation[f][0], -self.rotation[f][1], -self.rotation[f][2], -self.rotation[f][3]]).normalized()
        relative_pos = qrot(inv_quat, grid_pos - self.position[f])
        new_pos = qrot(self.rotation[f + 1],
                       relative_pos) + self.position[f + 1]
        collider_v = (new_pos - grid_pos) / dt  # TODO: revise
        return collider_v

    @ti.func
    def collide(self, f, grid_pos, v_out, dt):
        dist = self.sdf(f, grid_pos)
        influence = min(ti.exp(-dist * self.softness[None]), 1)
        if (self.softness[None] > 0 and influence > 0.1) or dist <= 0:
            D = self.normal(f, grid_pos)
            collider_v_at_grid = self.collider_v(f, grid_pos, dt)

            input_v = v_out - collider_v_at_grid
            normal_component = input_v.dot(D)

            grid_v_t = input_v - min(normal_component, 0) * D

            grid_v_t_norm = length(grid_v_t)
            grid_v_t_friction = grid_v_t / grid_v_t_norm * \
                max(0, grid_v_t_norm + normal_component * self.friction[None])
            flag = ti.cast(normal_component < 0 and ti.sqrt(
                grid_v_t.dot(grid_v_t)) > 1e-30, self.dtype)
            grid_v_t = grid_v_t_friction * flag + grid_v_t * (1 - flag)
            v_out = collider_v_at_grid + input_v * \
                (1 - influence) + grid_v_t * influence

            #print(self.position[f], f)
            #print(grid_pos, collider_v, v_out, dist, self.friction, D)
            # if v_out[1] > 1000:
            #print(input_v, collider_v_at_grid, normal_component, D)

        return v_out
    
    def _update_pose_in_vis_recorder(self, frame_idx, is_init = False):
        state_idx = frame_idx + (0 if is_init else 1)
        if is_init or self.is_recording():
            UpdateRigidBodyPoseMessage(
                self.unique_name, 
                plb_pose_2_z_up_rhs(self.get_state(state_idx, False)),
                frame_idx * self.STEP_INTERVAL
            ).send()


    def forward_kinematics(self, f):
        self.forward_kinematics_kernel(f)
        self._update_pose_in_vis_recorder(f)

    @ti.kernel
    def forward_kinematics_kernel(self, f: ti.i32):
        self.position[f+1] = max(min(self.position[f] +
                                 self.v[f], self.xyz_limit[1]), self.xyz_limit[0])
        # rotate in world coordinates about itself.
        self.rotation[f+1] = qmul(w2quat(self.w[f],
                                  self.dtype), self.rotation[f])

    @ti.complex_kernel
    def apply_robot_forward_kinemtaics(self, frame_idx: ti.i32, xyz_quat: torch.Tensor):
        """ The robot's foward kinematics computes the target postion and
        rotation of each primitive geometry for each substep. The method
        applies this computation results to the primitive geometries.

        Parameters
        ----------
        frame_idx: the time index
        xyz_quat: the position and global rotation the primitive geometry
            should be moved to
        """
        if xyz_quat.shape[-1] != 7:
            raise ValueError(f"XYZ expecting Tensor of shape (..., 7), got {xyz_quat.shape}")
        xyz_quat = z_up_rhs_pose_2_plb(
            pose = xyz_quat.detach().cpu().numpy(), 
            is_cylinder = self.cfg.shape == "Cylinder",
            cylinder_h = getattr(self, 'h') if hasattr(self, 'h') else 0
        )
        targetXYZ = xyz_quat[:3]
        targetQuat = xyz_quat[3:]

        self.position[frame_idx + 1] = np.clip(
            targetXYZ,
            self.xyz_limit[0].value,
            self.xyz_limit[1].value
        )
        self.rotation[frame_idx + 1] = targetQuat

        self._update_pose_in_vis_recorder(frame_idx)

    @ti.complex_kernel_grad(apply_robot_forward_kinemtaics)
    def forward_kinematics_gradient_backward_2_robot(self, frameIdx: ti.i32, xyz_quat: torch.Tensor):
        grads = torch.zeros_like(xyz_quat, device = xyz_quat.device)
        for i in range(3):
            grads[i] = self.position.grad[frameIdx + 1][i]
        for i in range(4):
            grads[3 + i] = self.rotation.grad[frameIdx + 1][i]
        grads = plb_pose_2_z_up_rhs(
            pose = xyz_quat.detach().cpu().numpy(), 
            is_cylinder = self.cfg.shape == "Cylinder",
            cylinder_h = getattr(self, 'h') if hasattr(self, 'h') else 0
        )
        xyz_quat.backward(grads, retain_graph=True)


    # state set and copy ...
    @ti.func
    def copy_frame(self, source, target):
        self.position[target] = self.position[source]
        self.rotation[target] = self.rotation[source]

    def _get_fcl_tf(self, f):
        state = self.get_state(f, False)
        state = plb_pose_2_z_up_rhs(state)
        pos, rot = state[:3], state[3:]
        return fcl.Transform(rot, pos)

    def to_fcl_obj_and_geom(self): 
        raise NotImplementedError

    @ti.kernel
    def get_state_kernel(self, f: ti.i32, controller: ti.ext_arr()):
        for j in ti.static(range(3)):
            controller[j] = self.position[f][j]
        for j in ti.static(range(4)):
            controller[j+self.dim] = self.rotation[f][j]

    @ti.complex_kernel
    def no_grad_get_state_kernel(self, f: ti.i32, controller: ti.ext_arr()):
        self.get_state_kernel(f, controller)

    @ti.complex_kernel_grad(no_grad_get_state_kernel)
    def no_grad_get_state_kernel_grad(self, f: ti.i32, controller: ti.ext_arr()):
        return

    @ti.kernel
    def set_state_kernel(self, f: ti.i32, controller: ti.ext_arr()):
        for j in ti.static(range(3)):
            self.position[f][j] = controller[j]
        for j in ti.static(range(4)):
            self.rotation[f][j] = controller[j+self.dim]

    def get_state(self, f, needs_grad=True):
        out = np.zeros((7), dtype=np.float64)
        if needs_grad:
            self.get_state_kernel(f, out)
        else:
            self.no_grad_get_state_kernel(f, out)
        return out

    def set_state(self, f, state):
        ss = self.get_state(f)
        ss[:len(state)] = state
        self.set_state_kernel(f, ss)

    @property
    def init_state(self):
        return self.cfg.init_pos + self.cfg.init_rot

    def initialize(self):
        cfg = self.cfg
        self.set_state(0, self.init_state)
        self.xyz_limit.from_numpy(np.array([cfg.lower_bound, cfg.upper_bound]))
        self.color[None] = cfg.color
        self.friction[None] = self.cfg.friction  # friction coefficient
        if self.action_dim > 0:
            self.action_scale[None] = cfg.action.scale

    @ti.kernel
    def set_action_kernel(self, s: ti.i32, action: ti.ext_arr()):
        for j in ti.static(range(self.action_dim)):
            self.action_buffer[s][j] = action[j]

    @ti.complex_kernel
    def no_grad_set_action_kernel(self, s, action):
        self.set_action_kernel(s, action)

    @ti.complex_kernel_grad(no_grad_set_action_kernel)
    def no_grad_set_action_kernel_grad(self, s, action):
        return

    @ti.kernel
    def get_action_grad_kernel(self, s: ti.i32, n: ti.i32, grad: ti.ext_arr()):
        for i in range(0, n):
            for j in ti.static(range(self.action_dim)):
                grad[i, j] = self.action_buffer.grad[s+i][j]

    @ti.kernel
    def get_step_action_grad_kernel(self,s:ti.i32,grad:ti.ext_arr()):
        for i in ti.static(range(self.action_dim)):
            grad[i] = self.action_buffer.grad[s][i]

    @ti.kernel
    def set_velocity(self, s: ti.i32, n_substeps: ti.i32):
        # rewrite set velocity for different
        for j in range(s*n_substeps, (s+1)*n_substeps):
            for k in ti.static(range(3)):
                self.v[j][k] = self.action_buffer[s][k] * \
                    self.action_scale[None][k]/n_substeps
            if ti.static(self.action_dim > 3):
                for k in ti.static(range(3)):
                    self.w[j][k] = self.action_buffer[s][k+3] * \
                        self.action_scale[None][k+3]/n_substeps

    def set_action(self, s, n_substeps, action):
        # set actions for n_substeps ...
        if self.action_dim > 0:
            # HACK: taichi can't compute gradient to this.
            self.no_grad_set_action_kernel(s, action)
            self.set_velocity(s, n_substeps)

    def get_action_grad(self, s, n):
        if self.action_dim > 0:
            grad = np.zeros((n, self.action_dim), dtype=np.float64)
            self.get_action_grad_kernel(s, n, grad)
            return grad
        else:
            return None

    def get_step_action_grad(self,s):
        if self.action_dim > 0:
            grad = np.zeros(self.action_dim, dtype=np.float64)
            self.get_step_action_grad_kernel(s,grad)
            return grad
        else:
            return None

    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.shape = ''
        cfg.name = '' # default name 
        cfg.init_pos = (0.3, 0.3, 0.3)  # default color
        cfg.init_rot = (1., 0., 0., 0.)  # default color
        cfg.color = (0.3, 0.3, 0.3)  # default color
        cfg.lower_bound = (0., 0., 0.)  # default color
        cfg.upper_bound = (1., 1., 1.)  # default color
        cfg.friction = 0.9  # default color
        cfg.variations = None  # TODO: not support now

        action = cfg.action = CN()
        action.dim = 0  # in this case it can't move ...
        action.scale = ()
        return cfg

    @property
    def unique_name(self) -> str:
        return f"{self.cfg.shape}_{self.cfg.name if len(self.cfg.name) != 0 else abs(hash(self))}"


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
    
    def to_fcl_obj_and_geom(self, f):
        geom = fcl.Sphere(self.radius)
        tf = self._get_fcl_tf(f)
        obj = fcl.CollisionObject(geom, tf)
        return obj, geom

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.shape = 'Sphere'
        cfg.radius = 1.
        return cfg

    def _init_object_in_visualizer(self):
        AddRigidBodyPrimitiveMessage(
            self.unique_name,
            "bpy.ops.mesh.primitive_uv_sphere_add",
            radius = self.radius
        ).send()
        self._update_pose_in_vis_recorder(0, True)


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
    
    def to_fcl_obj_and_geom(self, f):
        geom = fcl.Cylinder(self.r, self.h)
        tf = self._get_fcl_tf(f)
        obj = fcl.CollisionObject(geom, tf)
        return obj, geom

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.shape = 'Cylinder'
        cfg.h = 0.2
        cfg.r = 0.1
        return cfg

    def _init_object_in_visualizer(self):
        AddRigidBodyPrimitiveMessage(
            self.unique_name,
            "bpy.ops.mesh.primitive_cylinder_add",
            radius = self.r,
            depth = self.h
        ).send()
        self._update_pose_in_vis_recorder(0, True)

    def _get_fcl_tf(self, f):
        state = self.get_state(f, False)
        state = plb_pose_2_z_up_rhs(state, True, self.h)
        pos, rot = state[:3], state[3:]
        return fcl.Transform(rot, pos)
    

    def _update_pose_in_vis_recorder(self, frame_idx, is_init=False):
        if is_init or self.is_recording():
            state_idx = frame_idx + (0 if is_init else 1)
            pose = plb_pose_2_z_up_rhs(self.get_state(state_idx, False), True, self.h)
            UpdateRigidBodyPoseMessage(self.unique_name, pose, frame_idx * self.STEP_INTERVAL).send()



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
    
    def to_fcl_obj_and_geom(self, f):
        x, y, z = self.cfg.size
        geom = fcl.Box(x, y, z)
        tf = self._get_fcl_tf(f)
        obj = fcl.CollisionObject(geom, tf)
        return obj, geom

    @classmethod
    def default_config(cls):
        cfg = Primitive.default_config()
        cfg.size = (0.1, 0.1, 0.1)
        return cfg

    def _init_object_in_visualizer(self):
        AddRigidBodyPrimitiveMessage(
            self.unique_name, 
            "bpy.ops.mesh.primitive_cube_add",
            size = 1.0,
            scale = self.cfg.size
        ).send()
        self._update_pose_in_vis_recorder(0, True)
