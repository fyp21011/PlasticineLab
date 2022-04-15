import os
from typing import Any, List, Dict, Iterable, Union, Generator
import warnings
import yaml

import numpy as np
import torch
from yacs.config import CfgNode as CN

from plb.config.utils import make_cls_config
from plb.utils import VisRecordable
from .controller import Controller
from plb.urdfpy import DiffRobot, Robot, Collision, DEVICE, Geometry
from plb.engine.primitive.primitive import Box, Sphere, Cylinder, Primitive, primitive_cfg_in_mem
from protocol import MeshesMessage, AddRigidBodyPrimitiveMessage, UpdateRigidBodyPoseMessage

class _RobotLinkVisualizer:
    def __init__(self, name: str, link_geometry: Geometry, init_pose: Union[np.ndarray, torch.Tensor]) -> None:
        self.name = name
        self.geometry = link_geometry
        if isinstance(init_pose, torch.Tensor):
            self.pose = init_pose.detach().cpu().numpy()
        else:
            self.pose = init_pose

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.geometry.mesh is not None:
            with open(self.geometry.mesh.filename, 'rb') as reader:
                filecontent = reader.read()
            MeshesMessage(
                self.name + '.' + os.path.splitext(self.geometry.mesh.filename)[1],
                filecontent,
                init_pose = self.pose
            ).send()
        else:
            if self.geometry.box is not None:
                init_msg = AddRigidBodyPrimitiveMessage(
                    self.name, 
                    "bpy.ops.mesh.primitive_cube_add",
                    size = 1.0,
                    scale = (self.geometry.box.size[0], self.geometry.box.size[1], self.geometry.box.size[2])
                )
            elif self.geometry.cylinder is not None:
                init_msg = AddRigidBodyPrimitiveMessage(
                    self.name, 
                    "bpy.ops.mesh.primitive_cylinder_add",
                    radius = self.geometry.cylinder.radius,
                    depth = self.geometry.cylinder.length
                )
            elif self.geometry.sphere is not None:
                init_msg = AddRigidBodyPrimitiveMessage(
                    self.name, 
                    "bpy.ops.mesh.primitive_uv_sphere_add",
                    radius = self.geometry.sphere.radius
                )
            else:
                raise NotImplementedError()
            init_msg.send()
            # then set the position
            UpdateRigidBodyPoseMessage(self.name, self.pose, frame_idx = 0).send()

    def __repr__(self) -> str:
        return f'{self.name} @ {self.pose}, linked to {self.geometry}'
    
    def __str__(self) -> str:
        return self.__repr__

class RobotsController(Controller, VisRecordable):
    """ A controller that maps robot to primitive links
    """
    def __init__(self) -> None:
        super().__init__()
        self.robots: List[DiffRobot] = []
        self.robot_action_dims: List[int] = [] # not [0]
        """ for the i-th robot in `self.robots`, 
        `self.robot_action_dims[i]` gives the numbe
        action dimensions of the robot. 
        """
        self.link_2_primitives: List[Dict[str, Primitive]] = []
        """ `link_2_primitives[RobotIdx][LinkName]` returns the primivite
        corresponds to the collision geometry of the link named as `LinkName`
        in `RobotIdx`'s robot
        """
        self.current_step = 0
        """pointer to the current step

        Example
        -------
        `self.current_step = 3` ==> 
        * the first tree, i.e., `self.flatten_actions[:3]` are executed
        * `self.flatten_actions[3]` are the one to be executed next
        """

    @property
    def not_empty(self) -> bool:
        return len(self.robots) > 0

    @classmethod
    def parse_config(cls, cfgs: List[Union[CN, str]]) -> "RobotsController":
        """ Parse the YAML configuration node for `Robots`
        
        Load the robots from the URDF files specified by the `Robots` config node.

        Params
        ------
        cfgs: the YAML list titled `Robots` in the env configuration file
        """
        outs = []
        for eachCfg in cfgs:
            if isinstance(eachCfg, CN):
                cfg = eachCfg
            else:
                cfg = CN(new_allowed=True)
                cfg = cfg._load_cfg_from_yaml_str(yaml.safe_dump(eachCfg))
            outs.append(make_cls_config(cls, cfg))
        rc = RobotsController()
        for eachOutCfg in outs:
            diffRobot = DiffRobot.load(eachOutCfg.path)
            rc.append_robot(diffRobot)
        return rc

    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.shape = 'Robot'
        cfg.path = ''
        return cfg

    @classmethod
    def _urdf_collision_to_primitive(cls, collision: Collision, pose: torch.Tensor, **kwargs) -> Union[Primitive, None]:
        """ Converting the URDF's collision geometry to primitive

        Params
        ------
        collision: the URDF robot's collision geometry
        pose: the pose matrix (4 * 4) of this geometry, which
            records the initial position of the geometry
        """
        if collision.geometry.box is not None:
            linkPrimitive = Box(cfg = primitive_cfg_in_mem(
                rawPose   = pose, 
                shapeName = 'Box',
                size      = tuple(collision.geometry.box.size),
                **kwargs
            ))
        elif collision.geometry.sphere is not None:
            linkPrimitive = Sphere(cfg = primitive_cfg_in_mem(
                rawPose   = pose,
                shapeName = 'Sphere',
                radius    = collision.geometry.sphere.radius,
                **kwargs
            ))
        elif collision.geometry.cylinder is not None:
            linkPrimitive = Cylinder(cfg = primitive_cfg_in_mem(
                rawPose   = pose,
                shapeName = 'Cylinder',
                r         = collision.geometry.cylinder.radius,
                h         = collision.geometry.cylinder.length,
                **kwargs
            ))
        else:
            warnings.warn(f"Not supported type: {type(collision.geometry.geometry)}")
            linkPrimitive = None
        return linkPrimitive

    def append_robot(self, robot: DiffRobot) -> Generator[Primitive, None, None]:
        """ Append a new URDF-loaded robot to the controller

        Params
        ------
        robot: the newly loaded robot

        Returns
        -----
        A sequence of primitives that are derived from the robot's links
        """
        self.robots.append(robot)
        self.robot_action_dims.append(sum(
            joint.action_dim
            for joint in robot.actuated_joints
        ))
        self.link_2_primitives.append({})
        robot_idx = len(self.robots) - 1
        for linkName, link in robot.link_map.items():
            if len(link.collisions) > 1:
                raise ValueError(f"{linkName} has multiple collision")
            if len(link.collisions) == 0:
                warnings.warn(f"{linkName} has no collision")
            else:
                # converting
                linkPrimitive = self._urdf_collision_to_primitive(
                    link.collisions[0],
                    link.collision_pose(0, 0), 
                    name = f'{robot.name}_{robot_idx}/{linkName}_collision'
                )
                if linkPrimitive is not None:
                    self.link_2_primitives[-1][linkName] = linkPrimitive
            for vis_idx, vis in enumerate(link.visuals):
                vis_name = f'{robot.name}_{robot_idx}/{linkName}_{vis_idx}'
                self.register_scene_init_callback(_RobotLinkVisualizer(vis_name, vis.geometry, link.init_pose))


    @staticmethod
    def _deflatten_robot_actions(robot: Robot, robotActions: torch.Tensor) -> List:
        """ Deflatten a robot's action list according to its actuatable joints'
        action dimenstions. 

        For example, if a robot has four joints, whose action spaces are 
        (1,1), (2,1), (6,1) and (1,1). Assume the input action list is
        ```
        [0.121, 0.272, 0.336, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.57]
        ``` 
        It will therefore be deflattened to
        ```
        [
            0.121, 
            [0.272, 0.336], 
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            1.57
        ]
        ```
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
    
    def set_action(self, step_idx: int, n_substep: int, actions: torch.Tensor):
        # inherited from Controller
        while self.current_step <= (step_idx + 1) * n_substep:
            if isinstance(actions, torch.Tensor):
                substep_actions = actions.clone().detach().requires_grad_(True).to(DEVICE)
            elif isinstance(actions, np.ndarray):
                substep_actions = torch.from_numpy(actions).requires_grad_(True).to(DEVICE)
            else:
                substep_actions = torch.tensor(actions,
                    dtype=torch.float64, requires_grad=True, device=DEVICE)
            dim_counter = 0
            for robot, action_dim in zip(self.robots, self.robot_action_dims):
                jointVelocity = self._deflatten_robot_actions(robot, 
                    substep_actions[dim_counter : dim_counter + action_dim])
                dim_counter += action_dim
                robot.link_fk_diff(jointVelocity)
            self.current_step += 1

    def _forward_kinematics(self, step_idx: int):
        # inherited from Controller
        robot_idx = 0
        for robot, primitive_dict in zip(self.robots, self.link_2_primitives):
            for name, link in robot._link_map.items():
                if name not in primitive_dict: continue
                pose = link.collision_pose(0, step_idx)
                primitive_dict[name].apply_robot_forward_kinemtaics(step_idx, pose)
                
                if self.is_recording():
                    pose = link.trajectory[step_idx]
                    for vis_idx, vis in enumerate(link.visuals):
                        vis_name = f'{robot.name}_{robot_idx}/{name}_{vis_idx}'
                        if vis.geometry.mesh is not None:
                            vis_name = vis_name + '.' + os.path.splitext(vis.geometry.mesh.filename)[-1]
                        UpdateRigidBodyPoseMessage(vis_name, pose.detach().cpu().numpy(), step_idx * self.STEP_INTERVAL).send()
            robot_idx += 1
                    

    def _forward_kinematics_grad(self, step_idx: int):
        # inherited from Controller
        for robot, primitive_dict in zip(reversed(self.robots), reversed(self.link_2_primitives)):
            for name, link in reversed(robot._link_map.items()):
                if name not in primitive_dict: continue
                primitive_dict[name].apply_robot_forward_kinemtaics.grad(
                    self     = primitive_dict[name],
                    frameIdx = step_idx, 
                    xyz_quat = link.trajectory[step_idx]
                )

    def get_step_grad(self, s: int) -> torch.Tensor:
        # inherited from Controller
        actionGrad = []
        for robot in self.robots:
            for grad in robot.fk_gradient(s + 1):
                actionGrad.append(grad)
        return torch.cat(actionGrad)
