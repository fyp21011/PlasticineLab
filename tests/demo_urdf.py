""" Functional tests for the URDF module

Usage
---
python3 functional_test.py [URDF path] [options]

Options
---
-c       use the collision mesh (otherwise the
         visualized mesh is used for renderring)
-a       animate the URDF (not supported in the
         current Open3D version)
--nogui  do not display any graphic output, but
         conducting analytical analysis
--action specify the file from which the animation
         actions is loaded
"""
import argparse
import copy
import cmd
import os
import pickle
import random
import time
from typing import Dict, List
import numpy as np

import open3d as o3d

from plb.urdfpy import Robot
from plb.urdfpy.manipulation import FK_CFG_Type

FRAME_RATE = 24
MESH_TYPE = o3d.geometry.TriangleMesh
VISUALIZER_TYPE = o3d.visualization.Visualizer


class Animator(cmd.Cmd):
    """ Parser the user's input to get the actions for the demo robot
    * LOAD [path]
        load a pre-saved action sequence from `path`
    * RUN
        run all through all the timesteps to make an animation
    * EXIT

    Parameters
    ----------
    robot: the URDF loaded robot
    """
    def __init__(self, robot: Robot) -> None:

        cmd.Cmd.__init__(self)
        self._robot = robot
        self._action_list: List[Dict] = []
        self._currentPose = None
        self._visualizer = o3d.visualization.Visualizer()
        self._cursor = 0
        self._smoothedVelocity: Dict[str, float] = {}

    def do_help(self, arg: str):
        print("Usage: ")
        print("* load [path]\n\tload a pre-saved action sequence from `path`")
        print("* run\n\trun all through all the timesteps to make an animation")
        print("* exit")

    def _render_current_actions(self, currentActions: Dict, isReset = False):
        """ Render the current action

        The "current" action is defined as the one pointed by self._cursor
        """
        if self._cursor == 0:
            self._currentPose = {}
            self._visualizer.create_window()
        if isReset:
            linkFk = self._robot.link_fk(cfg = currentActions, cfgType = FK_CFG_Type.angle)
        else:
            self._smoothedVelocity = {
                key: (currentActions[key] * (3 / FRAME_RATE) + self._smoothedVelocity.get(key, 0)) / 4
                for key in currentActions
            }
            linkFk = self._robot.link_fk(cfg = self._smoothedVelocity, cfgType = FK_CFG_Type.velocity)
        for link in linkFk:
            if link.collision_mesh is None:
                continue
            if link.name not in self._currentPose:
                cm = link.collision_mesh
                self._visualizer.add_geometry(cm)
            else:
                cm, oldPose = self._currentPose[link.name]
                cm.transform(np.linalg.inv(oldPose))
            pose = linkFk[link]
            cm.transform(pose)
            self._currentPose[link.name] = (cm, pose)
            cm.compute_vertex_normals()
            self._visualizer.update_geometry(cm)
        self._visualizer.poll_events()
        self._visualizer.update_renderer()

    def do_run(self, args: str):
        print("use ctrl-c to shut the animation")
        n = len(self._action_list)
        if n <= 0:
            print("no actions loaded")
            return
        try:
            while True:
                currentSec = self._cursor // FRAME_RATE
                initState  = currentSec % FRAME_RATE == 0
                print(f'renderring {currentSec}s', end = '\r')
                actions = self._action_list[currentSec % n]
                self._render_current_actions(actions, isReset=initState)
                self._cursor += FRAME_RATE if initState else 1
                time.sleep(1 / FRAME_RATE / 10)
        except KeyboardInterrupt:
            print('renderring has been finished')
            return

    def do_load(self, path: str):
        """ Callback for LOAD [path] command

        Load an action sequence pickle file

        Parameter
        ---------
        path: the commandline argument, specifying the file path
        """
        if not os.path.exists(path):
            print(f"NO SUCH FILE: {path}")
        else:
            with open(path, 'rb') as f:
                newActions = pickle.load(f)
                print(f'ACTIONS LOADED')
                if not isinstance(newActions, list):
                    raise ValueError(f'The loaded action_list expected to be a list, but a {type(self._action_list)} received')
                self._action_list.extend(newActions)
            if len(self._action_list) == 0:
                raise ValueError(f'Empty action file {path}')
    
    def do_exit(self, arg: str): 
        exit(0)


def show_static_robot(robot: Robot) -> None:
    """ Visualize a robot statically using random pose

    Paramters
    ---------
    robot: the loaded robot object
    """
    robotFk = robot.visual_mesh_fk({
        joint.name: random.uniform(0.0, 3.1415926)
        for joint in robot.joints
    })
    meshes = []
    for eachMesh in robotFk:
        visualMesh = copy.deepcopy(eachMesh)
        # We cannot modify the mesh in the robotFk dict
        # they are the reference to the origin ones
        visualMesh.transform(robotFk[eachMesh])
        visualMesh.compute_vertex_normals()
        meshes.append(visualMesh)
    o3d.visualization.draw_geometries(meshes)


def main(path:str, animate:bool=True, nogui:bool=False):
    robot = Robot.load(path)
    if not nogui:
        if animate:
            Animator(robot).cmdloop()
        else:
            show_static_robot(robot)
    else:
        cFk = robot.collision_mesh_fk()
        for mesh in cFk:
            assert mesh.has_triangles(), f"{mesh} has no triangles"
            print(mesh)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path',
        type=str,
        help='Path to URDF file that describes the robot'
    )
    parser.add_argument(
        '-a',
        '--animation',
        action='store_true',
        help='Visualize robot animation'
    )
    noGui = parser.add_argument(
        '--nogui',
        action='store_true',
        help='skip the GUI renderring, for machines with no display'
    )
    args = parser.parse_args()

    if args.nogui and args.animation:
        raise argparse.ArgumentError(noGui, "no gui cannot be set when -a (animation) is given")
    main(args.path, args.animation, args.nogui)
