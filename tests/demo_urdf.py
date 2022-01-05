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

FRAME_RATE = 1/24
MESH_TYPE = o3d.geometry.TriangleMesh
VISUALIZER_TYPE = o3d.visualization.Visualizer


class Animator(cmd.Cmd):
    """ Parser the user's input to get the actions for the demo robot

    * LOAD [path]
        load a pre-saved action sequence from `path`
    * ACT [joint_name] [values ...]
        add an action for a specified joint at the current timestep;
        use TAB to find the joint
    * NEXT
        proceed the current actions, and move to the next timestep
    * RUN
        run all through all the timesteps to make an animation
    * SAVE [path]
        save the entire action sequence to `path`
    * EXIT

    Parameters
    ----------
    names: a list of joints' names, which will used for tab auto-completionk
    path2Action: file storing the action sequences
    """
    def __init__(self, robot: Robot, path2Action: str = '') -> None:
        cmd.Cmd.__init__(self)

        self._names = list(robot.joint_map.keys())
        self._robot = robot
        self._action_list: List[Dict] = [{}]
        self._currentPose = None
        self._cursor = 0
        self._visualizer = None
        if len(path2Action) != 0:
            self.do_LOAD(path2Action)
            self.do_RESET(None)
            self.do_RUN(None)
            self.do_EXIT(None)
        else:
            self.do_help(None)

    def do_RESET(self, arg: str) -> None:
        """ Reset the FK after an action sequence is run
        """
        if self._visualizer:
            self._visualizer.clear_geometries()
            self._visualizer.destroy_window()
        self._visualizer = o3d.visualization.Visualizer()
        self._currentPose = None
        self._cursor = 0
    
    @property
    def prompt(self) -> str:
        return f"cursor @ {self._cursor} > "

    def preloop(self) -> None:
        self.do_RESET(None)
        return super().preloop()

    def do_help(self, arg: str):
        print("Usage: ")
        print("* LOAD [path]\n\tload a pre-saved action sequence from `path`")
        print("* ACT [joint_name] [values ...]\n\tadd action for a specified joint at the current timestep; use TAB to find the joint")
        print("* NEXT\n\tproceed the current action, and move to the next timestep")
        print("* RUN\n\trun all through all the timesteps to make an animation")
        print("* SAVE [path]\n\tsave the entire action sequence to path")
        print("* EXIT")

    def do_HELP(self, arg: str):
        return self.do_help(arg)

    def _render_current_actions(self):
        """ Render the current action

        The "current" action is defined as the one pointed by self._cursor
        """
        if self._currentPose is None:
            self._currentPose = {}
            self._visualizer.create_window()
        linkFk = self._robot.link_fk(cfg = self._action_list[self._cursor], cfgType = FK_CFG_Type.velocity)
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
        self._cursor += 1
        time.sleep(FRAME_RATE)

    def do_LOAD(self, path: str):
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
                self._action_list = pickle.load(f)
            print(f'ACTIONS LOADED')
            if not isinstance(self._action_list, list):
                raise ValueError(f'The loaded action_list expected to be a list, but a {type(self._action_list)} received')
            if len(self._action_list) == 0:
                raise ValueError(f'Empty action file {path}')

    def do_RUN(self, arg: str):
        """ Run the action sequence from the current cursor
        """
        while self._cursor < len(self._action_list):
            self._render_current_actions()
        self.do_RESET(None)

    def do_ACT(self, arg: str): 
        """ Callback to the ACT [joint name] [values...] command

        Add a joint-name to velocities pair to the current timestep

        Parameter
        ---------
        arg: the text argument, in form of joint-name, values...
            If there is only one value, the value will be parsed
            to float; otherwise, a numpy array
        """
        arg = arg.replace(', ', ' ')
        arg = arg.replace(',', ' ')
        words = arg.strip().split(' ')
        if len(words) <= 1:
            return
        jointName = words[0]
        if len(words) == 2:
            floatAction = float(words[1].strip())
        else:
            floatAction = []
            for word in words[1:]:
                floatAction.append(float(word.strip()))
            floatAction = np.array(floatAction)

        self._action_list[self._cursor][jointName] = floatAction
    
    def do_NEXT(self, arg: str):
        """ Render ONE frame and forward the cursor
        """
        self._render_current_actions()
        if len(self._action_list) == self._cursor:
            self._action_list.append({})

    def do_SAVE(self, arg: str):
        """ Save the current action sequence into a pickle file

        Parameter
        ---------
        arg: where the sequence is expected to be saved 
        """
        if arg:
            if len(self._action_list[-1]) == 0:
                self._action_list.pop()
            with open(arg, 'wb') as f:
                pickle.dump(self._action_list, f)
                print(f'the current action list is saved into {f}')
    
    def do_EXIT(self, arg: str): 
        exit(0)

    def complete_ACT(self, text: str, line: str, start_index, end_index):
        """ Tab auto-completion for robot's joint name
        """
        if text: 
            return [name for name in self._names if name.startswith(text)]
        else:
            return self._names

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


def main(path:str, animate:bool=True, nogui:bool=False, path4PickleLoad: str = ''):
    robot = Robot.load(path)
    if not nogui:
        if animate:
            a = Animator(
                robot       = robot,
                path2Action = path4PickleLoad
            )
            a.cmdloop()
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
    parser.add_argument(
        '--actions',
        type=str,
        help='path to actions file',
        default=''
    )
    noGui = parser.add_argument(
        '--nogui',
        action='store_true',
        help='skip the GUI renderring, for machines with no display'
    )
    args = parser.parse_args()

    if args.nogui and args.a:
        raise argparse.ArgumentError(noGui, "no gui cannot be set when -a (animation) is given")
    main(args.path, args.animation or len(args.actions) != 0, args.nogui, args.actions)
