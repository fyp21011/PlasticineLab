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
from typing import Any, Callable, Dict, List, Tuple, Union
import numpy as np

import open3d as o3d

from plb.urdfpy import Robot
from plb.urdfpy.link import Link
from plb.urdfpy.manipulation import FK_CFG_Type

FRAME_RATE = 1
MESH_TYPE = o3d.geometry.TriangleMesh
VISUALIZER_TYPE = o3d.visualization.Visualizer


class ActionParser(cmd.Cmd):
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
    names: a list of joints' names, which will used for tab auto-completion
    callback: the method to be invoked when the parsed action sequenced is
        expected to be executed, receiving two actions: actionDict, state
    initialState: the initial state for callback
    path2Action: file storing the action sequences
    """
    def __init__(self, names: List[str], callback: Callable[[Dict, Dict], Dict], initialState: Dict = None, path2Action: str = '') -> None:
        cmd.Cmd.__init__(self)

        self._names = names
        self._action_list: List[Dict] = [{}]
        self._callback = callback
        self._currentState = initialState
        self._cursor = 0
        if len(path2Action) != 0:
            self.do_LOAD(path2Action)
        self.do_help(None)

    @property
    def promote(self) -> str:
        return f"CURSOR @ {self._cursor} >"

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
        self._currentState = self._callback(self._action_list[self._cursor], self._currentState)

    def do_LOAD(self, path: str):
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
        while self._cursor < len(self._action_list):
            self._render_current_actions()
            self._cursor += 1
        self._action_list.append({})

    def do_ACT(self, arg: str): 
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
        self._render_current_actions()
        self._cursor += 1
        if len(self._action_list) == self._cursor:
            self._action_list.append({})

    def do_SAVE(self, arg: str):
        if arg:
            if len(self._action_list[-1]) == 0:
                self._action_list.pop()
            with open(arg, 'wb') as f:
                pickle.dump(self._action_list, f)
                print(f'the current action list is saved into {f}')
    
    def do_EXIT(self, arg: str): 
        exit(0)

    def complete_ACT(self, text: str, line: str, start_index, end_index):
        if text: 
            return [name for name in self._names if name.startswith(text)]
        else:
            return self._names


def _visualize_new_frame(
    vis:       VISUALIZER_TYPE,
    linkFk:    Dict[Link, Any],
    lastPoses: Union[None, Dict[str, Tuple[MESH_TYPE, np.ndarray]]] = None
) -> Dict[str, Tuple[MESH_TYPE, np.ndarray]]:
    """ Update the visualizer for a new frame

    Parameters
    ----------
    vis: the visualizer
    linkFk: the forward kinematics result of the current frame
    lastPoses: the returned value of THIS method at the previous
        frame. Set to None if this is the first frame.

    Returns
    -------
    Mapping from link names to the mesh that is currently
    renderred by the visualizer, together with its poses
    """
    if lastPoses is None:
        lastPoses = {}
    for link in linkFk:
        if link.collision_mesh is None:
            continue
        if link.name not in lastPoses:
            cm = link.collision_mesh
            vis.add_geometry(cm)
        else:
            cm, oldPose = lastPoses[link.name]
            cm.transform(np.linalg.inv(oldPose))
        pose = linkFk[link]
        cm.transform(pose)
        lastPoses[link.name] = (cm, pose)
        cm.compute_vertex_normals()
        vis.update_geometry(cm)
    vis.poll_events()
    vis.update_renderer()
    return lastPoses

def render_robot_animation(robot: Robot, visualizer: VISUALIZER_TYPE, actions: Dict[str, Any], currentPose: Dict[str, Any]):
    currentPose = _visualize_new_frame(
        visualizer, 
        robot.link_fk(
            cfg     = actions,
            cfgType = FK_CFG_Type.velocity
        ),
        lastPoses = currentPose
    )
    time.sleep(FRAME_RATE)
    return currentPose


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
            # init
            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window()
            currentPose = _visualize_new_frame(
                visualizer,
                robot.link_fk()
            )
            # run
            ActionParser(
                names        = list(robot.joint_map.keys()),
                callback     = render_robot_animation, 
                initialState = currentPose, 
                path2Action  = path4PickleLoad
            ).cmdloop()
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
    main(args.path, args.a, args.nogui, args.actions)
