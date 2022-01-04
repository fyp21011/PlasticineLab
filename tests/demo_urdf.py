""" Functional tests for the URDF module

Usage
---
python3 functional_test.py [URDF path] [options]

Options
---
-c      use the collision mesh (otherwise the
        visualized mesh is used for renderring)
-a      animate the URDF (not supported in the
        current Open3D version)
--nogui do not display any graphic output, but
        conducting analytical analysis
"""
import argparse
import copy
import cmd
import random
import time
from typing import Any, Callable, Dict, List, Tuple, Union
from typing_extensions import OrderedDict
import numpy as np

import open3d as o3d
from plb.urdfpy import Robot
from plb.urdfpy.link import Link
from plb.urdfpy.manipulation import FK_CFG_Type

FRAME_RATE = 1/24

MESH_TYPE = o3d.geometry.TriangleMesh


class ActionParser(cmd.Cmd):
    """ Parser the user's input to get the actions for the demo robot

    The parser expects three different inputs, in format of 
    'ACT [joint name] [values...]'
        to add a new action at the CURRENT timestep
    'NEXT'
        to finalized the current timestep and forward to the
        next timestep
    'RUN'
        execute the action sequence

    Parameters
    ----------
    names: a list of joints' names, which will used for tab auto-completion
    callback: the method to be invoked when the parsed action sequenced is
        expected to be executed
    """
    def __init__(self, names: List[str], callback: Callable[[List[Dict]], None]) -> None:
        cmd.Cmd.__init__(self)

        self._helpMsg = 'ACT [joint name] [values...] to add action, or \nNEXT to add timestep or \nRUN to run\n\nTry using tab to view the possible joint names'
        self._names = names
        self._action_list: List[Dict] = [{}]
        self._callback = callback

        self.do_help(None)

    def do_help(self, arg: str):
        print(self._helpMsg)

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

        self._action_list[-1][jointName] = floatAction
    
    def do_NEXT(self, arg: str):
        if len(self._action_list[-1]) == 0:
            print("THE LAST TIMESTEP IS EMPTY YET, NO NEW TIMESTEP BEING ADDED")
        else:
            self._action_list.append({})
            print(f"ADDING A NEW TIMESTEP, NOW TIMESTEP {len(self._action_list)}")

    def do_RUN(self, arg: str):
        if len(self._action_list[-1]) == 0:
            self._action_list.pop()
        if len(self._action_list) == 0:
            print('WARN: no action to be renderred')
            return
        self._callback(self._action_list)
        exit(0)

    def complete_ACT(self, text: str, line: str, start_index, end_index):
        if text: 
            return [name for name in self._names if name.startswith(text)]
        else:
            return self._names


def _visualize_new_frame(
    vis:       o3d.visualization.Visualizer,
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
        if link.name not in lastPoses:
            cm = link.collision_mesh
            vis.add_geometry(cm)
        else:
            cm, oldPose = lastPoses[link.name]
            cm.transform(np.linalg.inv(oldPose))
        pose = linkFk[link]
        cm.transform(pose)
        lastPoses[link.name] = (cm, pose)
        vis.update_geometry(cm)
    vis.update_renderer()
    return lastPoses

def show_dynamic_robot(robot: Robot, actionList: List[Dict[str, Any]]):
    """ Render the animation of the robot given a list of action configs

    Parameters
    ----------
    robot: the loaded URDF Robot
    actionList: a list of action configuration. Each element in the list
        corresponds to one timestep, which is a dict from Joint names to
        the velocity being applied onto the joints

    Returns
    -------
    None
    """
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    currentPose = _visualize_new_frame(
        visualizer,
        robot.link_fk()
    )
    for eachFrameActions in actionList:
        currentPose = _visualize_new_frame(
            visualizer, 
            robot.link_fk(
                cfg     = eachFrameActions,
                cfgType = FK_CFG_Type.velocity
            ),
            lastPoses = currentPose
        )
        time.sleep(FRAME_RATE)


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
            p = ActionParser(
                names    = list(robot.joint_map.keys()),
                callback = lambda actionsList : show_dynamic_robot(robot, actionsList)
            )
            p.cmdloop()
        else:
            show_static_robot(robot)
    else:
        cFk = robot.collision_mesh_fk()
        for mesh in cFk:
            assert mesh.has_triangles(), f"{mesh} has no triangles"
            print(mesh)

def test_cmd():
    robot = Robot.load('tests/data/ur5/ur5.urdf')
    def debug_cmd(args):
        for row in args: 
            print(row)
    p = ActionParser(names = list(robot.joint_map.keys()), callback = debug_cmd)
    p.cmdloop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path',
        type=str,
        help='Path to URDF file that describes the robot'
    )
    parser.add_argument(
        '-a',
        action='store_true',
        help='Visualize robot animation'
    )
    parser.add_argument(
        '-c',
        action='store_true',
        help='Use collision geometry'
    )
    noGui = parser.add_argument(
        '--nogui',
        action='store_true',
        help='skip the GUI renderring, for machines with no display'
    )
    args = parser.parse_args()

    if args.nogui and args.a:
        raise argparse.ArgumentError(noGui, "no gui cannot be set when -a (animation) is given")
    main(args.path, args.a, args.nogui)
