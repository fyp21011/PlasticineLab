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
import random
import time
from typing import Any, Dict, List, Tuple, Union
import numpy as np

import open3d as o3d
from plb.urdfpy import Robot
from plb.urdfpy.link import Link
from plb.urdfpy.manipulation import FK_CFG_Type

FRAME_RATE = 1/24

MESH_TYPE = o3d.geometry.TriangleMesh


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
            show_dynamic_robot(robot,) #TODO
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
