import pickle
from time import sleep

import torch

from plb.urdfpy import DiffRobot
from plb.urdfpy.diff_fk import _tensor_creator
from plb.utils import VisRecordable

robot = DiffRobot.load('tests/data/ur5/ur5.urdf')
with open('tests/data/ur5/real.pkl', 'rb') as f:
    actions = pickle.load(f)

VisRecordable.register_scene_init_callback(lambda : print("TURNED ON ONCE"))
VisRecordable.whether_on = lambda : True

for a in actions:
    joint_action = []
    for joint in robot._actuated_joints:
        joint_action.append(_tensor_creator(torch.tensor, a[joint.name]))
    robot.link_fk_diff(joint_action)
    sleep(1)


