import pickle
from time import sleep

import torch
import numpy as np

from plb.envs import PlasticineEnv
from plb.urdfpy import DiffRobot
from plb.urdfpy.diff_fk import _tensor_creator
from plb.utils import VisRecordable
from protocol import FinishAnimationMessage

# robot = DiffRobot.load('tests/data/ur5/ur5.urdf')
# with open('tests/data/ur5/real.pkl', 'rb') as f:
#     actions = pickle.load(f)

# VisRecordable.register_scene_init_callback(lambda : print("TURNED ON ONCE"))
# VisRecordable.whether_on = lambda : True

# for a in actions:
#     joint_action = []
#     for joint in robot._actuated_joints:
#         joint_action.append(_tensor_creator(torch.tensor, a[joint.name]))
#     robot.link_fk_diff(joint_action)
#     sleep(1)

# last = VisRecordable.current_frame_idx() + 1
# print("last frame idx is ", last)
# FinishAnimationMessage("unit_test_vis_recording", last).send()

TIME = 0

def done():
    FinishAnimationMessage("rope_robot_init", TIME * VisRecordable.STEP_INTERVAL).send()
    

env = PlasticineEnv(cfg_path = 'rope_robot.yml', version = 1)
assert env.action_space.shape == (12,)
VisRecordable.register_scene_end_callback(done)
VisRecordable.turn_on()
# env.step(np.zeros((12,)))
# for i in range(14):
#     env.step(np.random.rand(12, ) * 0.1)
#     sleep(0.1)
VisRecordable.turn_off()

