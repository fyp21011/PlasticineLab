from os import times
import pickle
from typing import Dict, List

import numpy

from plb.urdfpy import Robot
from plb.urdfpy.link import Link
from plb.urdfpy.manipulation import FK_CFG_Type

DEBUG = True

def _validate_actions(relativeSeq: List[Dict[str, float]], absoluteSeq:List[Dict[str, float]]) -> None: 
    assert len(relativeSeq) == len(absoluteSeq), \
        "The velocity and absolute action sequence differs in length"
    assert all(( 
        relativeSeq[0][key] == absoluteSeq[0][key] for key in relativeSeq[0]
    )), "The initial states of the absolute actions and the velocity actions are different"

    n = len(relativeSeq)
    cfg = {key:0 for key in relativeSeq[0].keys()}

    for timeCursor in range(0, n):
        deltaCfg = relativeSeq[timeCursor]
        for key in cfg:
            cfg[key] += deltaCfg.get(key, 0)
        trueCfg = absoluteSeq[timeCursor]
        for key in cfg: 
            if key in trueCfg:
                assert cfg[key] == trueCfg[key], \
                    f"ERROR: the integration of the JOINT:{key}'s velocity actions is not its" +\
                    f"absolute action\n\tExpecting {trueCfg[key]} after {timeCursor} timestep" +\
                    f", got {cfg[key]}"
            elif DEBUG:
                print(f"skipping {key} @ time-step={timeCursor}")

def _converting_keys(fkPose: Dict[Link, numpy.ndarray]) -> Dict[str, numpy.ndarray]:
    return {
        key.name: value for key, value in fkPose.items()
    }


def test_forward_kinematics():
    robotAbs = Robot.load('tests/data/ur5/ur5.urdf')
    robotVel = Robot.load('tests/data/ur5/ur5.urdf')

    with open('tests/data/ur5/real.pkl', 'rb') as reader:
        # positional actions
        posActions: List[Dict[str, float]] = pickle.load(reader)

    with open('tests/data/ur5/real_action.pkl', 'rb') as reader:
        # velocity-based actions
        velActions: List[Dict[str, float]] = pickle.load(reader)

    # first, verity the absActions is truely the intergrated result of velActions
    _validate_actions(velActions, posActions)

    totalTimeStep = len(posActions)
    for timeStep in range(0, totalTimeStep):
        velLinkPose = _converting_keys(
            robotVel.link_fk(velActions[timeStep], cfgType=FK_CFG_Type.velocity)
        )
        posLinkPose = _converting_keys(
            robotAbs.link_fk(posActions[timeStep])
        )
        for key in velLinkPose.keys():
            assert numpy.isclose(velLinkPose[key], posLinkPose[key], rtol=1e-3).all(), \
                f'velLinkPose[{key}] != posLinkPose[{key}] at time step {timeStep}'

        currentVelCfg = _converting_keys(robotVel._current_cfg)
        currentAbsCfg = _converting_keys(robotAbs._current_cfg)
        for key in currentVelCfg:
            assert numpy.isclose(currentVelCfg[key], currentAbsCfg[key], rtol=1e-3), \
                f'robotVel[{key}]({currentVelCfg[key]}) ' +\
                f'!= robotAbs[{key}]({currentAbsCfg[key]})' +\
                f' at time step {timeStep}'
        if DEBUG and 'shoulder_link' in velLinkPose and 'shoulder_link' in posLinkPose:
            print(f"{timeStep}:\n{velLinkPose['shoulder_link']}\nv.s.\n{posLinkPose['shoulder_link']}\n========")
    
    

