import numpy as np
import pytest
import open3d as o3d

from plb.urdfpy import Robot, Link, Joint, Transmission, Material
from plb.urdfpy.manipulation import FK_CFG_Type


def test_urdfpy(tmpdir):
    outfn = tmpdir.mkdir('urdf').join('ur5.urdf').strpath

    # Load
    u = Robot.load('tests/data/ur5/ur5.urdf')

    assert isinstance(u, Robot)
    for j in u.joints:
        assert isinstance(j, Joint)
    for l in u.links:
        assert isinstance(l, Link)
    for t in u.transmissions:
        assert isinstance(t, Transmission)
    for m in u.materials:
        assert isinstance(m, Material)

    # Test fk
    fk = u.link_fk()
    assert isinstance(fk, dict)
    for l in fk:
        assert isinstance(l, Link)
        assert isinstance(fk[l], np.ndarray)
        assert fk[l].shape == (4,4)

    fk = u.link_fk({'shoulder_pan_joint': 2.0})
    assert isinstance(fk, dict)
    for l in fk:
        assert isinstance(l, Link)
        assert isinstance(fk[l], np.ndarray)
        assert fk[l].shape == (4,4)

    fk = u.link_fk(np.zeros(6))
    assert isinstance(fk, dict)
    for l in fk:
        assert isinstance(l, Link)
        assert isinstance(fk[l], np.ndarray)
        assert fk[l].shape == (4,4)

    fk = u.link_fk(np.zeros(6), link='upper_arm_link')
    assert isinstance(fk, np.ndarray)
    assert fk.shape == (4,4)

    fk = u.link_fk(links=['shoulder_link', 'upper_arm_link'])
    assert isinstance(fk, dict)
    assert len(fk) == 2
    for l in fk:
        assert isinstance(l, Link)
        assert isinstance(fk[l], np.ndarray)
        assert fk[l].shape == (4,4)

    fk = u.link_fk(links=list(u.links)[:2])
    assert isinstance(fk, dict)
    assert len(fk) == 2
    for l in fk:
        assert isinstance(l, Link)
        assert isinstance(fk[l], np.ndarray)
        assert fk[l].shape == (4,4)

    cfg={j.name: 0.5 for j in u.actuated_joints}
    for _ in range(1000):
        fk = u.collision_mesh_fk(cfg=cfg)
        for key in fk:
            assert isinstance(fk[key], np.ndarray)
            assert fk[key].shape == (4,4)


    cfg={j.name: 0.5 for j in u.actuated_joints}
    for _ in range(1000):
        fk = u.collision_mesh_fk(cfg=cfg)
        for key in fk:
            assert isinstance(key, o3d.geometry.TriangleMesh)
            assert fk[key].shape == (4,4)

    # Test scale
    x = u.copy(scale=3)
    assert isinstance(x, Robot)
    x = x.copy(scale=[1,1,3])
    assert isinstance(x, Robot)

def test_velocity_fk():
    u = Robot.load('tests/data/ur5/ur5.urdf')
    u.link_fk({'shoulder_pan_joint': 0.5}, cfgType = FK_CFG_Type.velocity)
    u.link_fk({'shoulder_pan_joint': 0.5}, cfgType = FK_CFG_Type.velocity)
    u.link_fk({'shoulder_pan_joint': 0.5}, cfgType = FK_CFG_Type.velocity)
    lastFk = u.link_fk({'shoulder_pan_joint': 0.5}, cfgType = FK_CFG_Type.velocity)
    trueFk = u.link_fk({'shoulder_pan_joint': 2}, cfgType = FK_CFG_Type.angle)
    assert isinstance(trueFk, dict)
    assert isinstance(lastFk, dict)
    for l in lastFk:
        assert isinstance(l, Link)
        assert l in trueFk
        assert isinstance(lastFk[l], np.ndarray)
        assert isinstance(trueFk[l], np.ndarray)
        assert lastFk[l].shape == (4,4)
        assert trueFk[l].shape == (4,4)
        assert np.all(trueFk[l] == lastFk[l]), \
            f"ERROR: final position after integration of velocity "+ \
            f"is {lastFk[l]}; expecting: {trueFk[l]}"