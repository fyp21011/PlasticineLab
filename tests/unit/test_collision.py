import taichi as ti
from plb.engine.primitive.primitive import Cylinder, Sphere
from plb.engine.collision_manager import PrimitiveCollisionManager
from plb.engine.controller.robot_controller import RobotsController
from plb.urdfpy import DiffRobot

ti.init()

def test_sphere_collision_true():
    s1 = Sphere(cfg=Sphere.default_config())
    
    s2_cfg = Sphere.default_config()
    s2_cfg.radius = 2.
    s2_cfg.init_pos = (2.3, 0.3, 0.3)
    s2 = Sphere(cfg=s2_cfg)
    
    primitives = [s1, s2]
    
    for p in primitives:
        p.initialize()
    
    collision_env = PrimitiveCollisionManager(0, primitives)
    
    assert collision_env.check_robot_collision()

def test_sphere_collision_false():
    s1 = Sphere(cfg=Sphere.default_config())
    
    s2_cfg = Sphere.default_config()
    s2_cfg.radius = 2.
    s2_cfg.init_pos = (3.4, 0.3, 0.3)
    s2 = Sphere(cfg=s2_cfg)
    
    primitives = [s1, s2]
    
    for p in primitives:
        p.initialize()
    
    collision_env = PrimitiveCollisionManager(0, primitives)
    collided, _ = collision_env.check_robot_collision()

    assert not collided, f'Expected no collision but collided is {collided}'

def test_wrist_collision():
    wrist_1_cfg = Cylinder.default_config()
    wrist_1_cfg.h = 0.105
    wrist_1_cfg.init_pos = (0.10715, 0.14665900000400173, 0.8172499999999511)
    wrist_1_cfg.init_rot = (0.0, 1.0, 0.0, 0.0)
    wrist_1_cfg.r = 0.045

    wrist_2_cfg = Cylinder.default_config()
    wrist_2_cfg.h = 0.11
    wrist_2_cfg.init_pos = (0.1641499825612509, -0.005797202026907934, 0.8172500000009304)
    wrist_2_cfg.init_rot = (0.0, 0.7073882691671998, -0.706825181105366, 0.0)
    wrist_2_cfg.r = 0.045

    wrist_3_cfg = Cylinder.default_config()
    wrist_3_cfg.h = 0.03
    wrist_3_cfg.init_pos = (0.1961499952439775, -0.005479055095337266, 0.817250000000927)
    wrist_3_cfg.init_rot = (0.0, 0.7073882691671998, -0.706825181105366, 0.0)
    wrist_3_cfg.r = 0.04

    primitives = {
        'wrist_1': Cylinder(cfg = wrist_1_cfg),
        'wrist_2': Cylinder(cfg = wrist_2_cfg),
        'wrist_3': Cylinder(cfg = wrist_3_cfg)
    }
    for p in primitives.values():
        p.initialize()
    
    collision_env = PrimitiveCollisionManager(0, primitives)
    collided, contacts = collision_env.check_robot_collision()
    assert not collided, f'Expected no contacts, but got {contacts}'


def test_robot_self_collsion():
    rc = RobotsController()
    robot = DiffRobot.load('tests/data/ur5/ur5_primitive.urdf')
    rc.append_robot(robot)
    collision_primitives = rc.link_2_primitives[0]

    for p in collision_primitives.values():
        p.initialize()

    collision_env = PrimitiveCollisionManager(0, collision_primitives)
    collided, contacts = collision_env.check_robot_collision()
    
    print(collided, contacts)
    assert not collided, f'Expected no contacts, but got {contacts}'
