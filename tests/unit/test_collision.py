import taichi as ti
from plb.engine.primitive.primitive import Sphere
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

def test_robot_self_collsion():
    rc = RobotsController()
    robot = DiffRobot.load('tests/data/ur5/ur5_primitive.urdf')
    rc.append_robot(robot)
    collision_primitives = rc.link_2_primitives[0]

    collision_env = PrimitiveCollisionManager(0, collision_primitives)
    # TODO: how is robot primitives initiailzed?
    collided, contacts = collision_env.check_robot_collision()
    
    print(collided, contacts)
    assert not collided, f'Expected no contacts, but got {contacts}'
