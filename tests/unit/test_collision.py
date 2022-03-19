import taichi as ti
from plb.engine.primitive.primitives import Sphere
from plb.engine.collision_manager import PrimitiveCollisionManager
from plb.engine.robots import RobotsController
from plb.urdfpy import DiffRobot, Link, FK_CFG_Type, Mesh

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
    
    assert not collision_env.check_robot_collision()

def test_robot_self_collsion():
    rc = RobotsController()
    robot = DiffRobot.load('tests/data/ur5/ur5_primitive.urdf')
    collision_primitives = list(rc.append_robot(robot))
    collision_env = PrimitiveCollisionManager(0, collision_primitives)
    
    # TODO: how is robot primitives initiailzed?
    assert collision_env.check_robot_collision()



