from plb.envs import PlasticineEnv
from plb.engine.taichi_env import TaichiEnv

def test_load_robot_env():
    cfg = PlasticineEnv.load_varaints('rope_robot.yml', 1)
    tcEnv = TaichiEnv(cfg, False)
    assert len(tcEnv.primitives.action_dims) == 5
    assert tcEnv.primitives.action_dim == 12
    assert len(tcEnv.robots_controller.robots) == 1
    assert len(tcEnv.robots_controller.robots[0].link_map.keys()) == 11
