from plb.engine.controller.primitive_controller import PrimitivesController
from plb.engine.controller.robot_controller import RobotsController


class PrimitivesManager():
    """Please register free primitives before robots"""
    def __init__(self):
        super().__init__()
        self.primitives = []
        self.action_dims = [0]

    def register_free_primitives(self, primitives_controller: PrimitivesController):
        self.primitives += primitives_controller.primitives
        for primitive in primitives_controller.primitives:
            self.action_dims.append(self.action_dims[-1] + primitive.action_dim)

    def register_robot_primitives(self, robots_controller: RobotsController):
        for mapping_per_robot in robots_controller.link_2_primitives:
            self.primitives += mapping_per_robot.values()
        for dims in robots_controller.robot_action_dims:
            self.action_dims.append(self.action_dims[-1] + dims)

    @property
    def not_empty(self) -> bool:
        return len(self.primitives) > 0

    @property
    def action_dim(self):
        return self.action_dims[-1]

    @property
    def state_dim(self):
        return sum(i.state_dim for i in self.primitives)
    
    def set_softness(self, softness=666.):
        for i in self.primitives:
            i.softness[None] = softness

    def get_softness(self):
        return self.primitives[0].softness[None]

    def __getitem__(self, item):
        if isinstance(item, tuple):
            item = item[0]
        return self.primitives[item]

    def __len__(self):
        return len(self.primitives)

    def initialize(self):
        for i in self.primitives:
            i.initialize()