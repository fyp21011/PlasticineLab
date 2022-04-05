from plb.engine.controller import PrimitivesController
from plb.engine.controller import RobotsController


class PrimitivesFacade():
    """ A facade to handle all primitives

    In the simulator, primitives can come from two sources:
    a) free primitives, which are controlled by the 
        `plb.engine.controller.PrimitivesController`
    b) robots, each of which may consist of multiple
        primitives; all the robots are controlled through
        `plb.engine.controller.RobotsController`

    All the primitives coming from distinct sources share
    common physics and rigid-body properties. Thus, the
    facade provides a uniformed interface for simulation
    and collision handling. 
    """
    def __init__(self):
        super().__init__()
        self.primitives = []

    def register_free_primitives(self, primitives_controller: PrimitivesController):
        """ Register the free primitives from `PrimitivesController`

        params
        ------
        primitives_controller: a controller for all the free
            primitives in the environment
        """
        self.primitives += primitives_controller.primitives

    def register_robot_primitives(self, robots_controller: RobotsController):
        """ Register the primitives affiliated to all the robots

        Params
        ------
        robots_controller: a controller for all the robots
            in the environment, from which each robot's 
            primitive will be appended to the `self.primitives`
        """
        for mapping_per_robot in robots_controller.link_2_primitives:
            self.primitives += mapping_per_robot.values()

    @property
    def not_empty(self) -> bool:
        """ True is the facade is not dummy
        """
        return len(self.primitives) > 0

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