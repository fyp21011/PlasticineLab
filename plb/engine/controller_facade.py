from plb.engine.controller.primitive_controller import PrimitivesController
from plb.engine.controller.robot_controller import RobotsController


class ControllersFacade():
    def __init__(self):
        super().__init__()
        self.fpc = None
        self.rc = None
        self.accu_action_dims = [0] # accumulative action dimentions

    def register_controllers(self, primitives_controller: PrimitivesController=None, 
                             robots_controller: RobotsController=None):
        self.fpc = primitives_controller
        self.rc = robots_controller

        # free primitives accumulative DoFs
        if self.fpc != None:
            for primitive in self.fpc.primitives:
                self.accu_action_dims.append(self.accu_action_dims[-1] + primitive.action_dim)
        self.fpc_action_range = (0, self.accu_action_dims[-1])

        # robots accumulative DoFs
        if self.rc != None:
            for robot in self.rc.robots:
                robot_action_dim = sum(joint.action_dim for joint in robot.actuated_joints)
                self.accu_action_dims.append(self.accu_action_dims[-1] + robot_action_dim)
        self.rc_action_range = (self.fpc_action_range[1], self.accu_action_dims[-1])
        

    @property
    def not_empty(self) -> bool:
        return (self.fpc != None) or (self.rc != None)

    @property
    def action_dim(self):
        return self.accu_action_dims[-1]
    
    def set_action(self, s, n_substeps, action):
        if self.fpc != None:
            fpc_action = action[self.fpc_action_range[0]:self.fpc_action_range[1]]
            self.fpc.set_action(s, n_substeps, fpc_action)
        if self.rc != None:
            rc_action = action[self.rc_action_range[0]:self.rc_action_range[1]]
            self.rc.set_action(s, n_substeps, rc_action)