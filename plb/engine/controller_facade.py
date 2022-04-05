from plb.engine.controller.primitive_controller import PrimitivesController
from plb.engine.controller.robot_controller import RobotsController


class ControllersFacade():
    """ Facade of different types of controllers

    A controller manages multiple objects, accecpting an action
    array everytime and applying the actions onto the objects.
    Thus, a controller should implement the forward kinematics
    as well as the corresponding gradient function. For example,
    a robot controller manages 3 robots, with 6, 8 and 3 DoF
    repectively. A primitive controller may manage 2 balls which
    can be freely moved in the space, so each has 6 DoF. 
    
    A controller facade aggregate one robot controller and one
    primitive controller, summing their action dimensions. 
    In the aforesaid example, the facade will have
    6 + 8 + 3 + 6 + 6 = 29 action dimensions. Thus, the `env`
    does not need to know which action goes to which robot 
    or primitive, nor how the actions drive the system through
    forward kinematics. The facade distributed the actions to
    each controllers which thereafter execute forward kinematics
    algorithms. 
    """

    def __init__(self):
        super().__init__()
        self.fpc = None
        self.rc = None
        self.accu_action_dims = [0] # accumulative action dimentions

    def register_controllers(self, primitives_controller: PrimitivesController=None, 
                             robots_controller: RobotsController=None):
        """ Register controllers

        Params
        ------
        primitives_controller: a `plb.engine.controller.PrimitivesController`
            which manages ALL the free primitives (i.e., primitives does not
            belong to some robots)
        robots_controller: a `plb.engine.controller.RobotsControllers`, which
            manages ALL the robots in one environment. 
        """
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
        """ True if the facade facade is not dummy
        """
        return (self.fpc != None) or (self.rc != None)

    @property
    def action_dim(self):
        """ The total number of actions that
        the controller expects
        """
        return self.accu_action_dims[-1]
    
    def set_action(self, s, n_substeps, action):
        """ Distribute the actions to controllers
        """
        assert len(action) == self.action_dim, f"Expected action length to be {self.action_dim}, but got {len(action)}"
        if self.fpc != None:
            fpc_action = action[self.fpc_action_range[0]:self.fpc_action_range[1]]
            self.fpc.set_action(s, n_substeps, fpc_action)
        if self.rc != None:
            rc_action = action[self.rc_action_range[0]:self.rc_action_range[1]]
            self.rc.set_action(s, n_substeps, rc_action)