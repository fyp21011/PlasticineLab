from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Callable, Union

import numpy
import torch


class DiffFKWrapper:
    """ Wrapper of differentiable forward kinematics for `Controller`

    Two callable handlers, one for forward kinemtaics, one
    for the gradient backpropagation, are wrapped inside. 

    Params
    ------
    `fk_handler`: the handler for forward kinematics
    `fk_grad_handler`: the handler for gradient backpropagation of
        forward kinematics
    """
    def __init__(self, fk_handler: Callable[[int], None], fk_grad_handler: Callable[[int], None]) -> None:
        self._fk_handler = fk_handler
        self._grad_handler = fk_grad_handler

    def __call__(self, substep: int) -> None:
        """ Forward kinematics computation

        Params
        ------
        substep: the index of the substep when the forward
            kinematics is computed
        """
        if self._fk_handler != None:
            self._fk_handler(substep)

    def grad(self, substep: int) -> None:
        """ Backpropagate along the computation graph for
        forward kinematics

        Params
        ------
        substep: the index of the substep from which the
            backpropagation starts
        """
        if self._grad_handler != None:
            self._grad_handler(substep)

class Controller(ABC):
    @abstractproperty
    def forward_kinematics(self) -> DiffFKWrapper:
        """ For both forward kinematics and the backpropagation

        * `self.forward_kinematics(s)`: applies the FK till time `s`
        * `self.forward_kinematics.grad(s)`: backpropagation from time `s`
        """
        pass

    @abstractmethod
    def set_action(self, step: int, n_substep: int, action: Any) -> None:
        """ Set action for time [step * n_substep, (step + 1) * n_substep)

        Params
        ------
        step: step index
        n_substep: how many substeps one step contains
        action: the actions
        """
        pass

    @abstractmethod
    def get_step_grad(self, step: int) -> Union[torch.Tensor, numpy.ndarray, Any]:
        """ Get the gradient of a specified step
        
        Params
        ------
        step: index of step where the action gradient is concerned

        Returns
        -------
        Gradient, of type Tensor, NDArray or Taichi Field
        """
        pass
