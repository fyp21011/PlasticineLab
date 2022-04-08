from typing import Callable, List
import warnings

import numpy as np




class VisRecordable:
    """ An interface, marking a class is visually
    recordable and offer some utils method

    Some classes' instances in the plb represent
    or contains physical objects in the simulator,
    such as `Primitives`, `DiffRobot`, etc. Thus, 
    the movements or the deformations of objects
    may worth visually recording, such as in an
    animation form. The simulated world will be
    recorded into a scene, where the animation take
    place. 
    """
    _scene_init_callbacks: List[Callable] = []
    _scene_end_callbacks: List[Callable] = []
    _on = False

    STEP_INTERVAL = 24

    @classmethod
    def turn_on(cls):
        if not cls._on:
            cls._invoke_init_callbacks()
        cls._on = True
    
    @classmethod
    def turn_off(cls):
        if cls._on:
            cls._invoke_end_callbacks()
        cls._on = False
        
    
    @classmethod
    def register_scene_init_callback(cls, init_callback: Callable[[], None]) -> None:
        """ Register a callback when the recorder is turned on

        The callback is expected to conduct the initialization
        stuff of the scene where the recoder recode everything
        in the engine. 
        """
        cls._scene_init_callbacks.append(init_callback)
        if cls._on:
            init_callback()
        
    @classmethod
    def register_scene_end_callback(cls, end_callback: Callable[[], None]) -> None:
        """ Register a callback when the recorder is turned off
        
        The callback is expected mark the end of the scene and 
        shoulder some cleaning-up work, such as saving the
        scene. 
        """
        cls._scene_end_callbacks.append(end_callback)

    @classmethod
    def remove_callback(cls, callback: Callable[[], None]) -> None:
        """ Remove a previously registered callback, either as
        a scene_init_callback or a scene_end_callback

        If the provided `callback` does not exist, a 
        warning will be posted.
        """
        for callback_group in (cls._scene_init_callbacks, cls._scene_end_callbacks):
            for i in range(len(callback_group)):
                if callback_group[i] == callback:
                    del callback_group[i]
                    return 
        warnings.warn("the callback is not registered")
        

    @classmethod
    def _invoke_init_callbacks(cls) -> None:
        for callback in cls._scene_init_callbacks:
            callback()

    @classmethod
    def _invoke_end_callbacks(cls) -> None:
        for callback in cls._scene_end_callbacks:
            callback()

    @classmethod
    def is_recording(cls) -> bool:
        """ Whether the recorder is ON

        Return
        ------
        True === recorder is recording
        False === recorder is off
        """
        return cls._on

    @classmethod
    def y_up_2_z_up(cls, coors: np.ndarray) -> np.ndarray:
        original_shape = coors.shape
        coors = np.squeeze(coors)
        if coors.shape[-1] == 7:
            coors = coors.reshape((-1, 7))
            return np.reshape(coors[:, [1, 2, 0, 3, 5, 6, 4]], original_shape)
            # X, Y, Z, W, X_r, Y_r, Z_r -> Y, Z, X, W, Y_r, Z_r, X_r
        else:
            coors = np.reshape(coors, (-1, 3))
            return np.reshape(coors[:, [1, 2, 0]], original_shape) # XYZ -> YZX
