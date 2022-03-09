from typing import Any, Dict, List
import socket

import numpy as np

from plb.engine.primitive.primive_base import Primitive
from plb.urdfpy import Robot


class NetRenderer:
    def __init__(self,
        cfg: Dict[str, Any],
        free_primitives: List[Primitive],
        robots: List[Robot],
        ip:str = 'localhost',
        port: str = '4490'
    ) -> None:
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        """IPv4 TCP socket"""
        self.client.connect((ip, port))
        self.client.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, True)

    def set_particles(self, particles) -> None:
        pass

    def render_frame(self) -> None:
        pass

    def _config_env(self) -> None:
        pass

    def _connect_2_blender_server(self) -> None:
        pass

    def _send_info_2_blender_server(self, info: Any) -> None:
        pass