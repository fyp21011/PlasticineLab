import numpy as np
import fcl
from plb.engine.primitive.primitives import Box, Primitives, Sphere, Cylinder, Primitive


class PrimitiveCollisionManager:
    def __init__(self, frame, primitives) -> None:
        self.fcl_objs = [self.build_fcl_obj(primitive, frame) for primitive in primitives]
        self.manager = fcl.DynamicAABBTreeCollisionManager()
        self.manager.registerObjects(self.fcl_objs)
        self.manager.setup()

    def build_fcl_obj(self, primitive: Primitive, frame):
        if type(primitive) == Box:
            x, y, z = primitive.size.to_numpy()
            shape = fcl.Box(x, y, z)
        elif type(primitive) == Sphere:
            radius = primitive.radius
            shape = fcl.Sphere(radius)
        elif type(primitive) == Cylinder:
            radius = primitive.r
            height = primitive.h
            shape = fcl.Cylinder(radius, height)
        else:
            print('shape not yet supported')
            return None

        rot = primitive.rotation.to_numpy()[frame]
        pos = primitive.position.to_numpy()[frame]
        tf = fcl.Transform(rot, pos)
        obj = fcl.CollisionObject(shape, tf)
        return obj

    def check_robot_collision(self) -> bool:
        cdata = fcl.CollisionData()
        self.manager.collide(cdata, fcl.defaultCollisionCallback)
        return cdata.result.is_collision