from typing import Dict
import numpy as np
import fcl
from plb.engine.primitive.primitives import Box, Sphere, Cylinder, Primitive


class PrimitiveCollisionManager:
    def __init__(self, frame, primitives) -> None:
        if type(primitives) == dict:
            self.obj_names = list(primitives.keys())
            self.fcl_objs = [self.build_fcl_obj(primitive, frame) for primitive in primitives.values()]
            self.geom_id_to_name_map = {id(obj[1]): name for obj, name in zip(self.fcl_objs, self.obj_names)}
        else:
            self.fcl_objs = [self.build_fcl_obj(primitive, frame) for primitive in primitives]
        
        self.manager = fcl.DynamicAABBTreeCollisionManager()
        self.manager.registerObjects([obj[0] for obj in self.fcl_objs])
        self.manager.setup()

    def build_fcl_obj(self, primitive: Primitive, frame):
        if type(primitive) == Box:
            x, y, z = primitive.size.to_numpy()
            geom = fcl.Box(x, y, z)
        elif type(primitive) == Sphere:
            radius = primitive.radius
            geom = fcl.Sphere(radius)
        elif type(primitive) == Cylinder:
            radius = primitive.r
            height = primitive.h
            geom = fcl.Cylinder(radius, height)
        else:
            print(type(primitive), 'geom not yet supported')
            return None, None

        rot = primitive.rotation.to_numpy()[frame]
        pos = primitive.position.to_numpy()[frame]
        tf = fcl.Transform(rot, pos)
        obj = fcl.CollisionObject(geom, tf)
        return obj, geom

    def check_robot_collision(self, collision_callback=None) -> bool:
        if not collision_callback:
            collision_callback = fcl.defaultCollisionCallback
        
        cdata = fcl.CollisionData()
        self.manager.collide(cdata, collision_callback)
        
        contacts = []
        if self.geom_id_to_name_map:
            for contact in cdata.result.contacts:
                o1 = self.geom_id_to_name_map.get(id(contact.o1))
                o2 = self.geom_id_to_name_map.get(id(contact.o2))
                contacts.append((o1, o2))

        return cdata.result.is_collision, contacts