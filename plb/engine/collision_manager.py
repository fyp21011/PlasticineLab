import fcl
from plb.engine.primitive.primitive import Box, Sphere, Cylinder, Primitive

#TODO: reduce fcl objects regeneration?
class PrimitiveCollisionManager:
    def __init__(self, frame, primitives) -> None:
        self.geom_id_to_name_map = None
        if type(primitives) == dict: # legacy impl after using PrimitivesFacade
            self.obj_names = list(primitives.keys())

            # list of (CollisionObject, CollisionGeometry) for the geometry-to-name mapping
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
        
        # ref: https://github.com/BerkeleyAutomation/python-fcl#extracting-which-objects-are-in-collision
        # FCL report contacts with `fcl.CollisionGeometry` instead of `fcl.CollisionObject.`
        # Therefore we map the memory id of the CollisionGeometry to its corresponding CollisionObject and link name
        contacts = []
        if self.geom_id_to_name_map:
            for contact in cdata.result.contacts:
                o1 = self.geom_id_to_name_map.get(id(contact.o1))
                o2 = self.geom_id_to_name_map.get(id(contact.o2))
                contacts.append((o1, o2))

        return cdata.result.is_collision, contacts