import fcl

#TODO: reduce fcl objects regeneration?
class PrimitiveCollisionManager:
    def __init__(self, frame, primitives) -> None:
        assert len(primitives) > 0
        self.geom_id_to_name_map = None
        if type(primitives) == dict: # legacy impl before using PrimitivesFacade
            self.obj_names = list(primitives.keys())

            # list of (CollisionObject, CollisionGeometry) for the geometry-to-name mapping
            self.fcl_objs_and_geoms = [primitive.to_fcl_obj_and_geom(frame) for primitive in primitives.values()]
            
            self.geom_id_to_name_map = {id(geom): name for (_, geom), name in zip(self.fcl_objs_and_geoms, self.obj_names)}
        else:
            self.fcl_objs_and_geoms = [primitive.to_fcl_obj_and_geom(frame) for primitive in primitives]
        
        self.manager = fcl.DynamicAABBTreeCollisionManager()
        self.manager.registerObjects([obj for obj, _ in self.fcl_objs_and_geoms])
        self.manager.setup()

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