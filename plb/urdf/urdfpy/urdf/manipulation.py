from collections import OrderedDict
import copy
import os

from lxml import etree as ET
import networkx as nx
import numpy as np
import six

from urdfpy.urdf.base import URDFType
from urdfpy.urdf.joint import Joint
from urdfpy.urdf.transmission import Transmission
from urdfpy.urdf.link import Material, Link

class Robot(URDFType):
    """The top-level URDF specification for a robot.

    The URDF encapsulates an articulated object, such as a robot or a gripper.
    It is made of links and joints that tie them together and define their
    relative motions.

    Parameters
    ----------
    name : str
        The name of the URDF.
    links : list of :class:`.Link`
        The links of the URDF.
    joints : list of :class:`.Joint`, optional
        The joints of the URDF.
    transmissions : list of :class:`.Transmission`, optional
        The transmissions of the URDF.
    materials : list of :class:`.Material`, optional
        The materials for the URDF.
    other_xml : str, optional
        A string containing any extra XML for extensions.
    """
    _ATTRIBS = {
        'name': (str, True),
    }
    _ELEMENTS = {
        'links': (Link, True, True),
        'joints': (Joint, False, True),
        'transmissions': (Transmission, False, True),
        'materials': (Material, False, True),
    }
    _TAG = 'robot'

    def __init__(self, name, links, joints=None,
                 transmissions=None, materials=None, other_xml=None):
        if joints is None:
            joints = []
        if transmissions is None:
            transmissions = []
        if materials is None:
            materials = []

        self.name = name
        self.other_xml = other_xml

        # No setters for these
        self._links = list(links)
        self._joints = list(joints)
        self._transmissions = list(transmissions)
        self._materials = list(materials)

        # Set up private helper maps from name to value
        self._link_map = {}
        self._joint_map = {}
        self._transmission_map = {}
        self._material_map = {}

        for x in self._links:
            if x.name in self._link_map:
                raise ValueError('Two links with name {} found'.format(x.name))
            self._link_map[x.name] = x

        for x in self._joints:
            if x.name in self._joint_map:
                raise ValueError('Two joints with name {} '
                                 'found'.format(x.name))
            self._joint_map[x.name] = x

        for x in self._transmissions:
            if x.name in self._transmission_map:
                raise ValueError('Two transmissions with name {} '
                                 'found'.format(x.name))
            self._transmission_map[x.name] = x

        for x in self._materials:
            if x.name in self._material_map:
                raise ValueError('Two materials with name {} '
                                 'found'.format(x.name))
            self._material_map[x.name] = x

        # Synchronize materials between links and top-level set
        self._merge_materials()

        # Validate the joints and transmissions
        actuated_joints = self._validate_joints()
        self._validate_transmissions()

        # Create the link graph and base link/end link sets
        self._G = nx.DiGraph()

        # Add all links
        for link in self.links:
            self._G.add_node(link)

        # Add all edges from CHILDREN TO PARENTS, with joints as their object
        for joint in self.joints:
            parent = self._link_map[joint.parent]
            child = self._link_map[joint.child]
            self._G.add_edge(child, parent, joint=joint)

        # Validate the graph and get the base and end links
        self._base_link, self._end_links = self._validate_graph()

        # Cache the paths to the base link
        self._paths_to_base = nx.shortest_path(
            self._G, target=self._base_link
        )

        self._actuated_joints = self._sort_joints(actuated_joints)

        # Cache the reverse topological order (useful for speeding up FK,
        # as we want to start at the base and work outward to cache
        # computation.
        self._reverse_topo = list(reversed(list(nx.topological_sort(self._G))))

    @property
    def name(self):
        """str : The name of the URDF.
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def links(self):
        """list of :class:`.Link` : The links of the URDF.

        This returns a copy of the links array which cannot be edited
        directly. If you want to add or remove links, use
        the appropriate functions.
        """
        return copy.copy(self._links)

    @property
    def link_map(self):
        """dict : Map from link names to the links themselves.

        This returns a copy of the link map which cannot be edited
        directly. If you want to add or remove links, use
        the appropriate functions.
        """
        return copy.copy(self._link_map)

    @property
    def joints(self):
        """list of :class:`.Joint` : The links of the URDF.

        This returns a copy of the joints array which cannot be edited
        directly. If you want to add or remove joints, use
        the appropriate functions.
        """
        return copy.copy(self._joints)

    @property
    def joint_map(self):
        """dict : Map from joint names to the joints themselves.

        This returns a copy of the joint map which cannot be edited
        directly. If you want to add or remove joints, use
        the appropriate functions.
        """
        return copy.copy(self._joint_map)

    @property
    def transmissions(self):
        """list of :class:`.Transmission` : The transmissions of the URDF.

        This returns a copy of the transmissions array which cannot be edited
        directly. If you want to add or remove transmissions, use
        the appropriate functions.
        """
        return copy.copy(self._transmissions)

    @property
    def transmission_map(self):
        """dict : Map from transmission names to the transmissions themselves.

        This returns a copy of the transmission map which cannot be edited
        directly. If you want to add or remove transmissions, use
        the appropriate functions.
        """
        return copy.copy(self._transmission_map)

    @property
    def materials(self):
        """list of :class:`.Material` : The materials of the URDF.

        This returns a copy of the materials array which cannot be edited
        directly. If you want to add or remove materials, use
        the appropriate functions.
        """
        return copy.copy(self._materials)

    @property
    def material_map(self):
        """dict : Map from material names to the materials themselves.

        This returns a copy of the material map which cannot be edited
        directly. If you want to add or remove materials, use
        the appropriate functions.
        """
        return copy.copy(self._material_map)

    @property
    def other_xml(self):
        """str : Any extra XML that belongs with the URDF.
        """
        return self._other_xml

    @other_xml.setter
    def other_xml(self, value):
        self._other_xml = value

    @property
    def actuated_joints(self):
        """list of :class:`.Joint` : The joints that are independently
        actuated.

        This excludes mimic joints and fixed joints. The joints are listed
        in topological order, starting from the base-most joint.
        """
        return self._actuated_joints

    @property
    def actuated_joint_names(self):
        """list of :class:`.Joint` : The names of joints that are independently
        actuated.

        This excludes mimic joints and fixed joints. The joints are listed
        in topological order, starting from the base-most joint.
        """
        return [j.name for j in self._actuated_joints]

    def cfg_to_vector(self, cfg):
        """Convert a configuration dictionary into a configuration vector.

        Parameters
        ----------
        cfg : dict or None
            The configuration value.

        Returns
        -------
        vec : (n,) float
            The configuration vector, or None if no actuated joints present.
        """
        if cfg is None:
            if len(self.actuated_joints) > 0:
                return np.zeros(len(self.actuated_joints))
            else:
                return None
        elif isinstance(cfg, (list, tuple, np.ndarray)):
            return np.asanyarray(cfg)
        elif isinstance(cfg, dict):
            vec = np.zeros(len(self.actuated_joints))
            for i, jn in enumerate(self.actuated_joint_names):
                if jn in cfg:
                    vec[i] = cfg[jn]
            return vec
        else:
            raise ValueError('Invalid configuration: {}'.format(cfg))

    @property
    def base_link(self):
        """:class:`.Link`: The base link for the URDF.

        The base link is the single link that has no parent.
        """
        return self._base_link

    @property
    def end_links(self):
        """list of :class:`.Link`: The end links for the URDF.

        The end links are the links that have no children.
        """
        return self._end_links

    @property
    def joint_limit_cfgs(self):
        """tuple of dict : The lower-bound and upper-bound joint configuration
        maps.

        The first map is the lower-bound map, which maps limited joints to
        their lower joint limits.
        The second map is the upper-bound map, which maps limited joints to
        their upper joint limits.
        """
        lb = {}
        ub = {}
        for joint in self.actuated_joints:
            if joint.limit is not None:
                if joint.limit.lower is not None:
                    lb[joint] = joint.limit.lower
                if joint.limit.upper is not None:
                    ub[joint] = joint.limit.upper
        return (lb, ub)

    @property
    def joint_limits(self):
        """(n,2) float : A lower and upper limit for each joint.
        """
        limits = []
        for joint in self.actuated_joints:
            limit = [-np.infty, np.infty]
            if joint.limit is not None:
                if joint.limit.lower is not None:
                    limit[0] = joint.limit.lower
                if joint.limit.upper is not None:
                    limit[1] = joint.limit.upper
            limits.append(limit)
        return np.array(limits)

    def link_fk(self, cfg=None, link=None, links=None, use_names=False):
        """Computes the poses of the URDF's links via forward kinematics.

        Parameters
        ----------
        cfg : dict or (n), float
            A map from joints or joint names to configuration values for
            each joint, or a list containing a value for each actuated joint
            in sorted order from the base link.
            If not specified, all joints are assumed to be in their default
            configurations.
        link : str or :class:`.Link`
            A single link or link name to return a pose for.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only these links will be in the returned map. If neither
            link nor links are specified all links are returned.
        use_names : bool
            If True, the returned dictionary will have keys that are string
            link names rather than the links themselves.

        Returns
        -------
        fk : dict or (4,4) float
            A map from links to 4x4 homogenous transform matrices that
            position them relative to the base link's frame, or a single
            4x4 matrix if ``link`` is specified.
        """
        # Process config value
        joint_cfg = self._process_cfg(cfg)

        # Process link set
        link_set = set()
        if link is not None:
            if isinstance(link, six.string_types):
                link_set.add(self._link_map[link])
            elif isinstance(link, Link):
                link_set.add(link)
        elif links is not None:
            for lnk in links:
                if isinstance(lnk, six.string_types):
                    link_set.add(self._link_map[lnk])
                elif isinstance(lnk, Link):
                    link_set.add(lnk)
                else:
                    raise TypeError('Got object of type {} in links list'
                                    .format(type(lnk)))
        else:
            link_set = self.links

        # Compute forward kinematics in reverse topological order
        fk = OrderedDict()
        for lnk in self._reverse_topo:
            if lnk not in link_set:
                continue
            pose = np.eye(4, dtype=np.float64)
            path = self._paths_to_base[lnk]
            for i in range(len(path) - 1):
                child = path[i]
                parent = path[i + 1]
                joint = self._G.get_edge_data(child, parent)['joint']

                cfg = None
                if joint.mimic is not None:
                    mimic_joint = self._joint_map[joint.mimic.joint]
                    if mimic_joint in joint_cfg:
                        cfg = joint_cfg[mimic_joint]
                        cfg = joint.mimic.multiplier * cfg + joint.mimic.offset
                elif joint in joint_cfg:
                    cfg = joint_cfg[joint]
                pose = joint.get_child_pose(cfg).dot(pose)

                # Check existing FK to see if we can exit early
                if parent in fk:
                    pose = fk[parent].dot(pose)
                    break
            fk[lnk] = pose

        if link:
            if isinstance(link, six.string_types):
                return fk[self._link_map[link]]
            else:
                return fk[link]
        if use_names:
            return {ell.name: fk[ell] for ell in fk}
        return fk

    def link_fk_batch(self, cfgs=None, link=None, links=None, use_names=False):
        """Computes the poses of the URDF's links via forward kinematics in a batch.

        Parameters
        ----------
        cfgs : dict, list of dict, or (n,m), float
            One of the following: (A) a map from joints or joint names to vectors
            of joint configuration values, (B) a list of maps from joints or joint names
            to single configuration values, or (C) a list of ``n`` configuration vectors,
            each of which has a vector with an entry for each actuated joint.
        link : str or :class:`.Link`
            A single link or link name to return a pose for.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only these links will be in the returned map. If neither
            link nor links are specified all links are returned.
        use_names : bool
            If True, the returned dictionary will have keys that are string
            link names rather than the links themselves.

        Returns
        -------
        fk : dict or (n,4,4) float
            A map from links to a (n,4,4) vector of homogenous transform matrices that
            position the links relative to the base link's frame, or a single
            nx4x4 matrix if ``link`` is specified.
        """
        joint_cfgs, n_cfgs = self._process_cfgs(cfgs)

        # Process link set
        link_set = set()
        if link is not None:
            if isinstance(link, six.string_types):
                link_set.add(self._link_map[link])
            elif isinstance(link, Link):
                link_set.add(link)
        elif links is not None:
            for lnk in links:
                if isinstance(lnk, six.string_types):
                    link_set.add(self._link_map[lnk])
                elif isinstance(lnk, Link):
                    link_set.add(lnk)
                else:
                    raise TypeError('Got object of type {} in links list'
                                    .format(type(lnk)))
        else:
            link_set = self.links

        # Compute FK mapping each link to a vector of matrices, one matrix per cfg
        fk = OrderedDict()
        for lnk in self._reverse_topo:
            if lnk not in link_set:
                continue
            poses = np.tile(np.eye(4, dtype=np.float64), (n_cfgs, 1, 1))
            path = self._paths_to_base[lnk]
            for i in range(len(path) - 1):
                child = path[i]
                parent = path[i + 1]
                joint = self._G.get_edge_data(child, parent)['joint']

                cfg_vals = None
                if joint.mimic is not None:
                    mimic_joint = self._joint_map[joint.mimic.joint]
                    if mimic_joint in joint_cfgs:
                        cfg_vals = joint_cfgs[mimic_joint]
                        cfg_vals = joint.mimic.multiplier * cfg_vals + joint.mimic.offset
                elif joint in joint_cfgs:
                    cfg_vals = joint_cfgs[joint]
                poses = np.matmul(joint.get_child_poses(cfg_vals, n_cfgs), poses)

                if parent in fk:
                    poses = np.matmul(fk[parent], poses)
                    break
            fk[lnk] = poses

        if link:
            if isinstance(link, six.string_types):
                return fk[self._link_map[link]]
            else:
                return fk[link]
        if use_names:
            return {ell.name: fk[ell] for ell in fk}
        return fk

    def visual_mesh_fk(self, cfg=None, links=None):
        """Computes the poses of the URDF's visual triangle mesh using fk.

        Parameters
        ----------
        cfg : dict or (n), float
            A map from joints or joint names to configuration values for
            each joint, or a list containing a value for each actuated joint
            in sorted order from the base link.
            If not specified, all joints are assumed to be in their default
            configurations.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only meshes from these links will be in the returned map.
            If not specified, all links are returned.

        Returns
        -------
        fk : dict
            A map from :class:`open3d.geometry.TriangleMesh` objects that are
            part of the visual geometry of the specified links to the
            4x4 homogenous transform matrices that position them relative
            to the base link's frame.
        """
        lfk = self.link_fk(cfg=cfg, links=links)

        fk = OrderedDict()
        for link in lfk:
            for visual in link.visuals:
                geometryMesh = visual.geometry.meshes
                pose = lfk[link].dot(visual.origin)
                if visual.geometry.mesh is not None:
                    if visual.geometry.mesh.scale is not None:
                        S = np.eye(4, dtype=np.float64)
                        S[:3,:3] = np.diag(visual.geometry.mesh.scale)
                        pose = pose.dot(S)
                fk[geometryMesh] = pose
        return fk

    def collision_mesh_fk(self, cfg=None, links=None):
        """Computes the poses of the URDF's collision mesh using fk.

        Parameters
        ----------
        cfg : dict or (n), float
            A map from joints or joint names to configuration values for
            each joint, or a list containing a value for each actuated joint
            in sorted order from the base link.
            If not specified, all joints are assumed to be in their default
            configurations.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only triangle meshes from these links will be in the returned map.
            If not specified, all links are returned.

        Returns
        -------
        fk : dict
            A map from :class:`open3d.geometry.TriangleMesh` objects that are
            part of the collision geometry of the specified links to the
            4x4 homogenous transform matrices that position them relative
            to the base link's frame.
        """
        lfk = self.link_fk(cfg=cfg, links=links)

        fk = OrderedDict()
        for link in lfk:
            pose = lfk[link]
            cm = link.collision_mesh
            if cm is not None:
                fk[cm] = pose
        return fk

    def collision_mesh_fk_batch(self, cfgs=None, links=None):
        """Computes the poses of the URDF's collision mesh using fk.

        Parameters
        ----------
        cfgs : dict, list of dict, or (n,m), float
            One of the following: (A) a map from joints or joint names to vectors
            of joint configuration values, (B) a list of maps from joints or joint names
            to single configuration values, or (C) a list of ``n`` configuration vectors,
            each of which has a vector with an entry for each actuated joint.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only triangle meshes from these links will be in the returned map.
            If not specified, all links are returned.

        Returns
        -------
        fk : dict
            A map from :class:`open3d.geometry.TriangleMesh` objects that are
            part of the collision geometry of the specified links to the
            4x4 homogenous transform matrices that position them relative
            to the base link's frame.
        """
        lfk = self.link_fk_batch(cfgs=cfgs, links=links)

        fk = OrderedDict()
        for link in lfk:
            poses = lfk[link]
            cm = link.collision_mesh
            if cm is not None:
                fk[cm] = poses
        return fk


    def copy(self, name=None, prefix='', scale=None, collision_only=False):
        """Make a deep copy of the URDF.

        Parameters
        ----------
        name : str, optional
            A name for the new URDF. If not specified, ``self.name`` is used.
        prefix : str, optional
            A prefix to apply to all names except for the base URDF name.
        scale : float or (3,) float, optional
            A scale to apply to the URDF.
        collision_only : bool, optional
            If True, all visual geometry is redirected to the collision geometry.

        Returns
        -------
        copy : :class:`.URDF`
            The copied URDF.
        """
        return Robot(
            name = (name if name else self.name),
            links=[v.copy(prefix, scale, collision_only) for v in self.links],
            joints=[v.copy(prefix, scale) for v in self.joints],
            transmissions=[v.copy(prefix, scale) for v in self.transmissions],
            materials=[v.copy(prefix, scale) for v in self.materials],
            other_xml=self.other_xml
        )

    def save(self, file_obj):
        """Save this URDF to a file.

        Parameters
        ----------
        file_obj : str or file-like object
            The file to save the URDF to. Should be the path to the
            ``.urdf`` XML file. Any paths in the URDF should be specified
            as relative paths to the ``.urdf`` file instead of as ROS
            resources.

        Returns
        -------
        urdf : :class:`.URDF`
            The parsed URDF.
        """
        if isinstance(file_obj, six.string_types):
            path, _ = os.path.split(file_obj)
        else:
            path, _ = os.path.split(os.path.realpath(file_obj.name))

        node = self._to_xml(None, path)
        tree = ET.ElementTree(node)
        tree.write(file_obj, pretty_print=True,
                   xml_declaration=True, encoding='utf-8')

    def join(self, other, link, origin=None, name=None, prefix=''):
        """Join another URDF to this one by rigidly fixturing the two at a link.

        Parameters
        ----------
        other : :class:.`URDF`
            Another URDF to fuze to this one.
        link : :class:`.Link` or str
            The link of this URDF to attach the other URDF to.
        origin : (4,4) float, optional
            The location in this URDF's link frame to attach the base link of the other
            URDF at.
        name : str, optional
            A name for the new URDF.
        prefix : str, optional
            If specified, all joints and links from the (other) mesh will be pre-fixed
            with this value to avoid name clashes.

        Returns
        -------
        :class:`.URDF`
            The new URDF.
        """
        myself = self.copy()
        other = other.copy(prefix=prefix)

        # Validate
        link_names = set(myself.link_map.keys())
        other_link_names = set(other.link_map.keys())
        if len(link_names.intersection(other_link_names)) > 0:
            raise ValueError('Cannot merge two URDFs with shared link names')

        joint_names = set(myself.joint_map.keys())
        other_joint_names = set(other.joint_map.keys())
        if len(joint_names.intersection(other_joint_names)) > 0:
            raise ValueError('Cannot merge two URDFs with shared joint names')

        links = myself.links + other.links
        joints = myself.joints + other.joints
        transmissions = myself.transmissions + other.transmissions
        materials = myself.materials + other.materials

        if name is None:
            name = self.name

        # Create joint that links the two rigidly
        joints.append(Joint(
            name='{}_join_{}{}_joint'.format(self.name, prefix, other.name),
            joint_type='fixed',
            parent=link if isinstance(link, str) else link.name,
            child=other.base_link.name,
            origin=origin
        ))

        return Robot(name=name, links=links, joints=joints, transmissions=transmissions,
                    materials=materials)

    def _merge_materials(self):
        """Merge the top-level material set with the link materials.
        """
        for link in self.links:
            for v in link.visuals:
                if v.material is None:
                    continue
                if v.material.name in self.material_map:
                    v.material = self._material_map[v.material.name]
                else:
                    self._materials.append(v.material)
                    self._material_map[v.material.name] = v.material

    @staticmethod
    def load(file_obj):
        """Load a URDF from a file.

        Parameters
        ----------
        file_obj : str or file-like object
            The file to load the URDF from. Should be the path to the
            ``.urdf`` XML file. Any paths in the URDF should be specified
            as relative paths to the ``.urdf`` file instead of as ROS
            resources.

        Returns
        -------
        urdf : :class:`.URDF`
            The parsed URDF.
        """
        if isinstance(file_obj, six.string_types):
            if os.path.isfile(file_obj):
                parser = ET.XMLParser(remove_comments=True,
                                      remove_blank_text=True)
                tree = ET.parse(file_obj, parser=parser)
                path, _ = os.path.split(file_obj)
            else:
                raise ValueError('{} is not a file'.format(file_obj))
        else:
            parser = ET.XMLParser(remove_comments=True, remove_blank_text=True)
            tree = ET.parse(file_obj, parser=parser)
            path, _ = os.path.split(file_obj.name)

        node = tree.getroot()
        return Robot._from_xml(node, path)

    def _validate_joints(self):
        """Raise an exception of any joints are invalidly specified.

        Checks for the following:

        - Joint parents are valid link names.
        - Joint children are valid link names that aren't the same as parent.
        - Joint mimics have valid joint names that aren't the same joint.

        Returns
        -------
        actuated_joints : list of :class:`.Joint`
            The joints in the model that are independently controllable.
        """
        actuated_joints = []
        for joint in self.joints:
            if joint.parent not in self._link_map:
                raise ValueError('Joint {} has invalid parent link name {}'
                                 .format(joint.name, joint.parent))
            if joint.child not in self._link_map:
                raise ValueError('Joint {} has invalid child link name {}'
                                 .format(joint.name, joint.child))
            if joint.child == joint.parent:
                raise ValueError('Joint {} has matching parent and child'
                                 .format(joint.name))
            if joint.mimic is not None:
                if joint.mimic.joint not in self._joint_map:
                    raise ValueError(
                        'Joint {} has an invalid mimic joint name {}'
                        .format(joint.name, joint.mimic.joint)
                    )
                if joint.mimic.joint == joint.name:
                    raise ValueError(
                        'Joint {} set up to mimic itself'
                        .format(joint.mimic.joint)
                    )
            elif joint.joint_type != 'fixed':
                actuated_joints.append(joint)

        # Do a depth-first search
        return actuated_joints

    def _sort_joints(self, joints):
        """Sort joints by ascending distance from the base link (topologically).

        Parameters
        ----------
        joints : list of :class:`.Joint`
            The joints to sort.

        Returns
        -------
        joints : list of :class:`.Joint`
            The sorted joints.
        """
        lens = []
        for joint in joints:
            child_link = self._link_map[joint.child]
            lens.append(len(self._paths_to_base[child_link]))
        order = np.argsort(lens)
        return np.array(joints)[order].tolist()

    def _validate_transmissions(self):
        """Raise an exception of any transmissions are invalidly specified.

        Checks for the following:

        - Transmission joints have valid joint names.
        """
        for t in self.transmissions:
            for joint in t.joints:
                if joint.name not in self._joint_map:
                    raise ValueError('Transmission {} has invalid joint name '
                                     '{}'.format(t.name, joint.name))

    def _validate_graph(self):
        """Raise an exception if the link-joint structure is invalid.

        Checks for the following:

        - The graph is connected in the undirected sense.
        - The graph is acyclic in the directed sense.
        - The graph has only one base link.

        Returns
        -------
        base_link : :class:`.Link`
            The base link of the URDF.
        end_links : list of :class:`.Link`
            The end links of the URDF.
        """

        # Check that the link graph is weakly connected
        if not nx.is_weakly_connected(self._G):
            link_clusters = []
            for cc in nx.weakly_connected_components(self._G):
                cluster = []
                for n in cc:
                    cluster.append(n.name)
                link_clusters.append(cluster)
            message = ('Links are not all connected. '
                       'Connected components are:')
            for lc in link_clusters:
                message += '\n\t'
                for n in lc:
                    message += ' {}'.format(n)
            raise ValueError(message)

        # Check that link graph is acyclic
        if not nx.is_directed_acyclic_graph(self._G):
            raise ValueError('There are cycles in the link graph')

        # Ensure that there is exactly one base link, which has no parent
        base_link = None
        end_links = []
        for n in self._G:
            if len(nx.descendants(self._G, n)) == 0:
                if base_link is None:
                    base_link = n
                else:
                    raise ValueError('Links {} and {} are both base links!'
                                     .format(n.name, base_link.name))
            if len(nx.ancestors(self._G, n)) == 0:
                end_links.append(n)
        return base_link, end_links

    def _process_cfg(self, cfg):
        """Process a joint configuration spec into a dictionary mapping
        joints to configuration values.
        """
        joint_cfg = {}
        if cfg is None:
            return joint_cfg
        if isinstance(cfg, dict):
            for joint in cfg:
                if isinstance(joint, six.string_types):
                    joint_cfg[self._joint_map[joint]] = cfg[joint]
                elif isinstance(joint, Joint):
                    joint_cfg[joint] = cfg[joint]
        elif isinstance(cfg, (list, tuple, np.ndarray)):
            if len(cfg) != len(self.actuated_joints):
                raise ValueError('Cfg must have same length as actuated joints '
                                 'if specified as a numerical array')
            for joint, value in zip(self.actuated_joints, cfg):
                joint_cfg[joint] = value
        else:
            raise TypeError('Invalid type for config')
        return joint_cfg

    def _process_cfgs(self, cfgs):
        """Process a list of joint configurations into a dictionary mapping joints to
        configuration values.

        This should result in a dict mapping each joint to a list of cfg values, one
        per joint.
        """
        joint_cfg = {j : [] for j in self.actuated_joints}
        n_cfgs = None
        if isinstance(cfgs, dict):
            for joint in cfgs:
                if isinstance(joint, six.string_types):
                    joint_cfg[self._joint_map[joint]] = cfgs[joint]
                else:
                    joint_cfg[joint] = cfgs[joint]
                if n_cfgs is None:
                    n_cfgs = len(cfgs[joint])
        elif isinstance(cfgs, (list, tuple, np.ndarray)):
            n_cfgs = len(cfgs)
            if isinstance(cfgs[0], dict):
                for cfg in cfgs:
                    for joint in cfg:
                        if isinstance(joint, six.string_types):
                            joint_cfg[self._joint_map[joint]].append(cfg[joint])
                        else:
                            joint_cfg[joint].append(cfg[joint])
            elif cfgs[0] is None:
                pass
            else:
                cfgs = np.asanyarray(cfgs, dtype=np.float64)
                for i, j in enumerate(self.actuated_joints):
                    joint_cfg[j] = cfgs[:,i]
        else:
            raise ValueError('Incorrectly formatted config array')

        for j in joint_cfg:
            if len(joint_cfg[j]) == 0:
                joint_cfg[j] = None
            elif len(joint_cfg[j]) != n_cfgs:
                raise ValueError('Inconsistent number of configurations for joints')

        return joint_cfg, n_cfgs

    @classmethod
    def _from_xml(cls, node, path):
        valid_tags = set(['joint', 'link', 'transmission', 'material'])
        kwargs = cls._parse(node, path)

        extra_xml_node = ET.Element('extra')
        for child in node:
            if child.tag not in valid_tags:
                extra_xml_node.append(child)

        data = ET.tostring(extra_xml_node)
        kwargs['other_xml'] = data
        return Robot(**kwargs)

    def _to_xml(self, parent, path):
        node = self._unparse(path)
        if self.other_xml:
            extra_tree = ET.fromstring(self.other_xml)
            for child in extra_tree:
                node.append(child)
        return node
