import copy
import os

import numpy as np
import open3d as o3d
import six

from urdfpy.urdf.base import URDFType
from urdfpy.utils import (
    parse_origin,
    unparse_origin,
    get_filename,
    load_mesh,
    configure_origin
)

class Box(URDFType):
    """A rectangular prism whose center is at the local origin.

    Parameters
    ----------
    size : (3,) float
        The length, width, and height of the box in meters.
    """

    _ATTRIBS = {
        'size': (np.ndarray, True)
    }
    _TAG = 'box'

    def __init__(self, size):
        self.size = size
        self._meshes = None

    @property
    def size(self):
        """(3,) float : The length, width, and height of the box in meters.
        """
        return self._size

    @size.setter
    def size(self, value):
        self._size = np.asanyarray(value).astype(np.float64)
        self._meshes = None

    @property
    def meshes(self):
        """a pen3d.geometry.TriangleMesh object : The triangular meshes
        whose vertex normal has been computed already
        """
        if self._meshes is None:
            self._meshes = o3d.geometry.TriangleMesh.create_box(
                self._size[0],
                self._size[1],
                self._size[2]
            )
            T = np.eye(4)
            T[:3] += (
                -1 * self._size[0], 
                -1 * self._size[1], 
                -1 * self._size[2],
                0
            )
            self._meshes.transform(T)
            self._meshes.compute_vertex_normals()
        return self._meshes

    def copy(self, prefix='', scale=None):
        """Create a deep copy with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all names.

        Returns
        -------
        :class:`.Box`
            A deep copy.
        """
        if scale is None:
            scale = 1.0
        b = Box(
            size=self.size.copy() * scale,
        )
        return b

class Cylinder(URDFType):
    """A cylinder whose center is at the local origin.

    Parameters
    ----------
    radius : float
        The radius of the cylinder in meters.
    length : float
        The length of the cylinder in meters.
    """

    _ATTRIBS = {
        'radius': (float, True),
        'length': (float, True),
    }
    _TAG = 'cylinder'

    def __init__(self, radius, length):
        self.radius = radius
        self.length = length
        self._meshes = None

    @property
    def radius(self):
        """float : The radius of the cylinder in meters.
        """
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = float(value)
        self._meshes = None

    @property
    def length(self):
        """float : The length of the cylinder in meters.
        """
        return self._length

    @length.setter
    def length(self, value):
        self._length = float(value)
        self._meshes = None

    @property
    def meshes(self):
        """a open3d.geometry.TriangleMesh object : The triangular meshes
        whose vertex normal has been computed already
        """
        if self._meshes is None:
            self._meshes = o3d.geometry.TriangleMesh.create_cylinder(
                radius = self.radius,
                height = self.length
            )
            self._meshes.compute_vertex_normals()
        return self._meshes

    def copy(self, prefix='', scale=None):
        """Create a deep copy with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all names.

        Returns
        -------
        :class:`.Cylinder`
            A deep copy.
        """
        if scale is None:
            scale = 1.0
        if isinstance(scale, (list, np.ndarray)):
            if scale[0] != scale[1]:
                raise ValueError('Cannot rescale cylinder geometry with asymmetry in x/y')
            c = Cylinder(
                radius=self.radius * scale[0],
                length=self.length * scale[2],
            )
        else:
            c = Cylinder(
                radius=self.radius * scale,
                length=self.length * scale,
            )
        return c

class Sphere(URDFType):
    """A sphere whose center is at the local origin.

    Parameters
    ----------
    radius : float
        The radius of the sphere in meters.
    """
    _ATTRIBS = {
        'radius': (float, True),
    }
    _TAG = 'sphere'

    def __init__(self, radius):
        self.radius = radius
        self._meshes = []

    @property
    def radius(self):
        """float : The radius of the sphere in meters.
        """
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = float(value)
        self._meshes = None

    @property
    def meshes(self):
        """a pen3d.geometry.TriangleMesh object : The triangular meshes
        whose vertex normal has been computed already
        """
        if self._meshes is None:
            self._meshes = o3d.geometry.TriangleMesh.create_sphere(self._radius)
            self._meshes.compute_vertex_normals()
        return self._meshes

    def copy(self, prefix='', scale=None):
        """Create a deep copy with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all names.

        Returns
        -------
        :class:`.Sphere`
            A deep copy.
        """
        if scale is None:
            scale = 1.0
        if isinstance(scale, (list, np.ndarray)):
            if scale[0] != scale[1] or scale[0] != scale[2]:
                raise ValueError('Spheres do not support non-uniform scaling!')
            scale = scale[0]
        s = Sphere(
            radius=self.radius * scale,
        )
        return s

class Mesh(URDFType):
    """A triangular mesh object.

    Parameters
    ----------
    filename : str
        The path to the mesh that contains this object. This can be
        relative to the top-level URDF or an absolute path.
    scale : (3,) float, optional
        The scaling value for the mesh along the XYZ axes.
        If ``None``, assumes no scale is applied.
    meshes : o3d.geometry.TriangleMesh object
        If not specified, the mesh is loaded from the file.
    """
    _ATTRIBS = {
        'filename': (str, True),
        'scale': (np.ndarray, False)
    }
    _TAG = 'mesh'

    def __init__(self, filename, scale=None, meshes=None):
        if meshes is None:
            meshes = load_mesh(filename)
        self.filename = filename
        self.scale = scale
        self.meshes = meshes

    @property
    def filename(self):
        """str : The path to the mesh file for this object.
        """
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value

    @property
    def scale(self):
        """(3,) float : A scaling for the mesh along its local XYZ axes.
        """
        return self._scale

    @scale.setter
    def scale(self, value):
        if value is not None:
            value = np.asanyarray(value).astype(np.float64)
        self._scale = value

    @property
    def meshes(self):
        """a pen3d.geometry.TriangleMesh object : The triangular meshes
        whose vertex normal has been computed already
        """
        return self._meshes

    @meshes.setter
    def meshes(self, value):
        if isinstance(value, six.string_types):
            value = load_mesh(value)
        elif not isinstance(value, o3d.geometry.TriangleMesh):
            raise TypeError('Mesh requires a open3d.geometry.TriangleMesh')
        self._meshes = value

    @classmethod
    def _from_xml(cls, node, path):
        kwargs = cls._parse(node, path)

        # Load the mesh, combining collision geometry meshes for simplicity
        fn = get_filename(path, kwargs['filename'])
        combine = node.getparent().getparent().tag == Collision._TAG
        meshes = load_mesh(fn)
        if combine:
            # Simplify the geometry
            voxel_size = max(meshes.get_max_bound() - meshes.get_min_bound()) / 32
            meshes = meshes.simplify_vertex_clustering(
                voxel_size=voxel_size,
                contraction=o3d.geometry.SimplificationContraction.Average
            )
        kwargs['meshes'] = meshes

        return Mesh(**kwargs)

    def _to_xml(self, parent, path):
        # Get the filename
        fn = get_filename(path, self.filename, makedirs=True)

        # Export the meshes as a single file
        meshes = self.meshes
        if isinstance(meshes, list) and len(meshes) == 1:
            meshes = meshes[0]
        o3d.io.write_triangle_mesh(fn, meshes)

        # Unparse the node
        node = self._unparse(path)
        return node

    def copy(self, prefix='', scale=None):
        """Create a deep copy with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all names.

        Returns
        -------
        :class:`.Sphere`
            A deep copy.
        """
        meshes = copy.deepcopy(self.meshes)
        if scale is not None:
            sm = np.eye(4)
            if isinstance(scale, (list, np.ndarray)):
                sm[:3,:3] = np.diag(scale)
            else:
                sm[:3,:3] = np.diag(np.repeat(scale, 3))
            meshes = meshes.transform(sm)
        base, fn = os.path.split(self.filename)
        fn = '{}{}'.format(prefix, self.filename)
        m = Mesh(
            filename=os.path.join(base, fn),
            scale=(self.scale.copy() if self.scale is not None else None),
            meshes=meshes
        )
        return m

class Geometry(URDFType):
    """A wrapper for all geometry types.

    Only one of the following values can be set, all others should be set
    to ``None``.

    Parameters
    ----------
    box : :class:`.Box`, optional
        Box geometry.
    cylinder : :class:`.Cylinder`
        Cylindrical geometry.
    sphere : :class:`.Sphere`
        Spherical geometry.
    mesh : :class:`.Mesh`
        Mesh geometry.
    """

    _ELEMENTS = {
        'box': (Box, False, False),
        'cylinder': (Cylinder, False, False),
        'sphere': (Sphere, False, False),
        'mesh': (Mesh, False, False),
    }
    _TAG = 'geometry'

    def __init__(self, box=None, cylinder=None, sphere=None, mesh=None):
        if (box is None and cylinder is None and
                sphere is None and mesh is None):
            raise ValueError('At least one geometry element must be set')
        self.box = box
        self.cylinder = cylinder
        self.sphere = sphere
        self.mesh = mesh

    @property
    def box(self):
        """:class:`.Box` : Box geometry.
        """
        return self._box

    @box.setter
    def box(self, value):
        if value is not None and not isinstance(value, Box):
            raise TypeError('Expected Box type')
        self._box = value

    @property
    def cylinder(self):
        """:class:`.Cylinder` : Cylinder geometry.
        """
        return self._cylinder

    @cylinder.setter
    def cylinder(self, value):
        if value is not None and not isinstance(value, Cylinder):
            raise TypeError('Expected Cylinder type')
        self._cylinder = value

    @property
    def sphere(self):
        """:class:`.Sphere` : Spherical geometry.
        """
        return self._sphere

    @sphere.setter
    def sphere(self, value):
        if value is not None and not isinstance(value, Sphere):
            raise TypeError('Expected Sphere type')
        self._sphere = value

    @property
    def mesh(self):
        """:class:`.Mesh` : Mesh geometry.
        """
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        if value is not None and not isinstance(value, Mesh):
            raise TypeError('Expected Mesh type')
        self._mesh = value

    @property
    def geometry(self):
        """:class:`.Box`, :class:`.Cylinder`, :class:`.Sphere`, or
        :class:`.Mesh` : The valid geometry element.
        """
        if self.box is not None:
            return self.box
        if self.cylinder is not None:
            return self.cylinder
        if self.sphere is not None:
            return self.sphere
        if self.mesh is not None:
            return self.mesh
        return None

    @property
    def meshes(self):
        """a pen3d.geometry.TriangleMesh object : The triangular meshes
        whose vertex normal has been computed already
        """
        return self.geometry.meshes

    def copy(self, prefix='', scale=None):
        """Create a deep copy with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all names.

        Returns
        -------
        :class:`.Geometry`
            A deep copy.
        """
        v = Geometry(
            box=(self.box.copy(prefix=prefix, scale=scale) if self.box else None),
            cylinder=(self.cylinder.copy(prefix=prefix, scale=scale) if self.cylinder else None),
            sphere=(self.sphere.copy(prefix=prefix, scale=scale) if self.sphere else None),
            mesh=(self.mesh.copy(prefix=prefix, scale=scale) if self.mesh else None),
        )
        return v

class Collision(URDFType):
    """Collision properties of a link.

    Parameters
    ----------
    geometry : :class:`.Geometry`
        The geometry of the element
    name : str, optional
        The name of the collision geometry.
    origin : (4,4) float, optional
        The pose of the collision element relative to the link frame.
        Defaults to identity.
    """

    _ATTRIBS = {
        'name': (str, False)
    }
    _ELEMENTS = {
        'geometry': (Geometry, True, False),
    }
    _TAG = 'collision'

    def __init__(self, name, origin, geometry):
        self.geometry = geometry
        self.name = name
        self.origin = origin

    @property
    def geometry(self):
        """:class:`.Geometry` : The geometry of this element.
        """
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        if not isinstance(value, Geometry):
            raise TypeError('Must set geometry with Geometry object')
        self._geometry = value

    @property
    def name(self):
        """str : The name of this collision element.
        """
        return self._name

    @name.setter
    def name(self, value):
        if value is not None:
            value = str(value)
        self._name = value

    @property
    def origin(self):
        """(4,4) float : The pose of this element relative to the link frame.
        """
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = configure_origin(value)

    @classmethod
    def _from_xml(cls, node, path):
        kwargs = cls._parse(node, path)
        kwargs['origin'] = parse_origin(node)
        return Collision(**kwargs)

    def _to_xml(self, parent, path):
        node = self._unparse(path)
        node.append(unparse_origin(self.origin))
        return node

    def copy(self, prefix='', scale=None):
        """Create a deep copy of the visual with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all joint and link names.

        Returns
        -------
        :class:`.Visual`
            A deep copy of the visual.
        """
        origin=self.origin.copy()
        if scale is not None:
            if not isinstance(scale, (list, np.ndarray)):
                scale = np.repeat(scale, 3)
            origin[:3,3] *= scale
        return Collision(
            name='{}{}'.format(prefix, self.name),
            origin=origin,
            geometry=self.geometry.copy(prefix=prefix, scale=scale),
        )
