import copy

from lxml import etree as ET
import numpy as np
import PIL
import six

from urdfpy.urdf.base import URDFType
from urdfpy.urdf.geometry import Geometry, Collision
from urdfpy.utils import (
    parse_origin,
    unparse_origin,
    get_filename,
    configure_origin
)

class Texture(URDFType):
    """An image-based texture, as an embedded element in Material

    Parameters
    ----------
    filename : str
        The path to the image that contains this texture. This can be
        relative to the top-level URDF or an absolute path.
    image : :class:`PIL.Image.Image`, optional
        The image for the texture.
        If not specified, it is loaded automatically from the filename.
    """

    _ATTRIBS = {
        'filename': (str, True)
    }
    _TAG = 'texture'

    def __init__(self, filename, image=None):
        if image is None:
            image = PIL.image.open(filename)
        self.filename = filename
        self.image = image

    @property
    def filename(self):
        """str : Path to the image for this texture.
        """
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = str(value)

    @property
    def image(self):
        """:class:`PIL.Image.Image` : The image for this texture.
        """
        return self._image

    @image.setter
    def image(self, value):
        if isinstance(value, str):
            value = PIL.Image.open(value)
        if isinstance(value, np.ndarray):
            value = PIL.Image.fromarray(value)
        elif not isinstance(value, PIL.Image.Image):
            raise ValueError('Texture only supports numpy arrays '
                             'or PIL images')
        self._image = value

    @classmethod
    def _from_xml(cls, node, path):
        kwargs = cls._parse(node, path)

        # Load image
        fn = get_filename(path, kwargs['filename'])
        kwargs['image'] = PIL.Image.open(fn)

        return Texture(**kwargs)

    def _to_xml(self, parent, path):
        # Save the image
        filepath = get_filename(path, self.filename, makedirs=True)
        self.image.save(filepath)

        return self._unparse(path)

    def copy(self, prefix='', scale=None):
        """Create a deep copy with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all names.

        Returns
        -------
        :class:`.Texture`
            A deep copy.
        """
        v = Texture(
            filename=self.filename,
            image=self.image.copy()
        )
        return v

class Material(URDFType):
    """A material for a Visual object, associated with some geometry.

    Parameters
    ----------
    name : str
        The name of the material.
    color : (4,) float, optional
        The RGBA color of the material in the range [0,1].
    texture : :class:`.Texture`, optional
        A texture for the material.
    """
    _ATTRIBS = {
        'name': (str, True)
    }
    _ELEMENTS = {
        'texture': (Texture, False, False),
    }
    _TAG = 'material'

    def __init__(self, name, color=None, texture=None):
        self.name = name
        self.color = color
        self.texture = texture

    @property
    def name(self):
        """str : The name of the material.
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def color(self):
        """(4,) float : The RGBA color of the material, in the range [0,1].
        """
        return self._color

    @color.setter
    def color(self, value):
        if value is not None:
            value = np.asanyarray(value).astype(float)
            value = np.clip(value, 0.0, 1.0)
            if value.shape != (4,):
                raise ValueError('Color must be a (4,) float')
        self._color = value

    @property
    def texture(self):
        """:class:`.Texture` : The texture for the material.
        """
        return self._texture

    @texture.setter
    def texture(self, value):
        if value is not None:
            if isinstance(value, six.string_types):
                image = PIL.Image.open(value)
                value = Texture(filename=value, image=image)
            elif not isinstance(value, Texture):
                raise ValueError('Invalid type for texture -- expect path to '
                                 'image or Texture')
        self._texture = value

    @classmethod
    def _from_xml(cls, node, path):
        kwargs = cls._parse(node, path)

        # Extract the color -- it's weirdly an attribute of a subelement
        color = node.find('color')
        if color is not None:
            color = np.fromstring(color.attrib['rgba'], sep=' ', dtype=np.float64)
        kwargs['color'] = color

        return Material(**kwargs)

    def _to_xml(self, parent, path):
        # Simplify materials by collecting them at the top level.

        # For top-level elements, save the full material specification
        if parent.tag == 'robot':
            node = self._unparse(path)
            if self.color is not None:
                color = ET.Element('color')
                color.attrib['rgba'] = np.array2string(self.color)[1:-1]
                node.append(color)

        # For non-top-level elements just save the material with a name
        else:
            node = ET.Element('material')
            node.attrib['name'] = self.name
        return node

    def copy(self, prefix='', scale=None):
        """Create a deep copy of the material with the prefix applied to all names.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all joint and link names.

        Returns
        -------
        :class:`.Material`
            A deep copy of the material.
        """
        return Material(
            name='{}{}'.format(prefix, self.name),
            color=self.color,
            texture=self.texture
        )

class Visual(URDFType):
    """Visual properties of a link.

    Parameters
    ----------
    geometry : :class:`.Geometry`
        The geometry of the element
    name : str, optional
        The name of the visual geometry.
    origin : (4,4) float, optional
        The pose of the visual element relative to the link frame.
        Defaults to identity.
    material : :class:`.Material`, optional
        The material of the element.
    """
    _ATTRIBS = {
        'name': (str, False)
    }
    _ELEMENTS = {
        'geometry': (Geometry, True, False),
        'material': (Material, False, False),
    }
    _TAG = 'visual'

    def __init__(self, geometry, name=None, origin=None, material=None):
        self.geometry = geometry
        self.name = name
        self.origin = origin
        self.material = material

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
        """str : The name of this visual element.
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

    @property
    def material(self):
        """:class:`.Material` : The material for this element.
        """
        return self._material

    @material.setter
    def material(self, value):
        if value is not None:
            if not isinstance(value, Material):
                raise TypeError('Must set material with Material object')
        self._material = value

    @classmethod
    def _from_xml(cls, node, path):
        kwargs = cls._parse(node, path)
        kwargs['origin'] = parse_origin(node)
        return Visual(**kwargs)

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
        return Visual(
            geometry=self.geometry.copy(prefix=prefix, scale=scale),
            name='{}{}'.format(prefix, self.name),
            origin=origin,
            material=(self.material.copy(prefix=prefix) if self.material else None),
        )

class Link(URDFType):
    """A link of a rigid object.

    Parameters
    ----------
    name : str
        The name of the link.
    visuals : list of :class:`.Visual`, optional
        The visual properties of the link.
    collsions : list of :class:`.Collision`, optional
        The collision properties of the link.
    """

    _ATTRIBS = {
        'name': (str, True),
    }
    _ELEMENTS = {
        'visuals': (Visual, False, True),
        'collisions': (Collision, False, True),
    }
    _TAG = 'link'

    def __init__(self, name, visuals, collisions):
        self.name = name
        self.visuals = visuals
        self.collisions = collisions

        self._collision_mesh = None

    @property
    def name(self):
        """str : The name of this link.
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def visuals(self):
        """list of :class:`.Visual` : The visual properties of this link.
        """
        return self._visuals

    @visuals.setter
    def visuals(self, value):
        if value is None:
            value = []
        else:
            value = list(value)
            for v in value:
                if not isinstance(v, Visual):
                    raise ValueError('Expected list of Visual objects')
        self._visuals = value

    @property
    def collisions(self):
        """list of :class:`.Collision` : The collision properties of this link.
        """
        return self._collisions

    @collisions.setter
    def collisions(self, value):
        if value is None:
            value = []
        else:
            value = list(value)
            for v in value:
                if not isinstance(v, Collision):
                    raise ValueError('Expected list of Collision objects')
        self._collisions = value

    @property
    def collision_mesh(self):
        """:class:`open3d.geometry.TriangleMesh` : A single collision mesh for
        the link, specified in the link frame, or None if there isn't one.
        """
        if len(self.collisions) == 0:
            return None
        if self._collision_mesh is None:
            meshesList = []
            # list of meshes from ALL collision objects
            for c in self.collisions:
                m = c.geometry.meshes
                # NOTE: m here is a o3d.geometry.TriangleMesh
                m = copy.deepcopy(m)
                pose = c.origin
                if c.geometry.mesh is not None:
                    # c.geometry.mesh  is  the  Mesh  object,  an
                    # instance of the derived class from URDFType
                    if c.geometry.mesh.scale is not None:
                        S = np.eye(4)
                        S[:3,:3] = np.diag(c.geometry.mesh.scale)
                        pose = pose.dot(S)
                m.transform(pose)
                meshesList.append(m)
            if len(meshesList) == 0:
                return None
            self._collision_mesh = meshesList[0]
            for otherMesh in meshesList[1:]:
                self._collision_mesh = self._collision_mesh + otherMesh
        return self._collision_mesh

    def copy(self, prefix='', scale=None, collision_only=False):
        """Create a deep copy of the link.

        Parameters
        ----------
        prefix : str
            A prefix to apply to all joint and link names.

        Returns
        -------
        link : :class:`.Link`
            A deep copy of the Link.
        """
        cm = self._collision_mesh
        if scale is not None and self.collision_mesh is not None:
            sm = np.eye(4)
            if not isinstance(scale, (list, np.ndarray)):
                scale = np.repeat(scale, 3)
            sm[:3,:3] = np.diag(scale)
            cm = copy.deepcopy(self.collision_mesh)
            cm.transform(sm)

        visuals = None
        if not collision_only:
            visuals=[v.copy(prefix=prefix, scale=scale) for v in self.visuals]

        cpy = Link(
            name='{}{}'.format(prefix, self.name),
            visuals=visuals,
            collisions=[v.copy(prefix=prefix, scale=scale) for v in self.collisions],
        )
        cpy._collision_mesh = cm
        return cpy

