
from __future__ import annotations

import numpy as np
import constants as const
from typing import Union, List, Tuple
import skimage
from dataclasses import dataclass
from enum import Enum
from visualization import visualize_data as vis

np.random.seed(const.RANDOM_SEED_FOR_NUMPY)
MAX_PAD_SIZE = const.MAX_PAD_SIZE


class Orientation(Enum):
    Up: int = 0
    Up_Right: int = 1
    Right: int = 2
    Down_Right: int = 3
    Down: int = 4
    Down_Left: int = 5
    Left: int = 6
    Up_Left: int = 7

    def rotate(self, affine_matrix: Union[np.ndarray | None] = None, rotation: float = 0):
        assert affine_matrix is None or rotation == 0

        times = 0
        if rotation != 0:
            times = rotation // (np.pi/4)
        if affine_matrix is not None:
            if np.all(affine_matrix[:2, :2].astype(int) == np.array([[0, -1], [1, 0]])):
                times = 2
            if np.all(affine_matrix[:2, :2].astype(int) == np.array([[-1, 0], [0, -1]])):
                times = 4
            if np.all(affine_matrix[:2, :2].astype(int) == np.array([[0, 1], [-1, 0]])):
                times = 6

        value = self.value - times
        if value < 0:
            value = 8 + value

        return Orientation(value=value)


@dataclass
class Dimension2D:
    dx: int = 3
    dy: int = 3


class Point:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0, array: Union[None, List, np.ndarray] = None):
        if array is None:
            self.x = x
            self.y = y
            self.z = z  # This is used to define over or under
        else:
            self.x = array[0]
            self.y = array[1]
            self.z = array[2]

    def __add__(self, other) -> Point:
        return Point(self.x+other.x, self.y+other.y, self.z+other.z)

    def __sub__(self, other) -> Point:
        return Point(self.x-other.x, self.y-other.y, self.z-other.z)

    def __repr__(self) -> str:
        return f'Point(X = {self.x}, Y = {self.y}, Z = {self.z})'

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def __eq__(self, other: Union[List, np.ndarray, Tuple]) -> bool:
        return np.all(self.x == other[0], self.y == other[1], self.z == other[2])

    @staticmethod
    def point_from_numpy(array: np.ndarray):
        return Point(array[0], array[1], array[2])

    def transform(self, affine_matrix: Union[np.ndarray | None] = None,
                  rotation: float = 0,
                  shear: Union[List | np.ndarray | Point | None] = None,
                  translation: Union[List | np.ndarray | Point | None] = None,
                  scale: Union[List | np.ndarray | Point | None] = None):

        assert affine_matrix is None or np.all([rotation == 0, shear is None, translation is None, scale is None])

        x = self.x
        y = self.y

        if affine_matrix is not None:
            a0 = affine_matrix[0, 0]
            a1 = affine_matrix[0, 1]
            a2 = affine_matrix[0, 2]
            b0 = affine_matrix[1, 0]
            b1 = affine_matrix[1, 1]
            b2 = affine_matrix[1, 2]

            self.x = a0 * x + a1 * y + a2
            self.y = b0 * x + b1 * y + b2
        else:
            if translation is not None:
                translation_x = translation[0] if type(translation) != Point else translation.x
                translation_y = translation[1] if type(translation) != Point else translation.y
            else:
                translation_x = 0
                translation_y = 0

            if scale is not None:
                scale_x = scale[0] if type(scale) != Point else scale.x
                scale_y = scale[1] if type(scale) != Point else scale.y
            else:
                scale_x = 1
                scale_y = 1

            if shear is not None:
                shear_x = shear[0] if type(shear) != Point else shear.x
                shear_y = shear[1] if type(shear) != Point else shear.y
            else:
                shear_x = 0
                shear_y = 0

            self.x = scale_x * x * (np.cos(rotation) + np.tan(shear_y) * np.sin(rotation)) - \
                     scale_y * y * (np.tan(shear_x) * np.cos(rotation) + np.sin(rotation)) + translation_x

            self.y = scale_x * x * (np.sin(rotation) - np.tan(shear_y) * np.cos(rotation)) - \
                     scale_y * y * (np.tan(shear_x) * np.sin(rotation) - np.cos(rotation)) + translation_y


class Vector:
    def __init__(self, orientation:Orientation = Orientation.Up,
                 length: Union[None, int] = 0,
                 origin: Point = Point(0, 0, 0)):
        self.orientation = orientation
        self.length = length
        self.origin = origin

    def __repr__(self):
        return f'Vector(Orientation: {self.orientation}, Length: {self.length}, Origin Point: {self.origin})'

    def transform(self, affine_matrix: Union[np.ndarray | None] = None,
                  rotation: float = 0,
                  shear: Union[List | np.ndarray | Point | None] = None,
                  translation: Union[List | np.ndarray | Point | None] = None,
                  scale: Union[List | np.ndarray | Point | None] = None):
        assert affine_matrix is None or np.all([rotation == 0, shear is None, translation is None, scale is None])

        if rotation != 0:
            self.orientation = self.orientation.rotate(rotation=rotation)
        if affine_matrix is not None:
            self.orientation = self.orientation.rotate(affine_matrix=affine_matrix)

        self.origin.transform(affine_matrix, rotation, shear, translation, scale)


class Bbox:
    def __init__(self, top_left: Point = Point(0, 0), bottom_right: Point = Point(0, 0)):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.center: Point = self._calculate_center()

    def _calculate_center(self):
        center = Point(x=(self.bottom_right.x - self.top_left.x) / 2,
                       y=(self.bottom_right.y - self.top_left.y)/2)
        return center

    def __getattr__(self, center: str) -> Point:
        self.center = self._calculate_center()
        return self.center

    def __deepcopy__(self, memo):
        return self

    def __repr__(self):
        return f'Bbox(Top Left: {self.top_left}, Bottom Right: {self.bottom_right}, Center: {self.center})'

    def transform(self, affine_matrix: Union[np.ndarray | None] = None,
                  rotation: float = 0,
                  shear: Union[List | np.ndarray | Point | None] = None,
                  translation: Union[List | np.ndarray | Point | None] = None,
                  scale: Union[List | np.ndarray | Point | None] = None):
        assert affine_matrix is None or np.all([rotation == 0, shear is None, translation is None, scale is None])

        rot = 0
        if rotation == np.pi/2:
            rot = 1
        if affine_matrix is not None and np.all(affine_matrix[:2, :2].astype(int) == np.array([[0, -1], [1, 0]])):
            rot = 1
        if rotation == np.pi:
            rot = 2
        if affine_matrix is not None and np.all(affine_matrix[:2, :2].astype(int) == np.array([[-1, 0], [0, -1]])):
            rot = 2
        if rotation == 3*np.pi/2:
            rot = 3
        if affine_matrix is not None and np.all(affine_matrix[:2, :2].astype(int) == np.array([[0, 1], [-1, 0]])):
            rot = 3

        if rot == 1:
            p1 = self.top_left
            p1.transform(affine_matrix, rotation, shear, translation, scale)
            p2 = self.bottom_right
            p2.transform(affine_matrix, rotation, shear, translation, scale)
            self.top_left = Point.point_from_numpy(np.array([p1.x, p2.y, 0]))
            self.bottom_right = Point.point_from_numpy(np.array([p2.x, p1.y, 0]))

        elif rot == 2:
            p = self.top_left
            self.top_left = self.bottom_right
            self.bottom_right = p

        elif rot == 3:
            p1 = self.top_left
            p1.transform(affine_matrix, rotation, shear, translation, scale)
            p2 = self.bottom_right
            p2.transform(affine_matrix, rotation, shear, translation, scale)

            self.top_left = Point.point_from_numpy(np.array([p2.x, p1.y, 0]))
            self.bottom_right = Point.point_from_numpy(np.array([p1.x, p2.y, 0]))

        else:
            self.top_left.transform(affine_matrix, rotation, shear, translation, scale)
            self.bottom_right.transform(affine_matrix, rotation, shear, translation, scale)

        self.center = self.center


@dataclass
class Symmetry:
    bbox: Bbox = Bbox()
    axis: Vector = Vector()


class Example:
    def __init__(self):
        self.number_of_canvasses = np.random.randint(4, const.MAX_EXAMPLE_PAIRS + 2, 2)
        self.input_size = Dimension2D(np.random.randint(const.MIN_PAD_SIZE, const.MAX_PAD_SIZE),
                                      np.random.randint(const.MIN_PAD_SIZE, const.MAX_PAD_SIZE))
        self.output_size = Dimension2D(np.random.randint(self.input_size.dx, const.MAX_PAD_SIZE),
                                       np.random.randint(self.input_size.dy, const.MAX_PAD_SIZE))
        print(self.input_size, self.output_size)

    def generate_canvasses(self):
        pass


class Canvas:
    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        self.objects = []


class ObjectType(Enum):
    Point: int = 0
    Parallelogram: int = 1
    Cross: int = 2
    Line: int = 3
    Pi: int = 4
    Angle: int = 5
    ZigZag: int = 6
    Fish: int = 7
    Bolt: int = 8
    Spiral: int = 9
    InverseCross: int = 10
    Tie: int = 11
    Random: int = 12


class Object:

    def __init__(self, height: int = 3, width: int = 3, actual_pixels: Union[None, np.ndarray] = None,
                 imagined_pixels: Union[None, np.ndarray] = None,
                 canvas_pos: Union[List | np.ndarray] = (0, 0)):
        if actual_pixels is None:
            self.actual_pixels = np.ones((height, width))
        else:
            self.actual_pixels = actual_pixels

        if imagined_pixels is None:
            self.imagined_pixels = self.actual_pixels
        else:
            self.imagined_pixels = imagined_pixels

        self.canvas_pos = np.array(canvas_pos)
        self.dimensions = Dimension2D(self.actual_pixels.shape[1], self.actual_pixels.shape[0])

        self.number_of_coloured_pixels: int = int(np.sum([1 for i in self.actual_pixels for k in i if k > 1]))
        self.symmetry: Symmetry = Symmetry()

        self.reset_dimensions()

        self.child_objects = {}

    def reset_dimensions(self, translation: Dimension2D = Dimension2D(0, 0)):
        self.dimensions.dx = self.actual_pixels.shape[1]
        self.dimensions.dy = self.actual_pixels.shape[0]

        bb_top_left = Point(self.canvas_pos[0] + translation.dx, self.canvas_pos[1] + translation.dy + self.dimensions.dy)
        bb_bottom_right = Point(bb_top_left.x + self.dimensions.dx, self.canvas_pos[1] + translation.dy)

        self.bbox = Bbox(top_left=bb_top_left, bottom_right=bb_bottom_right)

    def scale(self, factor: int):
        """
        Scales the object. A positive factor adds pixels and a negative factor removes pixels.
        :param factor: Integer
        :return: ndArray. The scaled pic
        """
        assert factor != 0, print('factor cannot be 0')

        pic = self.actual_pixels

        if factor < 0:
            assert np.abs(factor) * np.min(pic.shape) < 3, print(
                f'Downsizing by {np.abs(factor)} will result in too small an image')

        if factor > 0:
            scaled = np.ones(np.array(pic.shape) * factor)
            for x in range(pic.shape[0]):
                for y in range(pic.shape[1]):
                    scaled[x * factor:(x + 1) * factor, y * factor:(y + 1) * factor] = pic[x, y]
        else:
            factor = np.abs(factor)
            scaled = np.ones(np.ceil(np.array(pic.shape) / factor).astype(np.int32))
            for x in range(scaled.shape[0]):
                for y in range(scaled.shape[1]):
                    scaled[x, y] = pic[x * factor, y * factor]

        self.actual_pixels = scaled
        self.reset_dimensions()

    def rotate(self, times: Union[1, 2, 3], center: Union[np.ndarray, List, Point] = (0, 0)):
        """
        Rotate the object counter-clockwise by times multiple of 90 degrees
        :param times: 1, 2 or 3 times
        :return:
        """
        radians = np.pi/2 * times
        degrees = 90 * times
        self.actual_pixels = skimage.transform.rotate(self.actual_pixels, degrees, resize=True, order=0, center=center)
        self.imagined_pixels = skimage.transform.rotate(self.imagined_pixels, degrees, resize=True, order=0, center=center)

        if type(center) == Point:
            center = center.to_numpy()
        if len(center) == 2:
            center = np.array([center[0], center[1], 0])

        print(center)
        print(self.canvas_pos)
        center += np.array([self.canvas_pos[0], self.canvas_pos[1], 0])
        print(center)
        print('--------')
        print(self.bbox)
        self.bbox.transform(translation=-center)
        print(-center)
        print(self.bbox)
        self.bbox.transform(rotation=radians)
        print(self.bbox)
        print(center)
        self.bbox.transform(translation=center)
        print(self.bbox)
        self.dimensions.dx = self.actual_pixels.shape[1]
        self.dimensions.dy = self.actual_pixels.shape[0]

    def mirror(self, axis: Orientation, on_axis=False):
        if axis == Orientation.Up or axis == Orientation.Down:
            concat_pixels = np.flipud(self.actual_pixels)
            if on_axis:
                concat_pixels = concat_pixels[:-1, :] if axis == Orientation.Up else concat_pixels[1:, :]

            self.actual_pixels = np.concatenate((self.actual_pixels, concat_pixels), axis=0) if axis == Orientation.Down else \
                np.concatenate((concat_pixels, self.actual_pixels), axis=0)
            new_symmetry_axis_origin = Point(0, self.actual_pixels.shape[0] / 2 - 0.5)
            new_symmetry_axis = Vector(orientation=Orientation.Right, origin=new_symmetry_axis_origin,
                                       length=self.actual_pixels.shape[1])

        elif axis == Orientation.Left or axis == Orientation.Right:
            concat_pixels = np.fliplr(self.actual_pixels)
            if on_axis:
                concat_pixels = concat_pixels[:, -1] if axis == Orientation.Right else concat_pixels[:, 1:]
            self.actual_pixels = np.concatenate((self.actual_pixels, concat_pixels), axis=1) if axis == Orientation.Right else \
                np.concatenate((concat_pixels, self.actual_pixels), axis=1)
            new_symmetry_axis_origin = Point(self.actual_pixels.shape[1] / 2 - 0.5, 0)
            new_symmetry_axis = Vector(orientation=Orientation.Down, origin=new_symmetry_axis_origin,
                                       length=self.actual_pixels.shape[0])

        bbox_translation = Dimension2D(0, 0)
        if axis == Orientation.Left:
            bbox_translation = Dimension2D(concat_pixels.shape[1], 0)
        if axis == Orientation.Up:
            bbox_translation = Dimension2D(0, concat_pixels.shape[0])

        #TODO fix the symmetry
        #self.symmetries.append(Symmetry(axis=new_symmetry_axis, bbox=cp.deepcopy(self.bbox)))
        self.reset_dimensions(translation=bbox_translation)

    def flip(self, axis: Orientation, on_axis=False):
        """
        Flips the object along an axis and possibly copies it
        :param axis: The direction to flip towards. The edge of the bbox toward that direction becomes the axis of the flip.
        :param on_axis: Should the flip happen on axis or over the axis
        :param copy: Whether to keep the original and add the flipped copy attached to the original or just flip in place
        :return: Nothing
        """

        if axis == Orientation.Up or axis == Orientation.Down:
            self.actual_pixels = np.flipud(self.actual_pixels)
        elif axis == Orientation.Left or axis == Orientation.Right:
            self.actual_pixels = np.fliplr(self.actual_pixels)

    def copy(self):
        pass

    def add(self, other: Object):
        pass

    def subtract(self, other: Object):
        pass

    def superimpose(self, other: Object, z_order :int = 1):
        pass

    def move_along_z(self):
        pass

    def show(self, symmetries_on=True):
        xmin = self.bbox.top_left.x + 0.5
        xmax = self.bbox.bottom_right.x + 0.5
        ymin = self.bbox.bottom_right.y + 0.5
        ymax = self.bbox.top_left.y + 0.5
        extent = [xmin , xmax , ymin , ymax]
        print(extent)
        fig, ax = vis.plot_data(self.actual_pixels, extent=extent)

        '''
        if symmetries_on:
            for sym in self.symmetries:
                if sym.axis.orientation == Orientation.Down:
                    xs = [sym.axis.origin.x, sym.bbox.bottom_right]
                    ys = [sym.axis.origin.y, ]
                elif sym.axis.orientation == Orientation.Right:
                    xs = [sym.axis.origin.x, ]
                    ys = [sym.axis.origin.y, ]
                ax.plot(xs, ys)

                line_at = sym.axis.origin.x if sym.axis.orientation == Orientation.Down else sym.axis.origin.y
                line_min = sym.bbox.top_left.y - 0.5 if sym.axis.orientation == Orientation.Down else sym.bbox.top_left.x - 0.5
                line_max = sym.bbox.bottom_right.y + 0.5 if sym.axis.orientation == Orientation.Down else sym.bbox.bottom_right.x + 0.5

                #plt_lines = plt.vlines if sym.axis.orientation == Orientation.Down else plt.hlines
                #plt_lines(line_at, line_min, line_max)
        '''