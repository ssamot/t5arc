
import numpy as np
import constants as const
from typing import Union, List
import skimage
from dataclasses import dataclass, field
from enum import Enum
import copy as cp
from data_generators.object_recognition.object_recognition_output_dict import *


np.random.seed(const.RANDOM_SEED_FOR_NUMPY)
MAX_PAD_SIZE = const.MAX_PAD_SIZE


class Colour(Enum):
    Purple: int = 1
    Burgundy: int = 2
    Red: int = 3
    Orange: int = 4
    Yellow: int = 5
    Green: int = 6
    Blue: int = 7
    Aqua: int = 8
    Grey: int = 9
    Black: int = 10


class Orientation(Enum):
    Up: int = 0
    Up_Right: int = 1
    Right: int = 2
    Down_Right: int = 3
    Down: int = 4
    Down_Left: int = 5
    Left: int = 6
    Up_Left: int = 7


@dataclass
class Dimension2D:
    dx: int = 3
    dy: int = 3


class Point:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z  # This is used to define over or under

    def __add__(self, other):
        return Point(self.x+other.x, self.y+other.y, self.z+other.z)

    def __sub__(self, other):
        return Point(self.x-other.x, self.y-other.y, self.z-other.z)

    def __repr__(self):
        return f'Point(X = {self.x}, Y = {self.y}, Z = {self.z})'


class Vector:
    def __init__(self, orientation:Orientation = Orientation.Up,
                 length: Union[None, int] = 1,
                 origin: Point = Point(0, 0, 0)):
        self.orientation = orientation
        self.length = length
        self.origin = origin

    def __repr__(self):
        return f'Vector(Orientation: {self.orientation}, Length: {self.length}, Origin Point: {self.origin})'

class Bbox:
    def __init__(self, top_left: Point = Point(0, 0), bottom_right: Point = Point(1, 1)):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.center: Point = self._calculate_center()

    def _calculate_center(self):
        center = Point(x=(self.bottom_right.x - self.top_left.x) / 2 - 0.5,
                       y=(self.bottom_right.y - self.top_left.y)/2 - 0.5)
        return center

    def __getattr__(self, center: str) -> Point:
        self.center = self._calculate_center()
        return self.center

    def __deepcopy__(self, memo):
        return self

    def __repr__(self):
        return f'Bbox(Top Left: {self.top_left}, Bottom Right: {self.bottom_right}, Center: {self.center})'


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


class Object:

    def __init__(self, height: int = 3, width: int = 3, pixels: Union[None, np.ndarray] = None,
                 canvas_pos: Point = Point(0, 0)):
        if pixels is None:
            self.pixels = np.ones((height, width))
        else:
            self.pixels = pixels
        self.dimensions = Dimension2D(self.pixels.shape[1], self.pixels.shape[0])
        self.bbox = Bbox(top_left=canvas_pos, bottom_right=canvas_pos+Point(self.dimensions.dx, self.dimensions.dy))
        self.number_of_coloured_pixels: int = 0
        self.symmetries: List[Symmetry] = []
        self.orientation: Union[Orientation, None]

    def reset_dimensions(self, translation: Dimension2D = Dimension2D(0, 0)):
        self.dimensions.dx = self.pixels.shape[1]
        self.dimensions.dy = self.pixels.shape[0]
        bb_top_left = Point(self.bbox.top_left.x + translation.dx, self.bbox.top_left.y + translation.dy)
        bb_bottom_right = Point(bb_top_left.x + self.dimensions.dx, bb_top_left.y + self.dimensions.dy)
        self.bbox = Bbox(top_left=bb_top_left, bottom_right=bb_bottom_right)

    def scale(self, factor: int):
        """
        Scales the object. A positive factor adds pixels and a negative factor removes pixels.
        :param factor: Integer
        :return: ndArray. The scaled pic
        """
        assert factor != 0, print('factor cannot be 0')

        pic = self.pixels

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

        self.pixels = scaled
        self.reset_dimensions()

    def rotate(self, times: Union[1, 2, 3]):
        """
        Rotate the object by some multiple of 90 degrees
        :param times: 1, 2 or 3 times
        :return:
        """

        pic = self.pixels

        degrees = 90 * times

        self.pixels = skimage.transform.rotate(pic, degrees)

    def flip(self, dir: Orientation, on_axis=False, copy=False):
        """
        Flips the object along an axis.
        :param dir: The direction to flip towards. The edge of the bbox toward that direction becomes the axis of the flip.
        :param on_axis: Should the flip happen on axis or over the axis
        :param copy: Whether to keep the original or not
        :return:
        """
        if copy:
            if dir == Orientation.Up or dir == Orientation.Down:
                concat_pixels = np.flipud(self.pixels)
                if on_axis:
                    concat_pixels = concat_pixels[:-1, :] if dir == Orientation.Up else concat_pixels[1:, :]

                self.pixels = np.concatenate((self.pixels, concat_pixels), axis=0) if dir == Orientation.Down else \
                    np.concatenate((concat_pixels, self.pixels), axis=0)
                new_symmetry_axis_origin = Point(0, self.pixels.shape[0] / 2 - 0.5)
                new_symmetry_axis = Vector(orientation=Orientation.Right, origin=new_symmetry_axis_origin,
                                           length=self.pixels.shape[1])

            elif dir == Orientation.Left or dir == Orientation.Right:
                concat_pixels = np.fliplr(self.pixels)
                if on_axis:
                    concat_pixels = concat_pixels[:, -1] if dir == Orientation.Right else concat_pixels[:, 1:]
                self.pixels = np.concatenate((self.pixels, concat_pixels), axis=1) if dir == Orientation.Right else \
                    np.concatenate((concat_pixels, self.pixels), axis=1)
                new_symmetry_axis_origin = Point(self.pixels.shape[1] / 2 - 0.5, 0)
                new_symmetry_axis = Vector(orientation=Orientation.Down, origin=new_symmetry_axis_origin,
                                           length=self.pixels.shape[0])

            bbox_translation = Dimension2D(0, 0)
            if dir == Orientation.Left:
                bbox_translation = Dimension2D(concat_pixels.shape[1], 0)
            if dir == Orientation.Up:
                bbox_translation = Dimension2D(0, concat_pixels.shape[0])

            self.symmetries.append(Symmetry(axis=new_symmetry_axis, bbox=cp.deepcopy(self.bbox)))
            self.reset_dimensions(translation=bbox_translation)

        else:
            if dir == Orientation.Up or dir == Orientation.Down:
                self.pixels = np.flipud(self.pixels)
            elif dir == Orientation.Left or dir == Orientation.Right:
                self.pixels = np.fliplr(self.pixels)



