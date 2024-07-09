
from __future__ import annotations

import numpy as np
import constants as const
from typing import Union, List
import skimage
from enum import Enum
from visualization import visualize_data as vis
from data_generators.object_recognition.basic_geometry import Point, Vector, Bbox, Orientation, Dimension2D

np.random.seed(const.RANDOM_SEED_FOR_NUMPY)
MAX_PAD_SIZE = const.MAX_PAD_SIZE


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


# TODO: This is not used at the moment. Either use it or delete it.
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

    def __init__(self, name: str = 'Zero', height: int = 3, width: int = 3, actual_pixels: Union[None, np.ndarray] = None,
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
        self.symmetries: List = []

        self.reset_dimensions()

        self.child_objects = {}

    def reset_dimensions(self):
        self.dimensions.dx = self.actual_pixels.shape[1]
        self.dimensions.dy = self.actual_pixels.shape[0]

        bb_top_left = Point(self.canvas_pos[0], self.canvas_pos[1] + self.dimensions.dy - 1)
        bb_bottom_right = Point(bb_top_left.x + self.dimensions.dx - 1, self.canvas_pos[1])

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
        :param center: The point of the axis of rotation
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

        center += np.array([self.canvas_pos[0], self.canvas_pos[1], 0])
        self.bbox.transform(translation=-center)
        self.bbox.transform(rotation=radians)
        self.bbox.transform(translation=center)

        for sym in self.symmetries:
            sym.transform(translation=-center)
            sym.transform(rotation=radians)
            sym.transform(translation=center)

        self.dimensions.dx = self.actual_pixels.shape[1]
        self.dimensions.dy = self.actual_pixels.shape[0]

    def mirror(self, axis: Orientation, on_axis=False):
        if axis == Orientation.Up or axis == Orientation.Down:
            concat_pixels = np.flipud(self.actual_pixels)
            if on_axis:
                concat_pixels = concat_pixels[:-1, :] if axis == Orientation.Up else concat_pixels[1:, :]

            self.actual_pixels = np.concatenate((self.actual_pixels, concat_pixels), axis=0) if axis == Orientation.Down else \
                np.concatenate((concat_pixels, self.actual_pixels), axis=0)

            new_symmetry_axis_origin = Point(self.canvas_pos[0], self.actual_pixels.shape[0] / 2 + self.canvas_pos[1]) \
                if axis == Orientation.Up else Point(self.canvas_pos[0], self.canvas_pos[1])
            new_symmetry_axis_origin.y -= 0.5
            if on_axis and axis == Orientation.Down:
                new_symmetry_axis_origin.y -= 0.5

            if on_axis and axis == Orientation.Down:
                for sym in self.symmetries:
                    if sym.orientation == Orientation.Right and sym.origin.y > new_symmetry_axis_origin.y:
                        sym.origin.y -= 1

            symmetry_vector = Vector(orientation=Orientation.Right, origin=new_symmetry_axis_origin,
                                     length=self.actual_pixels.shape[1])

        elif axis == Orientation.Left or axis == Orientation.Right:
            concat_pixels = np.fliplr(self.actual_pixels)
            if on_axis:
                concat_pixels = concat_pixels[:, 1:] if axis == Orientation.Right else concat_pixels[:, :-1]

            self.actual_pixels = np.concatenate((self.actual_pixels, concat_pixels), axis=1) if axis == Orientation.Right else \
                np.concatenate((concat_pixels, self.actual_pixels), axis=1)

            new_symmetry_axis_origin = Point(self.actual_pixels.shape[1] / 2 + self.canvas_pos[0], self.canvas_pos[1])\
                if axis == Orientation.Right else Point(self.canvas_pos[0], self.canvas_pos[1])
            new_symmetry_axis_origin.x -= 0.5
            if on_axis and axis == Orientation.Left:
                new_symmetry_axis_origin.x -= 0.5
            symmetry_vector = Vector(orientation=Orientation.Up, origin=new_symmetry_axis_origin,
                                     length=self.actual_pixels.shape[0])

            if on_axis and axis == Orientation.Left:
                for sym in self.symmetries:
                    if sym.orientation == Orientation.Up and sym.origin.x > new_symmetry_axis_origin.x:
                        sym.origin.x -= 1

        if axis == Orientation.Left:
            self.canvas_pos[0] -= self.dimensions.dx
        if axis == Orientation.Down:
            self.canvas_pos[1] -= self.dimensions.dy

        self.symmetries.append(symmetry_vector)

        self.reset_dimensions()

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
        xmin = self.bbox.top_left.x - 0.5
        xmax = self.bbox.bottom_right.x + 0.5
        ymin = self.bbox.bottom_right.y - 0.5
        ymax = self.bbox.top_left.y + 0.5
        extent = [xmin, xmax, ymin, ymax]
        fig, ax = vis.plot_data(self.actual_pixels, extent=extent)

        if symmetries_on:
            for sym in self.symmetries:

                line_at = sym.origin.x if (sym.orientation == Orientation.Up or sym.orientation == Orientation.Down) else sym.origin.y

                line_min = self.bbox.top_left.y + 0.5 if (sym.orientation == Orientation.Up or sym.orientation == Orientation.Down) else self.bbox.top_left.x - 0.5
                line_max = self.bbox.bottom_right.y - 0.5 if (sym.orientation == Orientation.Up or sym.orientation == Orientation.Down) else self.bbox.bottom_right.x + 0.5

                plt_lines = ax.vlines if (sym.orientation == Orientation.Up or sym.orientation == Orientation.Down) else ax.hlines
                plt_lines(line_at, line_min, line_max, linewidth=2)
