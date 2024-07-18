
from __future__ import annotations

import numpy as np
import numpy.lib.index_tricks as ndi
import constants as const
from typing import List
from enum import Enum
from data_generators.object_recognition.object import Object
from data_generators.object_recognition.basic_geometry import Point, Bbox, Dimension2D, Orientation, OrientationZ, Vector

np.random.seed(const.RANDOM_SEED_FOR_NUMPY)
MAX_PAD_SIZE = const.MAX_PAD_SIZE


# TODO: This is not used at the moment. Either use it or delete it.
class ObjectType(Enum):
    Dot: int = 0
    Parallelogram: int = 1
    Cross: int = 2
    Hole: int = 3
    Pi: int = 4
    Angle: int = 5
    ZigZag: int = 6
    Fish: int = 7
    Bolt: int = 8
    Spiral: int = 9
    InverseCross: int = 10
    Tie: int = 11
    Random: int = 12


class Dot(Object):
    def __init__(self, size: None | np.ndarray | Dimension2D | List = np.array([1, 1]), canvas_pos: Point = Point(0, 0),
                 colour: None | int = None, dot_pos: None | np.ndarray | List = np.array([0, 0]), id: None | int = None):
        """
        A single coloured pixel in a black background
        :param size: The [x, y] dimensions of the black background
        :param canvas_pos: Where in the canvas the object is placed
        :param colour: The colour of the single pixel
        :param dot_pos: The position on the single pixel in the black background
        """

        assert dot_pos is None if size is None else True, print("To make a random Dot both size and dot_pos must be None")

        if type(size) == Dimension2D:
            self.size = size
        elif type(size) == list or type(size) == np.ndarray:
            self.size = Dimension2D(size[0], size[1])
        elif size is None:
            size = np.random.randint(1, 6, 2)
            self.size = Dimension2D(size[0], size[1])

        if colour is None:
            color = np.random.randint(2, len(const.COLOR_MAP))

        if dot_pos is None:
            all_indices = np.array([i for i in ndi.ndindex((self.size.dx, self.size.dy))])
            dot_pos = tuple(all_indices[np.random.choice(range(len(all_indices)))])
        else:
            dot_pos = tuple(np.transpose(dot_pos))

        dot_pos = dot_pos

        actual_pixels = np.ones((self.size.dx, self.size.dy))
        actual_pixels[dot_pos] = color

        super().__init__(canvas_pos=canvas_pos, actual_pixels=actual_pixels, id=id)


class Primitive(Object):
    def __init__(self, size: Dimension2D | np.ndarray | List, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 colour: None | int = None):
        """
        A basic class for common Primitive data and methods
        :param size: The x, y size of the Primitive
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Primitive
        :param colour: The colour of the Primitive
        """

        if type(size) == Dimension2D:
            self.size = size
        elif type(size) == list or type(size) == np.ndarray:
            self.size = Dimension2D(size[0], size[1])
        elif size is None:
            size = np.random.randint(2, 20, 2)
            self.size = Dimension2D(size[0], size[1])

        if colour is None:
            self.colour = np.random.randint(2, len(const.COLOR_MAP))

        self.background_size = np.array([border_size[0] + self.size.dy + border_size[1],
                                         border_size[2] + self.size.dx + border_size[3]])

        self.border_size = border_size

    def print_border_size(self):
        """
        A pretty print of the border size
        :return:
        """
        print(f'Border sizes: Up = {self.border_size[0]}, Down = {self.border_size[1]}, Right = {self.border_size[2]}, ' \
                'Down = {self.border_size[2]}')

    def generate_actual_pixels(self, array: np.ndarray | int):
        """
        Embeds the array into the objects actual_pixels
        :param array:
        :return:
        """
        self.actual_pixels = np.ones(self.background_size)
        self.actual_pixels[self.border_size[1]: self.size.dy + self.border_size[1],
                           self.border_size[2]: self.size.dx + self.border_size[2]] = array

    def generate_symmetries(self, dirs: str = 'both'):
        """
        Generate symmetries in the centers of the primitive. This doesn't search to see if the symmetries should be there.
        It just creates symmetries whether they actually exist or not.
        :param dirs: 'both', 'x', 'y'. Create only an x only a y or both symmetries
        :return:
        """
        col_pixels_pos = self.get_coloured_pixels_positions()
        xmin = np.min(col_pixels_pos[:, 1])
        xmax = np.max(col_pixels_pos[:, 1])
        ymin = np.min(col_pixels_pos[:, 0])
        ymax = np.max(col_pixels_pos[:, 0])

        if dirs == 'both' or dirs == 'y':
            y_sym_origin = Point((xmax - xmin) / 2 + xmin, ymin)
            y_sym_length = ymax - ymin
            y_sym_or = Orientation.Up
            y_symmetry = Vector(orientation=y_sym_or, length=y_sym_length, origin=y_sym_origin)
            self.symmetries.append(y_symmetry)
        if dirs == 'both' or dirs == 'x':
            x_sym_origin = Point(xmin, (ymax - ymin) / 2 + ymin)
            x_sym_length = xmax - xmin
            x_sym_or = Orientation.Right
            x_symmetry = Vector(orientation=x_sym_or, length=x_sym_length, origin=x_sym_origin)
            self.symmetries.append(x_symmetry)


class Parallelogram(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None, id: None | int = None):
        """
        A solid colour parallelogram inside a black region (border)
        :param size: The size of the parallelogram
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the parallelogram
        :param canvas_pos: The position in the canvas
        :param id: The id of the object
        """

        Primitive.__init__(self, size=size, border_size=border_size, colour=colour)

        self.generate_actual_pixels(self.colour)

        Object.__init__(self, canvas_pos=canvas_pos, actual_pixels=self.actual_pixels, id=id)

        self.generate_symmetries()


class Cross(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None, id: None | int = None):
        """
        A single colour cross surrounded by black border
        :param size: Dimension2D. The x, y size of the cross. Since the cross has to be symmetric the dx, dy should be
        odd.
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the cross
        :param canvas_pos: The position on the canvas
        :param colour: The cross' colour
        :param id: The id of the object
        """

        Primitive.__init__(self, size=size, border_size=border_size, colour=colour)

        assert self.size.dx % 2 == 1 and self.size.dy % 2 == 1, print('To make a Cross the x and y size must be odd numbers')

        cross = []
        for x in range(self.size.dx):
            temp = np.ones(self.size.dy)
            if x != self.size.dx / 2 - 0.5:
                temp[int(self.size.dy / 2)] = self.colour
            else:
                temp *= self.colour
            cross.append(temp)
        cross = np.transpose(cross)
        self.generate_actual_pixels(cross)

        Object.__init__(self, actual_pixels=self.actual_pixels, canvas_pos=canvas_pos, id=id)

        self.generate_symmetries()


class Pi(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None, id: None | int = None):
        """
        A Pi shaped object.
        :param size: The x, y size of the object
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Pi
        :param canvas_pos: The position on the canvas
        :param colour: The Pi's colour
        :param id: The id of the object
        """

        Primitive.__init__(self, size=size, border_size=border_size, colour=colour)

        pi = []
        for y in range(self.size.dy):
            temp = np.ones(self.size.dx)
            if y == self.size.dy - 1:
                temp[:] = self.colour
            else:
                temp[0] = self.colour
                temp[-1] = self.colour
            pi.append(temp)
        pi = np.array(pi)

        self.generate_actual_pixels(pi)

        Object.__init__(self, actual_pixels=self.actual_pixels, canvas_pos=canvas_pos, id=id)

        self.generate_symmetries('y')


class Angle(Primitive):
    def __init__(self, size: np.ndarray | List, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None, id: None | int = None):
        """
        A 90 degrees down left pointing angle
        :param size: The x, y size of the object
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Angle
        :param canvas_pos: The position on the canvas
        :param colour: The Angle's colour
        :param id: The id of the object
        """
        Primitive.__init__(self, size=size, border_size=border_size, colour=colour)

        angle = []
        for y in range(self.size.dy):
            temp = np.ones(self.size.dx)
            if y != 0:
                temp[0] = self.colour
            else:
                temp[:] = self.colour
            angle.append(temp)
        angle = np.array(angle)

        self.generate_actual_pixels(angle)

        Object.__init__(self, actual_pixels=self.actual_pixels, canvas_pos=canvas_pos, id=id)


class Zigzag(Primitive):
    def __init__(self, height: int, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None, id: None | int = None,
                 double=False):
        """
        A Zigzag line (single pixel step) from top left to bottom right.
        :param height: The height of the Zigzag (which will determine its width also)
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the ZigZag
        :param canvas_pos: The position on the canvas
        :param colour: The ZigZag's colour
        :param id: The id of the object
        :param double: If True then the line is composed of two pixels, otherwise just of one
        """

        x = height+1 if double else height
        size = Dimension2D(x, height)
        Primitive.__init__(self, size=size, border_size=border_size, colour=colour)

        zigzag = []
        n = 2 if double else 1
        for y in range(self.size.dy):
            temp = np.ones(self.size.dx)
            temp[y:y+n] = self.colour
            zigzag.append(temp)
        angle = np.flipud(zigzag)

        self.generate_actual_pixels(angle)

        Object.__init__(self, actual_pixels=self.actual_pixels, canvas_pos=canvas_pos, id=id)


class Fish(Primitive):
    def __init__(self, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None, id: None | int = None):
        """
        A Fish like object
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Fish
        :param canvas_pos: The position on the canvas
        :param colour: The Fish's colour
        :param id: The id of the object
        """
        size = Dimension2D(3, 3)
        Primitive.__init__(self, size=size, border_size=border_size, colour=colour)

        fish = np.array([[1, self.colour, 1],
                         [self.colour, self.colour, self.colour],
                         [1, self.colour, self.colour]])

        self.generate_actual_pixels(fish)

        Object.__init__(self, actual_pixels=self.actual_pixels, canvas_pos=canvas_pos, id=id)


class InverseCross(Primitive):
    def __init__(self, height: int, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 fill_colour: None | int = None, fill_height: None | int = None, id: None | int = None):
        """
        A cross made out of a central dot with four arms 45 degrees to the vertical and perpendicular with a second
        colour (fill_colour) filling up to a point (fill_height) the inbetween pixels

        :param height: The height of the cross (it is fully symmetric so width is same as height)
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the cross
        :param canvas_pos: The position on the canvas
        :param colour: The cross' colour
        :param fill_colour: The 2nd colour to fill the black pixels surrounding the center dot
        :param fill_height: The number of pixels to get the 2nd colour away from the center
        :param id: The id of the object
        """
        assert height % 2 == 1, print('To make an Inverted Cross the height must be an odd number')

        if fill_height is not None:
            assert fill_height % 2 == 1, print('To fill an Inverted Cross the fill_height must be an odd number')

        size = Dimension2D(height, height)
        Primitive.__init__(self, size=size, border_size=border_size, colour=colour)

        if fill_colour is None:
            fill_colour = self.colour
            while fill_colour == self.colour:
                fill_colour = np.random.randint(2, len(const.COLOR_MAP))

        fill = int((self.size.dx - fill_height) / 2)
        cross = []
        for x in range(self.size.dx):
            temp = np.ones(self.size.dy)

            if fill_height is not None:
                if fill <= x < self.size.dx - fill:
                    temp[fill:-fill] = fill_colour

            temp[x] = self.colour
            temp[-x-1] = self.colour
            cross.append(temp)
        cross = np.array(cross)

        self.generate_actual_pixels(cross)

        Object.__init__(self, actual_pixels=self.actual_pixels, canvas_pos=canvas_pos, id=id)

        self.generate_symmetries()


class Hole(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, thickness: List | np.ndarray = (1, 1, 1, 1),
                 border_size: np.ndarray | List = np.array([0, 0, 0, 0]), canvas_pos: Point = Point(0, 0),
                 colour: None | int = None, id: None | int = None):
        """
        This is a hole formed by an outside coloured parallelogram and an inside black parallelogram. The object also
        holds the position of the black hole as a self.hole_bbox Bbox.
        :param size: The x, y size of the surround
        :param thickness: The thickness of the coloured surround ([Up, Down, Left, Right[)
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the object
        :param canvas_pos: The position on the canvas
        :param colour: The surround colour
        :param id: The id of the object
        """

        Primitive.__init__(self, size=size, border_size=border_size, colour=colour)

        self.generate_actual_pixels(self.colour)

        th_up = thickness[0]
        th_down = thickness[1]
        th_left = thickness[2]
        th_right = thickness[3]

        self.actual_pixels[self.border_size[1] + th_down: self.size.dy + self.border_size[1] - th_up,
                           self.border_size[2] + th_left: self.size.dx + self.border_size[2] - th_right] = 1

        self.hole_bbox = Bbox(top_left=Point(self.border_size[2] + th_left, self.size.dy + self.border_size[1] - th_up - 1),
                              bottom_right=Point(self.size.dx + self.border_size[2] - th_right - 1, self.border_size[1] + th_down))

        Object.__init__(self, canvas_pos=canvas_pos, actual_pixels=self.actual_pixels, id=id)

        sym = None
        if th_up == th_down:
            sym = 'x'
        if th_left == th_right:
            sym = 'y'
        if th_up == th_down and th_left == th_right:
            sym = 'both'

        if sym is not None:
            self.generate_symmetries(sym)


class Random(Primitive):
    pass