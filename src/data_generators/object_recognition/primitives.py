
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
    #Line: int = 3  # Same as Parallelogram
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
    def __init__(self, size: None | np.ndarray | Bbox | List = np.array([1, 1]), canvas_pos: Point = Point(0, 0),
                 colour: None | int = None, dot_pos: None | np.ndarray | List = np.array([0, 0])):
        """
        A single coloured pixel in a black background
        :param size: The [x, y] dimensions of the black background
        :param canvas_pos: Where in the canvas the object is placed
        :param colour: The colour of the single pixel
        :param dot_pos: The position on the single pixel in the black background
        """

        assert dot_pos is None if size is None else True, print("To make a random Dot both size and dot_pos must be None")

        if type(size) == Bbox:
            size = np.array([size.bottom_right.x, size.top_left.y])
        elif type(size) == list:
            size = np.array(size)
        elif size is None:
            size = np.random.randint(1, 6, 2)

        size = np.flip(size)

        if colour is None:
            color = np.random.randint(2, len(const.COLOR_MAP))

        if dot_pos is None:
            all_indices = np.array([i for i in ndi.ndindex(tuple(size))])
            dot_pos = tuple(all_indices[np.random.choice(range(len(all_indices)))])
        else:
            dot_pos = tuple(np.transpose(dot_pos))

        dot_pos = dot_pos

        actual_pixels = np.ones(size)
        actual_pixels[dot_pos] = color

        super().__init__(canvas_pos=canvas_pos, actual_pixels=actual_pixels)


class Primitive(Object):
    def __init__(self, size: np.ndarray | Bbox | List, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 colour: None | int = None):

        if type(size) == Bbox:
            size = np.array([size.bottom_right.x, size.top_left.y])
        elif type(size) == list:
            size = np.array(size)
        elif size is None:
            size = np.random.randint(2, 20, 2)

        self.size = np.flip(size)

        if colour is None:
            self.colour = np.random.randint(2, len(const.COLOR_MAP))

        self.background_size = np.array([border_size[0] + size[1] + border_size[1],
                                         border_size[2] + size[0] + border_size[3]])


# TODO: Add symmetries
class Parallelogram(Primitive):
    def __init__(self, size: np.ndarray | Bbox | List, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None):
        """
        A solid colour parallelogram inside a black region (border)
        :param size: The size of the parallelogram
        :param border_size: [Up, Down, Left, Right]
        :param canvas_pos: The position in the canvas
        """

        Primitive.__init__(self, size=size, border_size=border_size, colour=colour)

        actual_pixels = np.ones(self.background_size)
        actual_pixels[border_size[0]: self.size[0] + border_size[0],
                      border_size[2]: self.size[1] + border_size[2]] = self.colour

        Object.__init__(self, canvas_pos=canvas_pos, actual_pixels=actual_pixels)


# TODO: Add symmetries
class Cross(Primitive):
    def __init__(self, size: np.ndarray | List, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None):

        assert size[0] % 2 == 1 and size[1] % 2 == 1, print('To make a Cross the x and y size must be odd numbers')

        Primitive.__init__(self, size=size, border_size=border_size, colour=colour)

        actual_pixels = np.ones(self.background_size)

        cross = []
        for x in range(self.size[0]):
            temp = np.ones(self.size[1])
            if x != self.size[0] / 2 - 0.5:
                temp[int(self.size[1] / 2)] = self.colour
            else:
                temp *= self.colour
            cross.append(temp)
        cross = np.array(cross)

        actual_pixels[border_size[0]: self.size[0] + border_size[0],
                      border_size[2]: self.size[1] + border_size[2]] = cross

        Object.__init__(self, actual_pixels=actual_pixels, canvas_pos=canvas_pos)


# TODO: Add symmetry
class Pi(Primitive):
    def __init__(self, size: np.ndarray | List, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None):

        Primitive.__init__(self, size=size, border_size=border_size, colour=colour)

        actual_pixels = np.ones(self.background_size)

        pi = []
        for x in range(self.size[0]):
            temp = np.ones(self.size[1])
            if x == 0:
                temp[:] = self.colour
            else:
                temp[0] = self.colour
                temp[-1] = self.colour
            pi.append(temp)
        pi = np.array(pi)

        actual_pixels[border_size[0]: self.size[0] + border_size[0],
                      border_size[2]: self.size[1] + border_size[2]] = pi

        Object.__init__(self, actual_pixels=actual_pixels, canvas_pos=canvas_pos)


class Angle(Primitive):
    def __init__(self, size: np.ndarray | List, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None):

        Primitive.__init__(self, size=size, border_size=border_size, colour=colour)

        actual_pixels = np.ones(self.background_size)

        angle = []
        for x in range(self.size[0]):
            temp = np.ones(self.size[1])
            if x != self.size[0] - 1:
                temp[0] = self.colour
            else:
                temp[:] = self.colour
            angle.append(temp)
        angle = np.array(angle)

        actual_pixels[border_size[0]: self.size[0] + border_size[0],
                      border_size[2]: self.size[1] + border_size[2]] = angle

        Object.__init__(self, actual_pixels=actual_pixels, canvas_pos=canvas_pos)


class Zigzag(Primitive):
    def __init__(self, height: int, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 double=False):

        x = height+1 if double else height
        size = np.array([x, height])
        Primitive.__init__(self, size=size, border_size=border_size, colour=colour)

        actual_pixels = np.ones(self.background_size)

        zigzag = []
        n = 2 if double else 1
        for x in range(self.size[0]):
            temp = np.ones(self.size[1])
            temp[x:x+n] = self.colour
            zigzag.append(temp)
        angle = np.array(zigzag)

        actual_pixels[border_size[0]: self.size[0] + border_size[0],
                      border_size[2]: self.size[1] + border_size[2]] = angle

        Object.__init__(self, actual_pixels=actual_pixels, canvas_pos=canvas_pos)


class Fish(Primitive):
    def __init__(self, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None):

        size = np.array([3, 3])
        Primitive.__init__(self, size=size, border_size=border_size, colour=colour)

        actual_pixels = np.ones(self.background_size)

        fish = np.array([[1, self.colour, self.colour],
                         [self.colour, self.colour, self.colour],
                         [1, self.colour, 1]])

        actual_pixels[border_size[0]: self.size[0] + border_size[0],
                      border_size[2]: self.size[1] + border_size[2]] = fish

        Object.__init__(self, actual_pixels=actual_pixels, canvas_pos=canvas_pos)


class InverseCross(Primitive):
    def __init__(self, height: int, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 fill_colour: None | int = None, fill_height: None | int = None):

        assert height % 2 == 1, print('To make an Inverted Cross the height must be an odd number')

        if fill_height is not None:
            assert fill_height % 2 == 1, print('To fill an Inverted Cross the fill_height must be an odd number')

        size = np.array([height, height])
        Primitive.__init__(self, size=size, border_size=border_size, colour=colour)

        if fill_colour is None:
            fill_colour = self.colour
            while fill_colour == self.colour:
                fill_colour = np.random.randint(2, len(const.COLOR_MAP))

        actual_pixels = np.ones(self.background_size)

        fill = int((self.size[0] - fill_height) / 2)
        cross = []
        for x in range(self.size[0]):
            temp = np.ones(self.size[1])

            if fill_height is not None:
                if fill <= x < self.size[0] - fill:
                    temp[fill:-fill] = fill_colour

            temp[x] = self.colour
            temp[-x-1] = self.colour
            cross.append(temp)
        cross = np.array(cross)

        actual_pixels[border_size[0]: self.size[0] + border_size[0],
                      border_size[2]: self.size[1] + border_size[2]] = cross

        Object.__init__(self, actual_pixels=actual_pixels, canvas_pos=canvas_pos)

