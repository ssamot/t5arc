
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


class ObjectType(Enum):
    Random: int = 0
    Parallelogram: int = 1
    Cross: int = 2
    Hole: int = 3
    Pi: int = 4
    InverseCross: int = 5
    Dot: int = 6
    Angle: int = 7
    Diagonal: int = 8
    Steps: int = 9
    Fish: int = 10
    Bolt: int = 11
    Spiral: int = 12
    Tie: int = 13
    Pyramid: int = 14

    @staticmethod
    def random():
        probabilities = np.array([0.2, 0.1, 0.1, 0.1, 0.1, 0.08, 0.08, 0.04, 0.04, 0.04, 0.04, 0.02, 0.02, 0.02, 0.02])
        return ObjectType(np.random.choice(np.arange(len(probabilities)), p=probabilities))


class Primitive(Object):
    def __init__(self, size: Dimension2D | np.ndarray | List, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 required_dist_to_others: np.ndarray | List = np.array([0, 0, 0, 0]), colour: None | int = None):
        """
        A basic class for common Primitive data and methods
        :param size: The x, y size of the Primitive
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Primitive
        :param required_dist_to_others: The [Up, Down, Left, Right] number of pixels that need to be empty around the
        object when its canvas_pos is calculated
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
        else:
            self.colour = colour

        self.background_size = np.array([border_size[0] + self.size.dy + border_size[1],
                                         border_size[2] + self.size.dx + border_size[3]])

        self.border_size = border_size
        self.required_dist_to_others = required_dist_to_others

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


class Random(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None, occupancy_prob: float = 0.5,
                 required_dist_to_others: np.ndarray | List = np.array([0, 0, 0, 0]),
                 _id: None | int = None):

        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        array = np.ones((self.size.dy, self.size.dx))
        for x in range(self.size.dx):
            for y in range(self.size.dy):
                if np.random.random() < occupancy_prob:
                    array[y, x] = self.colour

        self.generate_actual_pixels(array=array)

        Object.__init__(self, actual_pixels=self.actual_pixels, canvas_pos=canvas_pos, _id=_id)


class Parallelogram(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: np.ndarray | List = np.array([0, 0, 0, 0]), _id: None | int = None):
        """
        A solid colour parallelogram inside a black region (border)
        :param size: The size of the parallelogram
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the parallelogram
        :param canvas_pos: The position in the canvas
        :param _id: The id of the object
        """

        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        self.generate_actual_pixels(array=self.colour)

        Object.__init__(self, canvas_pos=canvas_pos, actual_pixels=self.actual_pixels, _id=_id)

        self.generate_symmetries()


class Cross(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: np.ndarray | List = np.array([0, 0, 0, 0]), _id: None | int = None):
        """
        A single colour cross surrounded by black border
        :param size: Dimension2D. The x, y size of the cross. Since the cross has to be symmetric the dx, dy should be
        odd.
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the cross
        :param canvas_pos: The position on the canvas
        :param colour: The cross' colour
        :param _id: The id of the object
        """

        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

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
        self.generate_actual_pixels(array=cross)

        Object.__init__(self, actual_pixels=self.actual_pixels, canvas_pos=canvas_pos, _id=_id)

        self.generate_symmetries()


class Hole(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, thickness: List | np.ndarray = (1, 1, 1, 1),
                 border_size: np.ndarray | List = np.array([0, 0, 0, 0]), canvas_pos: Point = Point(0, 0),
                 colour: None | int = None, required_dist_to_others: np.ndarray | List = np.array([0, 0, 0, 0]),
                 _id: None | int = None):
        """
        This is a hole formed by an outside coloured parallelogram and an inside black parallelogram. The object also
        holds the position of the black hole as a self.hole_bbox Bbox.
        :param size: The x, y size of the surround
        :param thickness: The thickness of the coloured surround ([Up, Down, Left, Right[)
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the object
        :param canvas_pos: The position on the canvas
        :param colour: The surround colour
        :param _id: The id of the object
        """

        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        self.generate_actual_pixels(array=self.colour)

        th_up = thickness[0]
        th_down = thickness[1]
        th_left = thickness[2]
        th_right = thickness[3]

        self.actual_pixels[self.border_size[1] + th_down: self.size.dy + self.border_size[1] - th_up,
                           self.border_size[2] + th_left: self.size.dx + self.border_size[2] - th_right] = 1

        self.hole_bbox = Bbox(top_left=Point(self.border_size[2] + th_left, self.size.dy + self.border_size[1] - th_up - 1),
                              bottom_right=Point(self.size.dx + self.border_size[2] - th_right - 1, self.border_size[1] + th_down))

        Object.__init__(self, canvas_pos=canvas_pos, actual_pixels=self.actual_pixels, _id=_id)

        sym = None
        if th_up == th_down:
            sym = 'x'
        if th_left == th_right:
            sym = 'y'
        if th_up == th_down and th_left == th_right:
            sym = 'both'

        if sym is not None:
            self.generate_symmetries(sym)


class Pi(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: np.ndarray | List = np.array([0, 0, 0, 0]), _id: None | int = None):
        """
        A Pi shaped object.
        :param size: The x, y size of the object
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Pi
        :param canvas_pos: The position on the canvas
        :param colour: The Pi's colour
        :param _id: The id of the object
        """

        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

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

        self.generate_actual_pixels(array=pi)

        Object.__init__(self, actual_pixels=self.actual_pixels, canvas_pos=canvas_pos, _id=_id)

        self.generate_symmetries('y')


class InverseCross(Primitive):
    def __init__(self, height: int, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: np.ndarray | List = np.array([0, 0, 0, 0]),
                 fill_colour: None | int = None, fill_height: None | int = None, _id: None | int = None):
        """
        A cross made out of a central dot with four arms 45 degrees to the vertical and perpendicular with a second
        colour (fill_colour) filling up to a point (fill_height) the inbetween pixels

        :param height: The height of the cross (it is fully symmetric so width is same as height)
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the cross
        :param canvas_pos: The position on the canvas
        :param colour: The cross' colour
        :param fill_colour: The 2nd colour to fill the black pixels surrounding the center dot
        :param fill_height: The number of pixels to get the 2nd colour away from the center
        :param _id: The id of the object
        """
        assert height % 2 == 1, print('To make an Inverted Cross the height must be an odd number')

        if fill_height is not None:
            assert fill_height % 2 == 1, print('To fill an Inverted Cross the fill_height must be an odd number')

        size = Dimension2D(height, height)
        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

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

        self.generate_actual_pixels(array=cross)

        Object.__init__(self, actual_pixels=self.actual_pixels, canvas_pos=canvas_pos, _id=_id)

        self.generate_symmetries()


class Dot(Primitive):
    def __init__(self, border_size: np.ndarray | List = np.array([1, 1, 1, 1]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: np.ndarray | List = np.array([0, 0, 0, 0]), _id: None | int = None):
        """
        A solid colour Dot inside a black region (border)
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Dot
        :param canvas_pos: The position in the canvas
        :param _id: The id of the object
        """

        Primitive.__init__(self, size=Dimension2D(1, 1), border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        self.generate_actual_pixels(array=self.colour)

        Object.__init__(self, canvas_pos=canvas_pos, actual_pixels=self.actual_pixels, _id=_id)


class Angle(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: np.ndarray | List = np.array([0, 0, 0, 0]), _id: None | int = None):
        """
        A 90 degrees down left pointing angle
        :param size: The x, y size of the object
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Angle
        :param canvas_pos: The position on the canvas
        :param colour: The Angle's colour
        :param _id: The id of the object
        """
        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        angle = []
        for y in range(self.size.dy):
            temp = np.ones(self.size.dx)
            if y != 0:
                temp[0] = self.colour
            else:
                temp[:] = self.colour
            angle.append(temp)
        angle = np.array(angle)

        self.generate_actual_pixels(array=angle)

        Object.__init__(self, actual_pixels=self.actual_pixels, canvas_pos=canvas_pos, _id=_id)


class Diagonal(Primitive):
    def __init__(self, length: int, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: np.ndarray | List = np.array([0, 0, 0, 0]), _id: None | int = None):
        """
        A Diagonal line
        :param length: The number of pixels
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Diagonal Line
        :param canvas_pos: The position on the canvas
        :param colour: The Diagonal Line's colour
        :param _id: The id of the object
        """

        size = Dimension2D(length, length)

        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        diagonal = np.ones((length, length))
        for y in range(self.size.dy):
            for x in range(self.size.dx):
                if x == y:
                    diagonal[y, x] = self.colour

        diagonal = np.array(diagonal)

        self.generate_actual_pixels(array=diagonal)

        Object.__init__(self, actual_pixels=self.actual_pixels, canvas_pos=canvas_pos, _id=_id)

        # TODO: Add diagonal symmetries
        '''
        sym_origin = Point(canvas_pos.x + border_size[2] + length, canvas_pos.y + border_size[1])
        self.symmetries.append(Vector(orientation=Orientation.Up_Left, length=length, origin=sym_origin))
        '''


class Steps(Primitive):
    def __init__(self, height: int, depth: int, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: np.ndarray | List = np.array([0, 0, 0, 0]), _id: None | int = None):
        """
        A Steps object from top left to bottom right.
        :param height: The height of the Steps (which will determine its width also)
        :param depth: The number of pixels filling the Steps
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Steps
        :param canvas_pos: The position on the canvas
        :param colour: The Steps' colour
        :param _id: The id of the object
        """

        size = Dimension2D(height, height)
        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        zigzag = []
        for y in range(self.size.dy):
            temp = np.ones(self.size.dx)
            s = y-depth + 1 if y-depth + 1 > 0 else 0
            temp[s:y + 1] = self.colour
            zigzag.append(temp)
        zigzag = np.flipud(zigzag)

        self.generate_actual_pixels(array=zigzag)

        Object.__init__(self, actual_pixels=self.actual_pixels, canvas_pos=canvas_pos, _id=_id)


class Fish(Primitive):
    def __init__(self, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: np.ndarray | List = np.array([0, 0, 0, 0]), _id: None | int = None):
        """
        A Fish like object. 50% of the times the center pixel will be black.
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Fish
        :param canvas_pos: The position on the canvas
        :param colour: The Fish's colour
        :param _id: The id of the object
        """
        size = Dimension2D(3, 3)
        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        center_colour = self.colour if np.random.random() > 0.5 else 1
        fish = np.array([[1, self.colour, 1],
                         [self.colour, center_colour, self.colour],
                         [1, self.colour, self.colour]])

        self.generate_actual_pixels(array=fish)

        Object.__init__(self, actual_pixels=self.actual_pixels, canvas_pos=canvas_pos, _id=_id)


class Bolt(Primitive):
    def __init__(self, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: np.ndarray | List = np.array([0, 0, 0, 0]), _id: None | int = None):
        """
                A Bolt like object (a 3x3 cross with its top left and bottom right filled.
                50% of the times the center pixel will be black.
                :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Bolt
                :param canvas_pos: The position on the canvas
                :param colour: The Bolt's colour
                :param _id: The id of the object
                """
        size = Dimension2D(3, 3)
        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        center_colour = self.colour if np.random.random() > 0.5 else 1
        bolt = np.array([[1, self.colour, self.colour],
                         [self.colour, center_colour, self.colour],
                         [self.colour, self.colour, 1]])

        self.generate_actual_pixels(array=bolt)

        Object.__init__(self, actual_pixels=self.actual_pixels, canvas_pos=canvas_pos, _id=_id)


class Tie(Primitive):
    def __init__(self, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: np.ndarray | List = np.array([0, 0, 0, 0]), _id: None | int = None):
        """
        A Tie like object (a 4 pixels square connected to a single pixel on its bottom left side)
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Tie
        :param canvas_pos: The position on the canvas
        :param colour: The Tie's colour
        :param _id: The id of the object
        """
        size = Dimension2D(3, 3)
        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        tie = np.array([[1, self.colour, self.colour],
                         [1, self.colour, self.colour],
                         [self.colour, 1, 1]])

        self.generate_actual_pixels(array=tie)

        Object.__init__(self, actual_pixels=self.actual_pixels, canvas_pos=canvas_pos, _id=_id)


class Spiral(Primitive):
    def __init__(self, size: Dimension2D | np.ndarray | List, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None, gap: int = 1,
                 required_dist_to_others: np.ndarray | List = np.array([0, 0, 0, 0]), _id: None | int = None):
        """
        A clockwise twisting Spiral.
        :param size: The x, y size of the object
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Spiral
        :param canvas_pos: The position on the canvas
        :param colour: The Spiral's colour
        :param gap: The number of pixels between parallel arms of the Spiral
        :param _id: The id of the object
        """

        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        top = 0
        bottom = size.dy - 1
        left = 0
        right = size.dx - 1
        spiral = np.ones((size.dy, size.dx))

        turn = 0

        while top <= bottom and left <= right:
            for i in range(left, right + 1):
                spiral[top, i] = self.colour
            top += 1
            if turn > 0:
                left += gap
            for i in range(top, bottom + 1):
                spiral[i, right] = self.colour
            right -= 1

            if top <= bottom:
                for i in range(right, left - 1, -1):
                    spiral[bottom, i] = self.colour
                bottom -= 1
                top += gap

            if left <= right:
                for i in range(bottom, top - 1, -1):
                    spiral[i, left] = self.colour
                left += 1
                right -= gap

            bottom -= gap
            turn += 1

        self.generate_actual_pixels(array=spiral)

        Object.__init__(self, actual_pixels=self.actual_pixels, canvas_pos=canvas_pos, _id=_id)


class Pyramid(Primitive):
    def __init__(self, height: int = 3, border_size: np.ndarray | List = np.array([0, 0, 0, 0]),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None, full: bool = True,
                 required_dist_to_others: np.ndarray | List = np.array([0, 0, 0, 0]), _id: None | int = None):
        """
        A Pyramid shaped object.
        :param height: height of the Pyramid
        :param border_size: The [Up, Down, Left, Right] black pixels surrounding the Pyramid
        :param canvas_pos: The position on the canvas
        :param colour: The Pyramid's colour
        :param full: If Ture the Pyramid will be full, otherwise it will be just two arms of a triangle
        :param _id: The id of the object
        """

        size = Dimension2D(height * 2 - 1, height)

        Primitive.__init__(self, size=size, border_size=border_size,
                           required_dist_to_others=required_dist_to_others, colour=colour)

        pyramid = []
        for y in range(self.size.dy):
            temp = np.ones(self.size.dx)
            s = int(self.size.dx/2) - y
            e = y - int(self.size.dx/2)
            if e == 0:
                e = self.size.dx
            if full:
                temp[s: e] = self.colour
            else:
                temp[s:s+1] = self.colour
                temp[e-1:e] = self.colour
            pyramid.append(temp)
        pi = np.flipud(np.array(pyramid))

        self.generate_actual_pixels(array=pi)

        Object.__init__(self, actual_pixels=self.actual_pixels, canvas_pos=canvas_pos, _id=_id)

        self.generate_symmetries('y')


