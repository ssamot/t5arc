
from __future__ import annotations

import numpy as np
import constants as const
from typing import Union, List
from dataclasses import dataclass
from enum import Enum
from dataclasses import dataclass

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


class OrientationZ(Enum):
    Away: int = -1
    Towards: int = 1


@dataclass
class Surround:
    Up: int = 0
    Down: int = 0
    Left: int = 0
    Right: int = 0


class Dimension2D:
    def __init__(self, dx: int = 3, dy:  int = 3, array: None | np.ndarray | List = None):
        if array is None:
            self.dx = dx
            self.dy = dy
        else:
            self.dx: int = array[0]
            self.dy: int = array[1]

    def __repr__(self) -> str:
        return f'Dimension:(dX = {self.dx}, dY = {self.dy})'

    def __add__(self, other) -> Dimension2D:
        if type(other) == Dimension2D:
            result = Dimension2D(self.dx + other.dx, self.dy + other.dy)
        if type(other) == Point:
            result = Dimension2D(self.dx + other.x, self.dy + other.y)
        if type(other) == list:
            result = Dimension2D(self.dx + other[0], self.dy + other[1])
        if type(other) == np.ndarray:
            result = Dimension2D(self.dx + other[0], self.dy + other[1])
        if type(other) == int or type(other) == float:
            result = Dimension2D(self.dx + other, self.dy + other)
        return result

    def __sub__(self, other) -> Dimension2D:
        if type(other) == Dimension2D:
            result = Dimension2D(self.dx - other.dx, self.dy - other.dy)
        if type(other) == Point:
            result = Dimension2D(self.dx - other.x, self.dy - other.y)
        if type(other) == list:
            result = Dimension2D(self.dx - other[0], self.dy - other[1])
        if type(other) == np.ndarray:
            result = Dimension2D(self.dx - other[0], self.dy - other[1])
        if type(other) == int or type(other) == float:
            result = Dimension2D(self.dx - other, self.dy - other)
        return result

    def __isub__(self, other) -> Dimension2D:
        return self.__sub__(other)

    def __iadd__(self, other) -> Dimension2D:
        return self.__add__(other)

    def __mul__(self, other: float | int | bool) -> Dimension2D:
        return Dimension2D(self.x * other, self.y * other)

    def __truediv__(self, other) -> Dimension2D:
        return Dimension2D(self.x / other, self.y / other)

    def to_numpy(self):
        return np.array([self.dx, self.dy])


class Point:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0, array: None | List | np.ndarray = None):
        if array is None:
            self.x = x
            self.y = y
            self.z = z  # This is used to define over or under
        else:
            self.x = array[0]
            self.y = array[1]
            self.z = array[2]

    def __add__(self, other) -> Point:
        if type(other) == Point:
            result = Point(self.x + other.x, self.y + other.y, self.z + other.z)
        if type(other) == list:
            result = Point(self.x + other[0], self.y + other[1], self.z + other[2])
        if type(other) == np.ndarray:
            result = Point(self.x + other[0], self.y + other[1], self.z + other[2])
        if type(other) == int or type(other) == float:
            result = Point(self.x + other, self.y + other, self.z + other)
        return result

    def __sub__(self, other) -> Point:
        if type(other) == Point:
            result = Point(self.x - other.x, self.y - other.y, self.z - other.z)
        if type(other) == list:
            result = Point(self.x - other[0], self.y - other[1], self.z - other[2])
        if type(other) == np.ndarray:
            result = Point(self.x - other[0], self.y - other[1], self.z - other[2])
        if type(other) == int or type(other) == float:
            result = Point(self.x - other, self.y - other, self.z - other)
        return result

    def __mul__(self, other: float | int | bool) -> Point:
        return Point(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other) -> Point:
        return Point(self.x / other, self.y / other, self.z / other)

    def __repr__(self) -> str:
        return f'Point(X = {self.x}, Y = {self.y}, Z = {self.z})'

    def __eq__(self, other: Point) -> bool:
        return np.all([self.x == other.x, self.y == other.y, self.z == other.z])

    def __isub__(self, other) -> Point:
        return self.__sub__(other)

    def __iadd__(self, other) -> Point:
        return self.__add__(other)

    def __neg__(self):
        return Point(-self.x, -self.y, -self.z)

    def __len__(self):
        return 3

    def __deepcopy__(self, memo) -> Point:
        return Point(self.x, self.y, self.z)

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @staticmethod
    def point_from_numpy(array: np.ndarray):
        z = 0
        if len(array) == 3:
            z = array[2]
        return Point(array[0], array[1], z)

    def transform(self, affine_matrix: np.ndarray | None = None,
                  rotation: float = 0,
                  shear: List | np.ndarray | Point | None = None,
                  translation: List | np.ndarray | Point | None = None,
                  scale: List | np.ndarray | Point | None = None):

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
                if type(translation) == Point:
                    translation = translation.to_numpy()
                translation_x = translation[0] if type(translation) != Point else translation.x
                translation_y = translation[1] if type(translation) != Point else translation.y
            else:
                translation_x = 0
                translation_y = 0

            if scale is not None:
                if type(scale) == Point:
                    scale = scale.to_numpy()
                scale_x = scale[0] if type(scale) != Point else scale.x
                scale_y = scale[1] if type(scale) != Point else scale.y
            else:
                scale_x = 1
                scale_y = 1

            if shear is not None:
                if type(shear) == Point:
                    shear = shear.to_numpy()
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
                 length: None | int = 0,
                 origin: Point = Point(0, 0, 0)):
        self.orientation = orientation
        self.length = length
        self.origin = origin

    def __repr__(self):
        return f'Vector(Orientation: {self.orientation}, Length: {self.length}, Origin Point: {self.origin})'

    def __deepcopy__(self, memo):
        return self

    # TODO: Need to deal with transformations other than rotation
    def transform(self, affine_matrix: np.ndarray | None = None,
                  rotation: float = 0,
                  shear: List | np.ndarray | Point | None = None,
                  translation: List | np.ndarray | Point | None = None,
                  scale: List | np.ndarray | Point | None = None):
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

    def __getattr__(self, center: str) -> Point:
        self.center = self._calculate_center()
        return self.center

    def __deepcopy__(self, memo):
        return self

    def __repr__(self):
        return f'Bbox(Top Left: {self.top_left}, Bottom Right: {self.bottom_right}, Center: {self.center})'

    def _calculate_center(self):
        center = Point(x=(self.bottom_right.x - self.top_left.x) / 2 + self.top_left.x,
                       y=(self.bottom_right.y - self.top_left.y)/2 + self.top_left.y)
        return center

    def transform(self, affine_matrix: np.ndarray | None = None,
                  rotation: float = 0,
                  shear: List | np.ndarray | Point | None = None,
                  translation: List | np.ndarray | Point | None = None,
                  scale: List | np.ndarray | Point | None = None):
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
            self.top_left = Point(self.bottom_right.x - 2*(self.bottom_right.x - self.top_left.x), self.top_left.x)
            self.bottom_right = Point(p.x, p.y - 2 * (p.y - self.bottom_right.y))

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

        self.center = self._calculate_center()



