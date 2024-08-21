
from __future__ import annotations

from enum import Enum
from typing import List, Union
from copy import copy, deepcopy
from typing import Tuple

import numpy as np
from scipy import ndimage as ndi
import skimage
from visualization import visualize_data as vis
from data_generators.object_recognition.basic_geometry import Point, Vector, Orientation, Surround, OrientationZ, Bbox,\
                                                              Dimension2D

import constants as const

#np.random.seed(const.RANDOM_SEED_FOR_NUMPY)
MAX_PAD_SIZE = const.MAX_PAD_SIZE


class Transformations(Enum):
    scale: int = 0
    rotate: int = 1
    shear: int = 2
    mirror: int = 3
    flip: int = 4
    randomise_colour: int = 5
    randomise_shape: int = 6

    def get_random_parameters(self, random_obj_or_not: str = 'Random'):
        args = {}
        if self.name == 'scale':
            args['factor'] = np.random.choice([-4, -3, -2, 2, 3, 4], p=[0.05, 0.15, 0.3, 0.3, 0.15, 0.05])
        if self.name == 'rotate':
            args['times'] = np.random.randint(1, 4)
        if self.name == 'shear':
            if random_obj_or_not == 'Random':
                args['_shear'] = int(np.random.gamma(shape=1, scale=15) + 10)  # Mainly between 1 and 75
            else:
                args['_shear'] = int(np.random.gamma(shape=1, scale=10) + 5)  # Mainly between 0.05 and 0.4
                args['_shear'] = 40 if args['_shear'] > 40 else args['_shear']
        if self.name == 'mirror' or self.name == 'flip':
            args['axis'] = np.random.choice([Orientation.Up, Orientation.Down, Orientation.Left, Orientation.Right])
        if self.name == 'mirror':
            args['on_axis'] = False if np.random.rand() < 0.5 else True
        if self.name == 'randomise_colour':
            if random_obj_or_not == 'Random':
                args['ratio'] = int(np.random.gamma(shape=1, scale=10) + 1)  # Mainly between 10 and 40
                args['ratio'] = int(40) if args['ratio'] > 40 else args['ratio']
            else:
                args['ratio'] = int(np.random.gamma(shape=1, scale=5) + 2)  # Mainly between 10 and 40
                args['ratio'] = int(15) if args['ratio'] > 15 else args['ratio']
        if self.name == 'randomise_shape':
            args['add_or_subtract'] = 'add' if np.random.random() > 0.5 else 'subtract'
            if random_obj_or_not == 'Random':
                args['ratio'] = int(np.random.gamma(shape=1, scale=7) + 1)  # Mainly between 0.1 and 0.3
                args['ratio'] = 30 if args['ratio'] > 30 else args['ratio']
            else:
                args['ratio'] = int(np.random.gamma(shape=1, scale=5) + 2)  # Mainly between 0.1 and 0.3
                args['ratio'] = 15 if args['ratio'] > 15 else args['ratio']
        return args

    @staticmethod
    def get_specific_parameters(transformation_index, values):
        args = {}
        if transformation_index == 0:
            args['factor'] = values
        elif transformation_index == 1:
            args['times'] = values
        elif transformation_index == 2:
            args['_shear'] = values
        elif transformation_index == 3:
            args['axis'] = values[0]
            args['on_axis'] = values[1]
        elif transformation_index == 5:
            args['ratio'] = values
        elif transformation_index == 6:
            args['ratio'] = values

        return args


class Object:

    def __init__(self, actual_pixels: np.ndarray, _id: None | int = None,
                 actual_pixels_id: int | None = None, border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: List | np.ndarray | Point = (0, 0, 0), canvas_id: int | None = None):

        self.id = _id
        self.actual_pixels_id = actual_pixels_id
        self.canvas_id = canvas_id
        self.actual_pixels = actual_pixels
        self._canvas_pos = canvas_pos
        self.border_size = border_size

        self._holes = None

        if type(canvas_pos) != Point:
            self._canvas_pos = Point.point_from_numpy(np.array(canvas_pos))

        self.rotation_axis = deepcopy(self._canvas_pos)

        self.dimensions = Dimension2D(self.actual_pixels.shape[1], self.actual_pixels.shape[0])

        self.number_of_coloured_pixels: int = int(np.sum([1 for i in self.actual_pixels for k in i if k > 1]))

        self.symmetries: List = []

        self.transformations: List = []

        self.reset_dimensions()

    @property
    def canvas_pos(self):
        return self._canvas_pos

    @canvas_pos.setter
    def canvas_pos(self, new_pos: Point):
        move = new_pos - self._canvas_pos
        self._canvas_pos = new_pos
        for sym in self.symmetries:
            sym.origin += move
        self.reset_dimensions()

    @property
    def holes(self):
        holes, n = self.detect_holes(self.actual_pixels)
        if n == 0:
            return None
        self._holes = holes
        return self._holes

    # Transformation methods
    def scale(self, factor: int):
        """
        Scales the object. A positive factor adds pixels and a negative factor removes pixels.
        :param factor: Integer
        :return: Nothing
        """

        def length_multiplier(factor):
            return factor + factor - 2

        if factor == 0:  # Factor cannot be 0 so in this case nothing happens
            return

        pic = self.actual_pixels

        if factor > 0:
            # If factor is > 0 it cannot blow up the object to more than MAX_PAD_SIZE
            if np.max(pic.shape) * factor > const.MAX_PAD_SIZE - 2:
                return
            scaled = np.ones(np.array(pic.shape) * factor)
            for x in range(pic.shape[0]):
                for y in range(pic.shape[1]):
                    scaled[x * factor:(x + 1) * factor, y * factor:(y + 1) * factor] = pic[x, y]
        else:
            # If factor is <0 it cannot shrink the object to something smaller than 2x2
            if np.abs(1/factor) * np.min(pic.shape) < 2:
                return
            scaled = np.ones(np.ceil(np.array(pic.shape) / np.abs(factor)).astype(np.int32))
            for x in range(scaled.shape[0]):
                for y in range(scaled.shape[1]):
                    scaled[x, y] = pic[x * np.abs(factor), y * np.abs(factor)]

        self.actual_pixels = scaled

        self.reset_dimensions()

        for sym in self.symmetries:
            if factor > 0:
                sym.length *= 2 * factor - 0.5 * (factor + 1)

                d = np.abs(sym.origin - self.canvas_pos)
                multiplier = Point(factor, factor) if sym.orientation == Orientation.Up else Point(factor + 1, factor)
                sym.origin.x = d.x * multiplier.x + self.canvas_pos.x if d.x > 0 else sym.origin.x
                sym.origin.y = d.y * multiplier.y + self.canvas_pos.y if d.y > 0 else sym.origin.y

                sym.origin.x += 0.5 * (factor -1) if sym.orientation == Orientation.Up\
                    else -(factor -1) + (factor - 2) *2
                sym.origin.y += (factor - 1) * 0.5 if sym.orientation == Orientation.Left\
                    else (factor - 1) * 0.5 - (factor - 2) * 0.5 - 0.5
            else:
                sym.length /= np.abs(factor)
                factor = 1/np.abs(factor)
                sym.origin = (sym.origin - self._canvas_pos) * factor + self._canvas_pos

        self.transformations.append([Transformations.scale, {'factor': factor}])

    def rotate(self, times: Union[1, 2, 3], center: np.ndarray | List | Point = (0, 0)):
        """
        Rotate the object counter-clockwise by times multiple of 90 degrees
        :param times: 1, 2 or 3 times
        :param center: The point of the axis of rotation
        :return:
        """
        radians = np.pi/2 * times
        degrees = -(90 * times)
        self.actual_pixels = skimage.transform.rotate(self.actual_pixels, degrees, resize=True, order=0, center=center)

        if type(center) == Point:
            center = center.to_numpy()
        if len(center) == 2:
            center = np.array([center[0], center[1], 0])

        center += np.array([self.rotation_axis.x, self.rotation_axis.y, 0]).astype(int)
        self.bbox.transform(translation=-center)
        self.bbox.transform(rotation=radians)
        self.bbox.transform(translation=center)
        self._canvas_pos.x = int(self.bbox.top_left.x)
        self._canvas_pos.y = int(self.bbox.bottom_right.y)

        for sym in self.symmetries:
            sym.transform(translation=-center)
            sym.transform(rotation=radians)
            sym.transform(translation=center)

        self.dimensions.dx = self.actual_pixels.shape[1]
        self.dimensions.dy = self.actual_pixels.shape[0]

        self.transformations.append([Transformations.rotate, {'times': times}])

    def shear(self, _shear: np.ndarray | List):
        """
        Shears the actual pixels
        :param _shear: Shear percentage (0 to 100)
        :return:
        """
        self.transformations.append([Transformations.shear, {'_shear': _shear}])

        self.flip(Orientation.Left)
        _shear = _shear / 100
        transform = skimage.transform.AffineTransform(shear=_shear)

        temp_pixels = self.actual_pixels[self.border_size.Down: self.dimensions.dy - self.border_size.Up,
                                         self.border_size.Right: self.dimensions.dx - self.border_size.Left]

        large_pixels = np.ones((300, 300))
        large_pixels[30: 30 + temp_pixels.shape[0], 170: 170 + temp_pixels.shape[1]] = temp_pixels
        large_pixels_sheared = skimage.transform.warp(large_pixels, inverse_map=transform.inverse, order=0)
        coloured_pos = np.argwhere(large_pixels_sheared > 1)

        top_left = coloured_pos.min(0)
        bottom_right = coloured_pos.max(0)
        new_pixels = large_pixels_sheared[top_left[0]:bottom_right[0] + 1, top_left[1]: bottom_right[1] + 1]
        self.actual_pixels = np.ones((new_pixels.shape[0] + self.border_size.Up + self.border_size.Down,
                                       new_pixels.shape[1] + self.border_size.Left + self.border_size.Right))
        self.actual_pixels[self.border_size.Down: new_pixels.shape[0] + self.border_size.Down,
                           self.border_size.Right: new_pixels.shape[1] + self.border_size.Right] = new_pixels

        self.flip(Orientation.Right)
        self.reset_dimensions()
        self.symmetries = []  # Loose any symmetries

    def mirror(self, axis: Orientation, on_axis=False):
        """
        Mirrors to object (copy, flip and move) along one of the edges (up, down, left or right). If on_axis is True
        the pixels along the mirror axis do not get copied
        :param axis: The axis of mirroring (e.g. Orientation.Up means along the top edge of the object)
        :param on_axis: If it is True the pixels along the mirror axis do not get copied
        :return:
        """
        if axis == Orientation.Up or axis == Orientation.Down:
            concat_pixels = np.flipud(self.actual_pixels)
            if on_axis:
                concat_pixels = concat_pixels[:-1, :] if axis == Orientation.Up else concat_pixels[1:, :]

            self.actual_pixels = np.concatenate((self.actual_pixels, concat_pixels), axis=0) \
                if axis == Orientation.Down else np.concatenate((concat_pixels, self.actual_pixels), axis=0)

            new_symmetry_axis_origin = Point(self._canvas_pos.x, self.actual_pixels.shape[0] / 2 + self._canvas_pos.y) \
                if axis == Orientation.Up else Point(self._canvas_pos.x, self._canvas_pos.y)

            new_symmetry_axis_origin.y -= 0.5
            if on_axis and axis == Orientation.Down:
                new_symmetry_axis_origin.y -= 0.5

            if on_axis and axis == Orientation.Down:
                for sym in self.symmetries:
                    if sym.orientation == Orientation.Right and sym.origin.y > new_symmetry_axis_origin.y:
                        sym.origin.y -= 1

            symmetry_vector = Vector(orientation=Orientation.Right, origin=new_symmetry_axis_origin,
                                     length=self.actual_pixels.shape[1] - 1)

        elif axis == Orientation.Left or axis == Orientation.Right:
            concat_pixels = np.fliplr(self.actual_pixels)
            if on_axis:
                concat_pixels = concat_pixels[:, 1:] if axis == Orientation.Right else concat_pixels[:, :-1]

            self.actual_pixels = np.concatenate((self.actual_pixels, concat_pixels), axis=1) if axis == Orientation.Right else \
                np.concatenate((concat_pixels, self.actual_pixels), axis=1)

            new_symmetry_axis_origin = Point(self.actual_pixels.shape[1] / 2 + self._canvas_pos.x, self._canvas_pos.y)\
                if axis == Orientation.Right else Point(self._canvas_pos.x, self._canvas_pos.y)

            new_symmetry_axis_origin.x -= 0.5
            if on_axis and axis == Orientation.Left:
                new_symmetry_axis_origin.x -= 0.5

            if on_axis and axis == Orientation.Left:
                for sym in self.symmetries:
                    if sym.orientation == Orientation.Up and sym.origin.x > new_symmetry_axis_origin.x:
                        sym.origin.x -= 1

            symmetry_vector = Vector(orientation=Orientation.Up, origin=new_symmetry_axis_origin,
                                     length=self.actual_pixels.shape[0] - 1)

        if axis == Orientation.Left:
            self._canvas_pos.x -= self.dimensions.dx
        if axis == Orientation.Down:
            self._canvas_pos.y -= self.dimensions.dy

        self.symmetries.append(symmetry_vector)

        self.reset_dimensions()

        self.transformations.append([Transformations.mirror, {'axis': axis}])

    def flip(self, axis: Orientation):
        """
        Flips the object along an axis and possibly copies it
        :param axis: The direction to flip towards. The edge of the bbox toward that direction becomes the axis of the flip.
        :return: Nothing
        """

        if axis == Orientation.Up or axis == Orientation.Down:
            self.actual_pixels = np.flipud(self.actual_pixels)
        elif axis == Orientation.Left or axis == Orientation.Right:
            self.actual_pixels = np.fliplr(self.actual_pixels)

        self.transformations.append([Transformations.flip, {'axis': axis}])

    def randomise_colour(self, ratio: int = 10, colour: str = 'random'):
        """
        Changes the colour of ratio of the coloured pixels (picked randomly) to a new random (not already there) colour
        :param ratio: The percentage of the coloured pixels to be recoloured (0 to 100)
        :param colour: The colour to change the pixels to. 'random' means a random colour (not already on the object),
        'x' means use the colour number x
        :return:
        """
        new_pixels_pos = self.pick_random_pixels(coloured_or_background='coloured', ratio=ratio)

        if new_pixels_pos is not None:
            if colour == 'random':
                colours = self.get_used_colours()
                new_colour = np.setdiff1d(np.arange(2, 11), colours)
            else:
                new_colour = int(colour)

            self.actual_pixels[new_pixels_pos[:, 0], new_pixels_pos[:, 1]] = np.random.choice(new_colour, size=1)

            self.symmetries = []

            self.transformations.append([Transformations.randomise_colour, {'ratio': ratio}])

    def randomise_shape(self, add_or_subtract: str = 'add', ratio: int = 10, colour: str = 'common'):
        """
        Adds or subtracts coloured pixels to the object
        :param add_or_subtract: To add or subtract pixels. 'add' or 'subtract'
        :param ratio: The percentage (ratio) of pixels to be added or subtracted
        :param colour: Whether the colour used for added pixels should be the most common one used or a random one or
        a specific one. 'common' or 'random' or 'x' where x is the colour number (from 2 to 10)
        :return:
        """
        coloured_or_background = 'background' if add_or_subtract == 'add' else 'coloured'
        new_pixels_pos = self.pick_random_pixels(coloured_or_background=coloured_or_background, ratio=ratio)

        if new_pixels_pos is not None:
            if add_or_subtract == 'add':
                if colour == 'common':
                    colours = self.actual_pixels[np.where(self.actual_pixels > 1)].astype(int)
                    new_colour = int(np.argmax(np.bincount(colours)))
                elif colour == 'random':
                    new_colour = np.random.randint(2, 10, 1)
                else:
                    new_colour = int(colour)

            elif add_or_subtract == 'subtract':
                new_colour = 1

            self.actual_pixels[new_pixels_pos[:, 0], new_pixels_pos[:, 1]] = new_colour

            self.symmetries = []

            self.transformations.append([Transformations.randomise_shape, {'ratio': ratio}])

    def create_random_hole(self, hole_size: int) -> bool:
        """
        Tries to create a hole in the Object of total empty pixels = hole_size or smaller. If it succeeds it returns True,
        otherwise it returns False. The hole creation process is totally random and cannot be controlled other than
        setting the maximum number of empty pixels generated.
        :param hole_size: The maximum possible number of connected empty pixels the hole is made out of
        :return: True if the process was successful (there is a hole) and False if not
        """
        pixels = copy(self.actual_pixels)
        initial_holes, initial_n_holes = self.detect_holes()
        connected_components, n = ndi.label(np.array(pixels > 1).astype(int))
        largest_component = 0
        for i in range(1, np.max(connected_components) + 1):
            if len(np.where(connected_components == i)[0]) > largest_component:
                largest_component = i

        if largest_component > 0:
            for _ in range(1000):  # Try 1000 times
                pixels = copy(self.actual_pixels)
                current_hole_size = 0
                hole_points = []
                focus_index = np.random.randint(0, len(np.where(connected_components == largest_component)[0]))
                focus_point = np.array([np.where(connected_components == largest_component)[0][focus_index],
                                        np.where(connected_components == largest_component)[1][focus_index]])
                hole_points.append(focus_point)
                current_hole_size += 1
                while hole_size > current_hole_size:
                    new_points = np.array([focus_point + np.array([-1, 0]), focus_point + np.array([1, 0]),
                                           focus_point + np.array([0, -1]), focus_point + np.array([0, 1])])
                    new_point_found = False
                    np.random.shuffle(new_points)
                    for p in new_points:
                        try:
                            if pixels[p[0], p[1]] == pixels[focus_point[0], focus_point[1]]:
                                hole_points.append(p)
                                current_hole_size += 1
                                new_point_found = True
                                break
                        except IndexError:
                            pass

                    if new_point_found:
                        focus_point = p
                    else:
                        break

                for hp in hole_points:
                    pixels[hp[0], hp[1]] = 1
                final_holes, final_n_holes = self.detect_holes(pixels)

                if final_n_holes > initial_n_holes:
                    self.actual_pixels = pixels
                    self.symmetries = []
                    return True

        return False

    # Utility methods
    def reset_dimensions(self):
        """
        Reset the self.dimensions and the self.bbox top left and bottom right points to fit the updated actual_pixels
        :return:
        """
        self.dimensions.dx = self.actual_pixels.shape[1]
        self.dimensions.dy = self.actual_pixels.shape[0]

        bb_top_left = Point(self._canvas_pos.x, self._canvas_pos.y + self.dimensions.dy - 1, self._canvas_pos.z)
        bb_bottom_right = Point(bb_top_left.x + self.dimensions.dx - 1, self._canvas_pos.y, self._canvas_pos.z)

        self.bbox = Bbox(top_left=bb_top_left, bottom_right=bb_bottom_right)

    def __copy__(self):
        new_obj = Object(actual_pixels=self.actual_pixels, _id=self.id, border_size=self.border_size,
                         canvas_pos=self.canvas_pos)
        new_obj.dimensions = copy(self.dimensions)
        new_obj.border_size = copy(self.border_size)
        new_obj.bbox = copy(self.bbox)
        new_obj.rotation_axis = copy(self.rotation_axis)
        for sym in self.symmetries:
            new_obj.symmetries.append(copy(sym))
        return new_obj

    def __add__(self, other: Object):
        pass

    def __sub__(self, other: object):
        pass

    def superimpose(self, other: Object, z_order: int = 1):
        pass

    def get_coloured_pixels_positions(self) -> np.ndarray:
        result = np.argwhere(self.actual_pixels > 1).astype(int)
        canv_pos = np.array([self._canvas_pos.to_numpy()[1], self._canvas_pos.to_numpy()[0]]).astype(int)
        return canv_pos + result

    def get_background_pixels_positions(self) -> np.ndarray:
        return np.argwhere(self.actual_pixels == 1)

    def get_used_colours(self) -> np.ndarray:
        coloured_pos = self.get_coloured_pixels_positions().astype(int)
        canv_pos = np.array([self._canvas_pos.to_numpy()[1], self._canvas_pos.to_numpy()[0]]).astype(int)
        coloured_pos -= canv_pos
        return np.unique(self.actual_pixels[coloured_pos[:, 0], coloured_pos[:, 1]])

    def set_colour_to_most_common(self):
        colours = self.actual_pixels[np.where(self.actual_pixels > 1)]
        self.colour = int(np.median(colours))

    def negative_colour(self):
        temp = copy(self.actual_pixels)
        self.actual_pixels = np.ones(self.actual_pixels.shape)
        self.actual_pixels[np.where(temp == 1)] = self.colour

    def pick_random_pixels(self, coloured_or_background: str = 'coloured', ratio: int = 10) -> None | np.ndarray:
        """
        Returns the positions (in the self.actual_pixels array) of a random number (ratio percentage) of either
        coloured or background pixels
        :param coloured_or_background: Whether the pixels should come from the coloured group or the background group.
        'coloured' or 'background'
        :param ratio: The ratio (percentage) of the picked pixels over the number of the pixels in their group
        :return:
        """
        if coloured_or_background == 'coloured':
            pixels_pos = self.get_coloured_pixels_positions()
            canv_pos = np.array([self._canvas_pos.to_numpy()[1], self._canvas_pos.to_numpy()[0]]).astype(int)
            pixels_pos -= canv_pos
        elif coloured_or_background == 'background':
            pixels_pos = self.get_background_pixels_positions()

        num_of_new_pixels = int((pixels_pos.size // 2) * ratio / 100)
        if num_of_new_pixels < 1:
            num_of_new_pixels = 1

        if len(pixels_pos) > 0:
            t = np.random.choice(np.arange(len(pixels_pos)), num_of_new_pixels)
            return pixels_pos[t]
        else:
            return None

    def detect_holes(self, pixels: np.ndarray | None = None) -> Tuple[np.ndarray, int]:
        """
        Detects the presence of any holes. A hole is a spatially contiguous set of empty pixels that are fully
        surrounded by coloured pixels.
        :param pixels: The 2d array of pixels in which to detect the presence of holes.
        :return: The array marked with the holes' labels and the number of holes found
        """
        if pixels is None:
            pixels = copy(self.actual_pixels)

        connected_components, n = ndi.label(np.array(pixels == 1).astype(int))

        for i in np.unique(connected_components):
            comp_coords = np.where(connected_components == i)
            if np.any(comp_coords[0] == 0) or \
                    np.any(comp_coords[1] == 0) or \
                    np.any(comp_coords[0] == pixels.shape[0] - 1) or \
                    np.any(comp_coords[1] == pixels.shape[1] - 1):
                connected_components[comp_coords[0], comp_coords[1]] = 0

        holes, n = ndi.label(connected_components)

        return holes, n

    def match(self, other: Object, after_rotation: bool = False, match_shape_only: bool = False,
              padding: Surround = Surround(0, 0, 0, 0)) -> List[Point] | List[List[Point, int]]:
        """
        Calculates the canvas positions that this Object should be moved to, to generate the best match
        (cross - correlation) with the other Object. Multiple best matches generate multiple positions.
        :param padding: Whether the Object should be padded with 1s to allow better matching.
        :param other: The other Object.
        :param after_rotation: Also rotates this Object to check for matches # TODO Not implemented yet!
        :param match_shape_only: Cares only about the shape of the Objects and ignores the colours
        :return: The List of Points that this Object could have as canvas_pos that would generate the largest overlap
        with the other Object. If after_rotation is True then each element of this list is a List[Point, int]
        where int is the rotation required for that match.
        """

        def match_for_a_specific_rotation(rot: int):

            self_dimensions = Dimension2D(self.dimensions.dx + padding.Left + padding.Right,
                                          self.dimensions.dy + padding.Up + padding.Down)

            other_dimensions = Dimension2D(other.dimensions.dx + padding.Left + padding.Right,
                                           other.dimensions.dy + padding.Up + padding.Down)

            x_base_dim = self_dimensions.dx + 2 * other_dimensions.dx - 2
            y_base_dim = self_dimensions.dy + 2 * other_dimensions.dy - 2
            base = np.zeros((y_base_dim, x_base_dim))

            x_res_dim = self_dimensions.dx + other_dimensions.dx - 1
            y_res_dim = self_dimensions.dy + other_dimensions.dy - 1
            result = np.zeros((y_res_dim, x_res_dim))

            base_rotated_object = copy(self)
            base_rotated_object.rotate(rot)

            base_rotated_pixels = np.ones((self_dimensions.dx, self_dimensions.dy))
            base_rotated_pixels[padding.Down: self_dimensions.dy - padding.Up,
                                padding.Left: self_dimensions.dx - padding.Right] = base_rotated_object.actual_pixels
            base[other_dimensions.dy - 1: other_dimensions.dy - 1 + self_dimensions.dy,
                 other_dimensions.dx - 1: other_dimensions.dx - 1 + self_dimensions.dx] = base_rotated_pixels

            for x in range(x_res_dim):
                for y in range(y_res_dim):
                    temp = copy(base[y: other_dimensions.dy + y, x: other_dimensions.dx + x])
                    other_pixels = np.ones((other_dimensions.dy, other_dimensions.dx))
                    other_pixels[padding.Down: other_dimensions.dy - padding.Up,
                                 padding.Left: other_dimensions.dx - padding.Right] = copy(other.actual_pixels)
                    if match_shape_only:
                        temp[np.where(temp == 1)] = 0
                        temp[np.where(temp > 1)] = 1
                        other_pixels[np.where(other_pixels == 1)] = 0
                        other_pixels[np.where(other_pixels > 1)] = 1
                    comp = (temp == other_pixels).astype(int)
                    result[y, x] = comp.sum()
            return result

        if not after_rotation:
            result = match_for_a_specific_rotation(0)
            best_relative_positions = np.argwhere(result == np.amax(result))
            best_positions = [Point(x=other.dimensions.dx - brp[1] - 1 + other.canvas_pos.x + 2 * padding.Left,
                                     y=other.dimensions.dy - brp[0] - 1 + other.canvas_pos.y + 2 * padding.Down)
                              for brp in best_relative_positions]
        else:
            result = []
            for rot in range(4):
                result.append(match_for_a_specific_rotation(rot))
            best_relative_positions = np.argwhere(result == np.amax(result))
            best_positions = [[Point(x=other.dimensions.dx - brp[2] - 1 + other.canvas_pos.x + 2 * padding.Left,
                                     y=other.dimensions.dy - brp[1] - 1 + other.canvas_pos.y + 2 * padding.Down),
                               brp[0]]
                              for brp in best_relative_positions]

        return best_positions

    def show(self, symmetries_on=True, show_holes=False):
        """
        Show a matplotlib.pyplot.imshow image of the actual_pixels array correctly colour transformed
        :param symmetries_on: Show the symmetries of the object as line
        :param show_holes: If True it marks any holes with white pixels
        :return: Nothing
        """
        xmin = self.bbox.top_left.x - 0.5
        xmax = self.bbox.bottom_right.x + 0.5
        ymin = self.bbox.bottom_right.y - 0.5
        ymax = self.bbox.top_left.y + 0.5
        extent = [xmin, xmax, ymin, ymax]

        pixels_to_show = copy(self.actual_pixels)
        if show_holes:
            pixels_to_show[np.where(self.holes > 0)] = 11
        ax = vis.plot_data(pixels_to_show, extent=extent)

        #TODO: DEAL WITH DIAGONAL SYMMETRIES!!!!
        if symmetries_on:
            for sym in self.symmetries:
                if sym.orientation == Orientation.Up or sym.orientation == Orientation.Down:
                    line_at = sym.origin.x
                    line_min = sym.origin.y - 0.5 if sym.orientation == Orientation.Up else sym.origin.y + 0.5
                    line_max = sym.origin.y + sym.length + 0.5 if sym.orientation == Orientation.Up else \
                        sym.origin.y - sym.length - 0.5
                    plt_lines = ax.vlines
                else:
                    line_at = sym.origin.y
                    line_min = sym.origin.x - 0.5 if sym.orientation == Orientation.Right else sym.origin.x + 0.5
                    line_max = sym.origin.x + sym.length + 0.5 if sym.orientation == Orientation.Right else \
                        sym.origin.x - sym.length - 0.5
                    plt_lines = ax.hlines

                plt_lines(line_at, line_min, line_max, linewidth=2)

