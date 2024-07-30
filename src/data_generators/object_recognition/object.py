
from __future__ import annotations

import copy as cp

import numpy as np
import skimage
from visualization import visualize_data as vis
from data_generators.object_recognition.basic_geometry import *

np.random.seed(const.RANDOM_SEED_FOR_NUMPY)
MAX_PAD_SIZE = const.MAX_PAD_SIZE


class Object:

    def __init__(self, actual_pixels: np.ndarray, _id: None | int = None,
                 border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: List | np.ndarray | Point = (0, 0, 0)):

        self.id = _id
        self.actual_pixels = actual_pixels
        self._canvas_pos = canvas_pos
        self.border_size = border_size

        if type(canvas_pos) != Point:
            self._canvas_pos = Point.point_from_numpy(np.array(canvas_pos))

        self.rotation_axis = cp.deepcopy(self._canvas_pos)

        self.dimensions = Dimension2D(self.actual_pixels.shape[1], self.actual_pixels.shape[0])

        self.number_of_coloured_pixels: int = int(np.sum([1 for i in self.actual_pixels for k in i if k > 1]))

        self.symmetries: List = []

        self.reset_dimensions()

    @property
    def canvas_pos(self):
        return self._canvas_pos

    @canvas_pos.setter
    def canvas_pos(self, new_pos):
        move = new_pos - self._canvas_pos
        self._canvas_pos = new_pos
        for sym in self.symmetries:
            sym.origin += move
        self.reset_dimensions()

    # Transformation methods
    def scale(self, factor: int):
        """
        Scales the object. A positive factor adds pixels and a negative factor removes pixels.
        :param factor: Integer
        :return: Nothing
        """

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

        for sym in self.symmetries:
            if factor > 0:
                edges = Point(0.5, 0.5)
                if sym.origin.x == 0:
                    edges.x = 0
                if sym.origin.y == 0:
                    edges.y = 0
                sym.origin = (sym.origin + edges - self._canvas_pos) * factor + self._canvas_pos
            else:
                factor = 1/np.abs(factor)
                sym.origin = (sym.origin - self._canvas_pos) * factor + self._canvas_pos
            sym.length *= factor

        self.reset_dimensions()

    def rotate(self, times: Union[1, 2, 3], center: np.ndarray | List | Point = (0, 0)):
        """
        Rotate the object counter-clockwise by times multiple of 90 degrees
        :param times: 1, 2 or 3 times
        :param center: The point of the axis of rotation
        :return:
        """
        radians = np.pi/2 * times
        degrees = 90 * times
        self.actual_pixels = skimage.transform.rotate(self.actual_pixels, degrees, resize=True, order=0, center=center)

        if type(center) == Point:
            center = center.to_numpy()
        if len(center) == 2:
            center = np.array([center[0], center[1], 0])

        center += np.array([self.rotation_axis.x, self.rotation_axis.y, 0]).astype(int)
        self.bbox.transform(translation=-center)
        self.bbox.transform(rotation=radians)
        self.bbox.transform(translation=center)
        self._canvas_pos.x = self.bbox.top_left.x
        self._canvas_pos.y = self.bbox.bottom_right.y

        for sym in self.symmetries:
            sym.transform(translation=-center)
            sym.transform(rotation=radians)
            sym.transform(translation=center)

        self.dimensions.dx = self.actual_pixels.shape[1]
        self.dimensions.dy = self.actual_pixels.shape[0]

    def shear(self, _shear: np.ndarray | List):
        """
        Shears the actual pixels
        :param _shear: Shear percentage (0 to 1)
        :return:
        """
        self.flip(Orientation.Left)
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

            #new_symmetry_axis_origin.y -= 0.5
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

            #new_symmetry_axis_origin.x -= 0.5
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

    def randomise_colour(self, ratio: float = 0.1, colour: str = 'random'):
        """
        Changes the colour of ratio of the coloured pixels (picked randomly) to a new random (not already there) colour
        :param ratio: The ratio of the coloured pixels to be recoloured
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

    def randomise_shape(self, add_or_subtract: str = 'add', ratio: float = 0.1, colour: str = 'common'):
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

    def copy(self):
        new_obj = Object(actual_pixels=self.actual_pixels, _id=self.id, border_size=self.border_size,
                         canvas_pos=self.canvas_pos)
        for sym in self.symmetries:
            new_obj.symmetries.append(sym.copy())
        return new_obj

    def __add__(self, other: Object):
        pass

    def __sub__(self, other: object):
        pass

    def superimpose(self, other: Object, z_order: int = 1):
        pass

    #TODO: This can now be achieved with the setter of canvas_pos. Do I still need this function?
    def move_along_z(self, orientation: OrientationZ | None = None, to_z: float | None = None):
        """
        Change the z of the canvas_pos and the bbox
        :param orientation: Change the z by one point in the orientation given
        :param to_z: Take the z to the to_z number
        :return:
        """
        assert orientation is not None or to_z is not None, print('Either Orientation or to_z must ')

        if orientation is not None:
            self._canvas_pos.z += -1 if orientation == OrientationZ.Away else 1
            self.bbox.top_left.z += -1 if orientation == OrientationZ.Away else 1
            self.bbox.bottom_right.z += -1 if orientation == OrientationZ.Away else 1
            for sym in self.symmetries:
                sym.origin.z += -1 if orientation == OrientationZ.Away else 1

        if to_z is not None:
            self._canvas_pos.z = to_z
            self.bbox.top_left.z = to_z
            self.bbox.bottom_right.z = to_z
            for sym in self.symmetries:
                sym.origin.z = to_z

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

    def pick_random_pixels(self, coloured_or_background: str = 'coloured', ratio: float = 0.1) -> None | np.ndarray:
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

        num_of_new_pixels = int((pixels_pos.size // 2) * ratio)
        if num_of_new_pixels < 1:
            num_of_new_pixels = 1

        if len(pixels_pos) > 0:
            t = np.random.choice(np.arange(len(pixels_pos)), num_of_new_pixels)
            return pixels_pos[t]
        else:
            return None

    def show(self, symmetries_on=True):
        """
        Show a matplotlib.pyplot.imshow image of the actual_pixels array correctly colour transformed
        :param symmetries_on: Show the symmetries of the object as line
        :return:
        """
        xmin = self.bbox.top_left.x - 0.5
        xmax = self.bbox.bottom_right.x + 0.5
        ymin = self.bbox.bottom_right.y - 0.5
        ymax = self.bbox.top_left.y + 0.5
        extent = [xmin, xmax, ymin, ymax]
        ax = vis.plot_data(self.actual_pixels, extent=extent)

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
