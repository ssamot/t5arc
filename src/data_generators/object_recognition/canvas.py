

from __future__ import annotations

import matplotlib.pyplot as plt
import json

from utils import *
from data.utils import *
from visualization import visualize_data as vis
from data_generators.object_recognition.primitives import *
from data_generators.object_recognition.basic_geometry import Point, Dimension2D

np.random.seed(const.RANDOM_SEED_FOR_NUMPY)
MAX_PAD_SIZE = const.MAX_PAD_SIZE


class Canvas:
    def __init__(self, size: Dimension2D | np.ndarray | List, objects: List[Object] | None = None,
                 _id: int | None = None):

        if type(size) != Dimension2D:
            self.size = Dimension2D(array=size)
        else:
            self.size = size

        if objects is None:
            self.objects = []
        else:
            self.objects = objects
        self.id = _id

        self.actual_pixels = np.ones((size.dy, size.dx))
        self.full_canvas = np.zeros((MAX_PAD_SIZE, MAX_PAD_SIZE))
        self.full_canvas[0: self.size.dy, 0:self.size.dx] = self.actual_pixels
        self.background_pixels = np.ndarray.copy(self.actual_pixels)

        self.embed_objects()

    def __repr__(self):
        return f'Canvas {self.id} with {len(self.objects)} Primitives'

    def get_coloured_pixels_positions(self) -> np.ndarray:
        """
        Returns the Union of the positions of the coloured pixels of all the objects in the self.object list
        :return: np.ndarray of the union of all the coloured pixels of all objects
        """
        result = self.objects[0].get_coloured_pixels_positions()
        for obj in self.objects[1:]:
            result = union2d(result, obj.get_coloured_pixels_positions())

        return result

    def where_object_fits_on_canvas(self, obj: Primitive) -> List[Point]:
        """
        Finds all the points on the Canvas that an Object can be placed (Object.canvas_pos) so that it is at least
        2/3 within the Canvas and that it is over and under other Objects on the Canvas by their required_dist_to_others
        :param obj: The Object to check
        :return:
        """
        available_canvas_points = []
        if np.any((self.size - obj.dimensions).to_numpy() < [0, 0]):
            return available_canvas_points
        for x in range(-obj.dimensions.dx//4, self.size.dx - 3 * obj.dimensions.dx//4 - 1):
            for y in range(-obj.dimensions.dy//4, self.size.dy - 3 * obj.dimensions.dy//4 - 1):
                obj.canvas_pos = Point(x, y, 0)
                overlap = False
                for obj_b in self.objects:
                    if do_two_objects_overlap(obj, obj_b):
                        overlap = True
                if not overlap:
                    available_canvas_points.append(Point(x, y, 0))

        return available_canvas_points

    def embed_objects(self):
        """
        Embeds all objects in the self.objects list onto the self.actual_pixels of the canvas. It uses the objects
        canvas_pos.z to define the order (objects with smaller z go first thus end up behind objects with larger z)
        :return:
        """
        self.actual_pixels = np.ndarray.copy(self.background_pixels)

        self.objects = sorted(self.objects, key=lambda obj: obj._canvas_pos.z)

        for i, obj in enumerate(self.objects):
            xmin = 0
            xmin_canv = obj.canvas_pos.x
            if xmin_canv >= self.actual_pixels.shape[1]:
                continue
            if xmin_canv < 0:
                xmin = np.abs(xmin_canv)
                xmin_canv = 0

            xmax = obj.dimensions.dx
            xmax_canv = obj.canvas_pos.x + obj.dimensions.dx
            if xmax_canv >= self.actual_pixels.shape[1]:
                xmax -= xmax_canv - self.actual_pixels.shape[1]
                xmax_canv = self.actual_pixels.shape[1]

            ymin = 0
            ymin_canv = obj.canvas_pos.y
            if ymin_canv >= self.actual_pixels.shape[0]:
                continue
            if ymin_canv < 0:
                ymin = np.abs(ymin_canv)
                ymin_canv = 0

            ymax = obj.dimensions.dy
            ymax_canv = obj.canvas_pos.y + obj.dimensions.dy
            if ymax_canv >= self.actual_pixels.shape[0]:
                ymax -= ymax_canv - self.actual_pixels.shape[0]
                ymax_canv = self.actual_pixels.shape[0]

            xmin = int(xmin)
            xmax = int(xmax)
            ymin = int(ymin)
            ymax = int(ymax)

            self.actual_pixels[ymin_canv: ymax_canv, xmin_canv: xmax_canv] = \
                obj.actual_pixels[ymin:ymax, xmin:xmax]
        self.full_canvas[0: self.size.dy, 0:self.size.dx] = self.actual_pixels

    def add_new_object(self, obj: Object):
        self.objects.append(obj)
        obj.canvas_id = self.id
        self.embed_objects()

    def remove_object(self, obj: Object):
        self.objects.remove(obj)
        self.embed_objects()

    def create_background_from_object(self, obj: Object):
        xmin = int(obj.canvas_pos.x)
        if xmin >= self.actual_pixels.shape[1]:
            return
        if xmin < 0:
            xmin = 0
        xmax = int(obj.canvas_pos.x + obj.dimensions.dx)
        if xmax >= self.actual_pixels.shape[1]:
            xmax = self.actual_pixels.shape[1]
        ymin = int(obj.canvas_pos.y)
        if ymin >= self.actual_pixels.shape[0]:
            return
        if ymin < 0:
            ymin = 0
        ymax = int(obj.canvas_pos.y + obj.dimensions.dy)
        if ymax >= self.actual_pixels.shape[0]:
            ymax = self.actual_pixels.shape[0]

        self.background_pixels[ymin: ymax, xmin: xmax] = obj.actual_pixels[: ymax - ymin, : xmax - xmin]
        self.embed_objects()

    def position_object(self, index: int, canvas_pos: Point):
        """
        Positions the object (with id = index) to the canvas_pos specified (the bottom left pixel of the object is
        placed to that canvas_pos)
        :param index: The id of the object
        :param canvas_pos: The Point specifying the coordinates on the canvas of the bottom left pixel of the object
        :return:
        """
        self.objects[index]._canvas_pos = canvas_pos
        self.embed_objects()

    def show(self, full_canvas=True, fig_to_add: None | plt.Figure = None, nrows: int = 0, ncoloumns: int = 0,
             index: int = 1):

        if full_canvas:
            xmin = - 0.5
            xmax = self.full_canvas.shape[1] - 0.5
            ymin = - 0.5
            ymax = self.full_canvas.shape[0] - 0.5
            extent = [xmin, xmax, ymin, ymax]
            if fig_to_add is None:
                vis.plot_data(self.full_canvas, extent=extent)
            else:
                ax = fig_to_add.add_subplot(nrows, ncoloumns, index)
                _ = vis.plot_data(self.full_canvas, extent=extent, axis=ax)
        else:
            xmin = - 0.5
            xmax = self.actual_pixels.shape[1] - 0.5
            ymin = - 0.5
            ymax = self.actual_pixels.shape[0] - 0.5
            extent = [xmin, xmax, ymin, ymax]
            if fig_to_add is None:
                fig, ax = vis.plot_data(self.actual_pixels, extent=extent)
            else:
                ax = fig_to_add.add_subplot(nrows, ncoloumns, index)
                _ = vis.plot_data(self.actual_pixels, extent=extent, axis=ax)

