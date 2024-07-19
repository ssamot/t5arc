

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from utils import *
from visualization import visualize_data as vis
from data_generators.object_recognition.primitives import *
from data_generators.object_recognition.basic_geometry import Point, Vector, Bbox, Orientation, Dimension2D, OrientationZ


np.random.seed(const.RANDOM_SEED_FOR_NUMPY)
MAX_PAD_SIZE = const.MAX_PAD_SIZE


class Canvas:
    def __init__(self, size: Dimension2D | np.ndarray | List, objects: List[Object] | None, num_of_objects:int = 0):

        if type(size) != Dimension2D:
            self.size = Dimension2D(array=size)
        else:
            self.size = size

        if objects is None:
            self.objects = []
            self.num_of_objects = num_of_objects
        else:
            self.objects = objects
            self.num_of_objects = len(objects)

        self.actual_pixels = np.ones((size[1], size[0]))
        self.full_canvas = np.zeros((MAX_PAD_SIZE, MAX_PAD_SIZE))
        self.full_canvas[0: self.size.dy, 0:self.size.dx] = self.actual_pixels

        self.embed_objects()

    def get_coloured_pixels_positions(self) -> np.ndarray:
        """
        Returns the Union of the positions of the coloured pixels of all the objects in the self.object list
        :return: np.ndarray of the union of all the coloured pixels of all objects
        """
        result = self.objects[0].get_coloured_pixels_positions()
        for obj in self.objects[1:]:
            result = union2d(result, obj.get_coloured_pixels_positions())

        return result

    def generate_random_objects(self):
        pass

    def randomise_empty_regions(self):
        pass

    def embed_objects(self):
        """
        Embeds all objects in the self.objects list onto the self.actual_pixels of the canvas. It uses the objects
        canvas_pos.z to define the order (objects with smaller z go first thus end up behind objects with larger z)
        :return:
        """
        self.actual_pixels[:, :] = 1

        self.objects = sorted(self.objects, key=lambda obj: obj.canvas_pos.z)

        for obj in self.objects:
            xmin = obj.canvas_pos.x
            if xmin >= self.actual_pixels.shape[1]:
                continue
            xmax = obj.canvas_pos.x + obj.dimensions.dx
            if xmax >= self.actual_pixels.shape[1]:
                xmax = self.actual_pixels.shape[1]
            ymin = obj.canvas_pos.y
            if ymin >= self.actual_pixels.shape[0]:
                continue
            ymax = obj.canvas_pos.y + obj.dimensions.dy
            if ymax >= self.actual_pixels.shape[0]:
                ymax = self.actual_pixels.shape[0]

            self.actual_pixels[ymin: ymax, xmin: xmax] = obj.actual_pixels[: ymax-ymin, : xmax-xmin]
            self.full_canvas[0: self.size.dy, 0:self.size.dx] = self.actual_pixels

    def position_object(self, index: int, canvas_pos: Point):
        """
        Positions the object (with id = index) to the canvas_pos specified (the bottom left pixel of the object is
        placed to that canvas_pos)
        :param index: The id of the object
        :param canvas_pos: The Point specifying the coordinates on the canvas of the bottom left pixel of the object
        :return:
        """
        self.objects[index].canvas_pos = canvas_pos
        self.embed_objects()

    def show(self, full_canvas=True) -> tuple[plt.Figure, {plt.vlines, plt.hlines}]:

        if full_canvas:
            xmin = - 0.5
            xmax = self.full_canvas.shape[1] - 0.5
            ymin = - 0.5
            ymax = self.full_canvas.shape[0] - 0.5
            extent = [xmin, xmax, ymin, ymax]
            fig, ax = vis.plot_data(self.full_canvas, extent=extent)
        else:
            xmin = - 0.5
            xmax = self.actual_pixels.shape[1] - 0.5
            ymin = - 0.5
            ymax = self.actual_pixels.shape[0] - 0.5
            extent = [xmin, xmax, ymin, ymax]
            fig, ax = vis.plot_data(self.actual_pixels, extent=extent)

        return fig, ax
