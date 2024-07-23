

from __future__ import annotations

import matplotlib.pyplot as plt
from utils import *
from data.utils import *
from visualization import visualize_data as vis
from data_generators.object_recognition.primitives import *
from data_generators.object_recognition.basic_geometry import Point, Vector, Bbox, Orientation, Dimension2D, OrientationZ


np.random.seed(const.RANDOM_SEED_FOR_NUMPY)
MAX_PAD_SIZE = const.MAX_PAD_SIZE


class Canvas:
    def __init__(self, size: Dimension2D | np.ndarray | List, objects: List[Object] | None = None, num_of_objects:int = 0):

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

        self.actual_pixels = np.ones((size.dy, size.dx))
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

    def where_object_fits_on_canvas(self, obj: Primitive) -> List[Point]:
        available_canvas_points = []
        for x in range(-obj.size.dx//2 + 1, self.size.dx - obj.size.dx//2):
            for y in range(-obj.size.dy//2 + 1, self.size.dy - obj.size.dy//2):
                obj.canvas_pos = Point(x, y, 0)
                overlap = False
                for obj_b in self.objects:
                    #print(obj_b.bbox)
                    if do_two_objects_overlap(obj, obj_b):
                        overlap = True
                    #print(obj_b.bbox)
                    #print('------')
                if not overlap:
                    available_canvas_points.append(Point(x, y, 0))

        return available_canvas_points

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

        self.objects = sorted(self.objects, key=lambda obj: obj._canvas_pos.z)

        for obj in self.objects:
            xmin = obj._canvas_pos.x
            if xmin >= self.actual_pixels.shape[1]:
                continue
            if xmin < 0:
                xmin = 0
            xmax = obj._canvas_pos.x + obj.dimensions.dx
            if xmax >= self.actual_pixels.shape[1]:
                xmax = self.actual_pixels.shape[1]
            ymin = obj._canvas_pos.y
            if ymin >= self.actual_pixels.shape[0]:
                continue
            if ymin < 0:
                ymin = 0
            ymax = obj._canvas_pos.y + obj.dimensions.dy
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
        self.objects[index]._canvas_pos = canvas_pos
        self.embed_objects()

    def show(self, full_canvas=True, fig_to_add: None | plt.Figure = None, nrows: int = 0, ncoloumns: int = 0,
             index: int = 1) -> tuple[plt.Figure, {plt.vlines, plt.hlines}]:

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
            #fig, ax = vis.plot_data(self.actual_pixels, extent=extent)

