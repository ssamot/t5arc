

from __future__ import annotations

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

    def generate_random_objects(self):
        pass

    def randomise_empty_regions(self):
        pass

    def embed_objects(self):

        for object in self.objects:
            xmin = object.canvas_pos.y
            xmax = object.canvas_pos.y + object.dimensions.dy
            ymin = object.canvas_pos.x
            ymax = object.canvas_pos.x + object.dimensions.dx

            self.actual_pixels[xmin: xmax, ymin: ymax] = object.actual_pixels
            self.full_canvas[0: self.size.dy, 0:self.size.dx] = self.actual_pixels

    def show(self, full_canvas=True):

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
