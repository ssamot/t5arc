

from __future__ import annotations

from copy import copy
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

from data.generators.example_generator.utils import do_two_objects_overlap
from data.generators.object_recognition.object import Object
from data.generators.object_recognition.utils import union2d
from data.generators import constants as const
from visualization import visualize_data as vis
from data.generators.object_recognition.primitives import Primitive, Random, Dot
from data.generators.object_recognition.basic_geometry import Point, Dimension2D

MAX_PAD_SIZE = const.MAX_PAD_SIZE


class Canvas:
    def __init__(self, size: Dimension2D | np.ndarray | List | None = None, objects: List[Primitive] | None = None,
                 _id: int | None = None, actual_pixels: np.ndarray | None = None):

        assert not(size is None and actual_pixels is None), print(f'Making a canvas with id {_id}. '
                                                                   f'Both size and actual_pixels are None!')

        if type(size) != Dimension2D and size is not None:
            self.size = Dimension2D(array=size)
        elif type(size) == Dimension2D:
            self.size = size

        if objects is None:
            self.objects = []
        else:
            self.objects = objects
        self.id = _id

        if actual_pixels is None:
            self.actual_pixels = np.ones((size.dy, size.dx))
        else:
            self.actual_pixels = actual_pixels
            self.size = Dimension2D(self.actual_pixels.shape[1], self.actual_pixels.shape[0])

        self.full_canvas = np.zeros((MAX_PAD_SIZE, MAX_PAD_SIZE))
        self.full_canvas[0: self.size.dy, 0:self.size.dx] = self.actual_pixels
        self.background_pixels = np.ndarray.copy(self.actual_pixels)

        self.embed_objects()

    def __repr__(self):
        return f'Canvas {self.id} with {len(self.objects)} Primitives'

    def __copy__(self):
        new_canvas = Canvas(size=self.size, _id=None)
        for o in self.objects:
            new_canvas.add_new_object(copy(o))
        return new_canvas

    def sort_objects_by_size(self, used_dim: str = 'area') -> List[Primitive]:
        """
        Returns a list of all the Object on Canvas sorted from largest to smallest according to the dimension used
        :param used_dim: The dimension to use to sort the Objects. It can be 'area', 'height', 'width', 'coloured_pixels'
        :return:
        """
        sorted_objects = np.array(copy(self.objects))
        dim = []
        for o in sorted_objects:
            if used_dim == 'area':
                metric = o.dimensions.dx * o.dimensions.dy
            elif used_dim == 'height':
                metric = o.dimensions.dy
            elif used_dim == 'length':
                metric = o.dimensions.dx
            elif used_dim == 'coloured_pixels':
                metric = len(o.get_coloured_pixels_positions())
            dim.append(metric)
        dim = np.array(dim)
        sorted_indices = dim.argsort()
        sorted_objects = sorted_objects[sorted_indices]

        return sorted_objects

    def group_objects_by_colour(self):
        pass

    def find_objects_of_colour(self, colour: int):
        result = []
        for obj in self.objects:
            if obj.colour == colour:
                result.append(obj)

        return result

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

        self.objects = sorted(self.objects, key=lambda obj: obj.canvas_pos.z)

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

            # The following will add to the canvas only the object's pixels that are not 1
            bbox_to_embed = copy(obj.actual_pixels[ymin:ymax, xmin:xmax])
            bbox_to_embed_in = copy(self.actual_pixels[ymin_canv: ymax_canv, xmin_canv: xmax_canv])
            bbox_to_embed_in[np.where(bbox_to_embed > 1)] = bbox_to_embed[np.where(bbox_to_embed > 1)]
            self.actual_pixels[ymin_canv: ymax_canv, xmin_canv: xmax_canv] = bbox_to_embed_in
            #self.actual_pixels[ymin_canv: ymax_canv, xmin_canv: xmax_canv] = \
            #    obj.actual_pixels[ymin:ymax, xmin:xmax]
        self.full_canvas[0: self.size.dy, 0:self.size.dx] = self.actual_pixels

    def add_new_object(self, obj: Object):
        self.objects.append(obj)
        obj.canvas_id = self.id
        self.embed_objects()

    def remove_object(self, obj: Object):
        self.objects.remove(obj)
        self.embed_objects()

    def split_object_by_colour(self, obj: Object) -> Dict:

        resulting_ids = {'id': [], 'actual_pixels_id': [], 'index': []}
        object_id = np.max([obj.id for obj in self.objects])
        actual_pixels_id = np.max([obj.actual_pixels_id for obj in self.objects])

        colours_in_places = obj.get_colour_groups()
        for col in colours_in_places:
            object_id += 1
            actual_pixels_id += 1
            canvas_pos = Point(colours_in_places[col][:, 1].min(), colours_in_places[col][:, 0].min())
            new_pixels_index = colours_in_places[col] - np.array([colours_in_places[col][:, 0].min(), colours_in_places[col][:, 1].min()])
            actual_pixels_index = colours_in_places[col] - np.array([obj.canvas_pos.y, obj.canvas_pos.x])

            size = Dimension2D(colours_in_places[col][:, 1].max() - colours_in_places[col][:, 1].min() + 1,
                               colours_in_places[col][:, 0].max() - colours_in_places[col][:, 0].min() + 1)

            if colours_in_places[col].shape[0] == 1:
                new_primitive = Dot(colour=col, canvas_pos=canvas_pos,
                                    _id=object_id, actual_pixels_id=actual_pixels_id)
            else:
                new_pixels = np.ones((size.dy, size.dx))
                new_pixels[new_pixels_index[:, 0], new_pixels_index[:, 1]] = \
                    obj.actual_pixels[actual_pixels_index[:, 0], actual_pixels_index[:, 1]]

                new_primitive = Random(size=Dimension2D(new_pixels.shape[1], new_pixels.shape[0]), canvas_pos=canvas_pos,
                                       _id=object_id, actual_pixels_id=actual_pixels_id)
                new_primitive.set_colour_to_most_common()
                new_primitive.actual_pixels[:, :] = new_pixels[:, :]
            resulting_ids['id'].append(object_id)
            resulting_ids['actual_pixels_id'].append(actual_pixels_id)
            self.add_new_object(new_primitive)
            resulting_ids['index'].append(len(self.objects) - 1)

        self.remove_object(obj)
        resulting_ids['index'] = np.array(resulting_ids['index']) - 1

        return resulting_ids

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

    def json_output(self, with_pixels: bool = False) -> dict:
        result = {'objects': []}
        if with_pixels:
            result['actual_pixels'] = self.actual_pixels.tolist()
            result['full_canvas'] = self.full_canvas.tolist()
        for o in self.objects:
            o_json = o.json_output()
            o_json.pop('id', None)
            o_json.pop('actual_pixels_id', None)
            o_json['actual_pixels'] = o.actual_pixels.tolist()
            result['objects'].append(o_json)

        return result

    def show(self, full_canvas=True, fig_to_add: None | plt.Figure = None, nrows: int = 0, ncoloumns: int = 0,
             index: int = 1, save_as: str | None = None, thin_lines: bool = False):

        if full_canvas:
            xmin = - 0.5
            xmax = self.full_canvas.shape[1] - 0.5
            ymin = - 0.5
            ymax = self.full_canvas.shape[0] - 0.5
            extent = [xmin, xmax, ymin, ymax]
            if fig_to_add is None:
                fig, _ = vis.plot_data(self.full_canvas, extent=extent, thin_lines=thin_lines)
            else:
                ax = fig_to_add.add_subplot(nrows, ncoloumns, index)
                _ = vis.plot_data(self.full_canvas, extent=extent, axis=ax, thin_lines=thin_lines)
        else:
            xmin = - 0.5
            xmax = self.actual_pixels.shape[1] - 0.5
            ymin = - 0.5
            ymax = self.actual_pixels.shape[0] - 0.5
            extent = [xmin, xmax, ymin, ymax]
            if fig_to_add is None:
                fig, _ = vis.plot_data(self.actual_pixels, extent=extent, thin_lines=thin_lines)
            else:
                ax = fig_to_add.add_subplot(nrows, ncoloumns, index)
                _ = vis.plot_data(self.actual_pixels, extent=extent, axis=ax, thin_lines=thin_lines)

        if fig_to_add is None and save_as is not None:
            fig.savefig(save_as)
            plt.close(fig)

