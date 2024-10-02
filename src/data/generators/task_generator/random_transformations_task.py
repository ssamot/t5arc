

from __future__ import annotations

from typing import List, Dict, Tuple

import numpy as np
from copy import copy

from matplotlib import pyplot as plt

from data.generators.object_recognition.canvas import Canvas
from data.generators.task_generator.task import Task
from data.generators.object_recognition.basic_geometry import Dimension2D, Point, Vector, RelativePoint, Orientation, \
    Bbox
from data.generators.object_recognition.primitives import Primitive, Parallelogram
from data.generators import constants as const
from data.generators.object_recognition.object import Transformations, Object

MAX_EXAMPLE_PAIRS = const.MAX_EXAMPLE_PAIRS
MIN_PAD_SIZE = const.MIN_PAD_SIZE
MAX_PAD_SIZE = const.MAX_PAD_SIZE

MIRROR_PROB = 0.05
MAX_NUM_OF_DIFFERENT_PRIMITIVES = 3
LARGE_OBJECT_THRESHOLD = 15
NUMBER_OF_INPUT_TRANSFORMATIONS = 5
NUMBER_OF_OUTPUT_TRANSFORMATIONS = 3

MIN_CANVAS_SIZE_FOR_BACKGROUND_OBJ = 10
PROB_OF_BACKGROUND_OBJ = 0.0

MAX_OVERLAP_RATIO = 0.5
MAX_SIZE_OF_OBJECT = 15

MAX_NUMBER_OF_MIRRORS = 10

PROBS_OF_INPUT_TRANSFORMATIONS = {'translate_to_coordinates': 0,
                                  'translate_by': 0,
                                  'translate_along': 0,
                                  'translate_relative_point_to_point': 0,
                                  'translate_until_touch': 0,
                                  'translate_until_fit': 0,
                                  'rotate': 0.2,
                                  'scale': 0.2,
                                  'shear': 0,
                                  'mirror': 0.1,
                                  'flip': 0.1,
                                  'grow': 0,
                                  'randomise_colour': 0.1,
                                  'randomise_shape': 0.1,
                                  'replace_colour': 0.2,
                                  'replace_all_colours': 0,
                                  'delete': 0,
                                  'fill': 0}


PROBS_OF_OUTPUT_TRANSFORMATIONS = {'translate_to_coordinates': 0,
                                   'translate_by': 0.1,
                                   'translate_along': 0.1,
                                   'translate_relative_point_to_point': 0,
                                   'translate_until_touch': 0.2,
                                   'translate_until_fit': 0.2,
                                   'rotate': 0.1,
                                   'scale': 0.1,
                                   'shear': 0,
                                   'mirror': 0.1,
                                   'flip': 0,
                                   'grow': 0,
                                   'randomise_colour': 0,
                                   'randomise_shape': 0,
                                   'replace_colour': 0.1,
                                   'replace_all_colours': 0,
                                   'delete': 0,
                                   'fill': 0}


class RandomTransformationsTask(Task):
    def __init__(self, num_of_outputs: int = 10):
        super().__init__(min_canvas_size_for_background_object=MIN_CANVAS_SIZE_FOR_BACKGROUND_OBJ,
                         prob_of_background_object=0, number_of_io_pairs=1)

        self.num_of_outputs = num_of_outputs

        min_pad_size = const.MIN_PAD_SIZE
        self.output_canvases = []
        for i in range(self.num_of_outputs):
            output_size = Dimension2D(np.random.randint(min_pad_size, const.MAX_PAD_SIZE),
                                      np.random.randint(min_pad_size, const.MAX_PAD_SIZE))

            output_canvas = Canvas(size=output_size, _id=i + 2)
            self.output_canvases.append(output_canvas)

    @staticmethod
    def redimension_canvas_if_required(canvas: Canvas, min_dimensions: Dimension2D):
        if canvas.size.dx < min_dimensions.dx:
            canvas.resize_canvas(Dimension2D(int(min_dimensions.dx + 1), canvas.size.dy))
            print(f'Redimensioned canvas to {canvas.size}')
        if canvas.size.dy < min_dimensions.dy:
            canvas.resize_canvas(Dimension2D( canvas.size.dx, int(min_dimensions.dy + 1)))
            print(f'Redimensioned canvas tp {canvas.size}')

    def do_output_transformations_with_random_parameters(self, obj: Primitive, transformations: List[int],
                                                         for_canvas: Canvas,
                                                         to_other_obj: Primitive | None = None):

        for transform_index in transformations:

            transform_function = Transformations(transform_index)

            random_obj_or_not = 'Random' if obj.get_str_type() == 'Random' else 'Non-Random'

            print(obj.get_str_type(), obj.id, transform_function)

            args = {}
            if transform_function.name == 'translate_to_coordinates':
                self.redimension_canvas_if_required(for_canvas,
                                                    Dimension2D(np.ceil(obj.dimensions.dx / 2),
                                                                np.ceil(obj.dimensions.dy / 2)))
                args['target_point'] = Point.random(min_x=0, max_x=for_canvas.size.dx - np.ceil(obj.dimensions.dx / 2),
                                                    min_y=0, max_y=for_canvas.size.dy - np.ceil(obj.dimensions.dy / 2),
                                                    min_z=-10, max_z=10)
                args['object_point'] = Point.random(min_x=obj.canvas_pos.x, max_x=obj.canvas_pos.x + obj.dimensions.dx,
                                                    min_y=obj.canvas_pos.y, max_y=obj.canvas_pos.y + obj.dimensions.dy,
                                                    min_z=0, max_z=0)
            if transform_function.name == 'translate_by':
                self.redimension_canvas_if_required(for_canvas,
                                                    Dimension2D(obj.canvas_pos.x + np.ceil(obj.dimensions.dx / 2),
                                                                obj.canvas_pos.y + np.ceil(obj.dimensions.dy / 2)))
                args['distance'] = Dimension2D.random(min_dx=-obj.canvas_pos.x - np.ceil(obj.dimensions.dx / 2),
                                                      max_dx=for_canvas.size.dx - obj.canvas_pos.x - np.ceil(obj.dimensions.dx / 2),
                                                      min_dy=-obj.canvas_pos.y - np.ceil(obj.dimensions.dy / 2),
                                                      max_dy=for_canvas.size.dy - obj.canvas_pos.y - np.ceil(obj.dimensions.dy / 2))
            if transform_function.name == 'translate_along':
                orientation_probs_mask = np.array([1]*8)
                if obj.canvas_pos.x <= 0:
                    orientation_probs_mask = orientation_probs_mask * np.array([1, 1, 1, 1, 1, 0, 0, 0])
                if obj.canvas_pos.y <= 0:
                    orientation_probs_mask = orientation_probs_mask * np.array([1, 1, 1, 0, 0, 0, 1, 1])
                if obj.canvas_pos.x >= for_canvas.size.dx - obj.dimensions.dx:
                    orientation_probs_mask = orientation_probs_mask * np.array([1, 0, 0, 0, 1, 1, 1, 1])
                if obj.canvas_pos.y >= for_canvas.size.dy - obj.dimensions.dy:
                    orientation_probs_mask = orientation_probs_mask * np.array([0, 0, 1, 1, 1, 1, 1, 0])

                orientation = Orientation(np.random.choice(range(8),
                                                           p=orientation_probs_mask / orientation_probs_mask.sum()))

                if orientation == Orientation.Up:
                    length = np.random.randint(1, for_canvas.size.dy - obj.canvas_pos.y - obj.dimensions.dy + 1)
                if orientation == Orientation.Up_Right:
                    length = np.random.randint(1, np.min([for_canvas.size.dy - obj.canvas_pos.y - obj.dimensions.dy + 1,
                                                          for_canvas.size.dx - obj.canvas_pos.x - obj.dimensions.dx + 1]))
                if orientation == Orientation.Right:
                    length = np.random.randint(1, for_canvas.size.dx - obj.canvas_pos.x - obj.dimensions.dx + 1)
                if orientation == Orientation.Down_Right:
                    length = np.random.randint(1, np.min([obj.canvas_pos.y + 1,
                                                          for_canvas.size.dx - obj.canvas_pos.x - obj.dimensions.dx + 1]))
                if orientation == Orientation.Down:
                    length = np.random.randint(1, obj.canvas_pos.y + 1)
                if orientation == Orientation.Down_Left:
                    length = np.random.randint(1, np.min([obj.canvas_pos.y + 1,
                                                          obj.canvas_pos.x + 1]))
                if orientation == Orientation.Left:
                    length = np.random.randint(1, obj.canvas_pos.x + 1)
                if orientation == Orientation.Up_Left:
                    length = np.random.randint(1, np.min([for_canvas.size.dy - obj.canvas_pos.y - obj.dimensions.dy + 1,
                                                          obj.canvas_pos.x + 1]))

                args['direction'] = Vector(orientation=orientation, length=length, origin=obj.canvas_pos)
            if transform_function.name == 'translate_relative_point_to_point':
                self.redimension_canvas_if_required(for_canvas,
                                                    Dimension2D(np.ceil(obj.dimensions.dx / 2),
                                                                np.ceil(obj.dimensions.dy / 2)))
                args['relative_point'] = RelativePoint.random()
                args['other_point'] = Point.random(min_x=np.ceil(obj.dimensions.dx / 2),
                                                   max_x=for_canvas.size.dx - np.ceil(obj.dimensions.dx / 2),
                                                   min_y=np.ceil(obj.dimensions.dy / 2),
                                                   max_y=for_canvas.size.dy - np.ceil(obj.dimensions.dy / 2),
                                                   min_z=obj.canvas_pos.z, max_z=obj.canvas_pos.z)
            if transform_function.name in ['translate_until_touch', 'translate_until_fit']:
                if to_other_obj is not None:
                    args['other'] = to_other_obj
                else:
                    objects_ids = [o.id for o in for_canvas.objects if o.id != obj.id]
                    choice_id = np.random.choice(objects_ids)
                    args['other'] = [o for o in for_canvas.objects if o.id == choice_id][0]
            if transform_function.name == 'rotate':
                args['times'] = np.random.randint(1, 4)
            if transform_function.name == 'scale':
                scale_probs_mask = np.array([1] * 6)
                if for_canvas.size.dx - obj.canvas_pos.x < 3 * obj.dimensions.dx or\
                        for_canvas.size.dy - obj.canvas_pos.y < 3 * obj.dimensions.dy:
                    scale_probs_mask = scale_probs_mask * np.array([1, 1, 1, 1, 1, 0])
                if for_canvas.size.dx - obj.canvas_pos.x < 2 * obj.dimensions.dx or \
                        for_canvas.size.dy - obj.canvas_pos.y < 2 * obj.dimensions.dy:
                    scale_probs_mask = scale_probs_mask * np.array([1, 1, 1, 1, 0, 0])
                if for_canvas.size.dx - obj.canvas_pos.x < 1 * obj.dimensions.dx or \
                        for_canvas.size.dy - obj.canvas_pos.y < 1 * obj.dimensions.dy:
                    scale_probs_mask = scale_probs_mask * np.array([1, 1, 1, 0, 0, 0])
                if obj.dimensions.dx < 8 or obj.dimensions.dy < 8:
                    scale_probs_mask = scale_probs_mask * np.array([0, 1, 1, 1, 1, 1])
                if obj.dimensions.dx < 6 or obj.dimensions.dy < 6:
                    scale_probs_mask = scale_probs_mask * np.array([0, 0, 1, 1, 1, 1])
                if obj.dimensions.dx < 4 or obj.dimensions.dy < 4:
                    scale_probs_mask = scale_probs_mask * np.array([0, 0, 0, 1, 1, 1])

                if scale_probs_mask.sum() > 0:
                    args['factor'] = np.random.choice([-4, -3, -2, 2, 3, 4], p=scale_probs_mask / scale_probs_mask.sum())
                else:
                    args['factor'] = 1
            if transform_function.name == 'shear':
                if random_obj_or_not == 'Random':
                    args['_shear'] = int(np.random.gamma(shape=1, scale=15) + 10)  # Mainly between 1 and 75
                else:
                    args['_shear'] = int(np.random.gamma(shape=1, scale=10) + 5)  # Mainly between 0.05 and 0.4
                    args['_shear'] = 40 if args['_shear'] > 40 else args['_shear']
            if transform_function.name == 'mirror' or transform_function.name == 'flip':
                args['axis'] = np.random.choice([Orientation.Up, Orientation.Down, Orientation.Left, Orientation.Right])
            if transform_function.name == 'mirror':
                args['on_axis'] = False if np.random.rand() < 0.5 else True
            if transform_function.name == 'randomise_colour':
                if random_obj_or_not == 'Random':
                    args['ratio'] = int(np.random.gamma(shape=2, scale=10) + 1)  # Mainly between 10 and 40
                    args['ratio'] = int(60) if args['ratio'] > 60 else args['ratio']
                else:
                    args['ratio'] = int(np.random.gamma(shape=3, scale=5) + 2)  # Mainly between 10 and 40
                    args['ratio'] = int(60) if args['ratio'] > 60 else args['ratio']
            if transform_function.name == 'randomise_shape':
                args['add_or_subtract'] = 'add' if np.random.random() > 0.5 else 'subtract'
                if random_obj_or_not == 'Random':
                    args['ratio'] = int(np.random.gamma(shape=3, scale=7) + 1)  # Mainly between 0.1 and 0.3
                    args['ratio'] = 50 if args['ratio'] > 50 else args['ratio']
                else:
                    args['ratio'] = int(np.random.gamma(shape=3, scale=5) + 2)  # Mainly between 0.1 and 0.3
                    args['ratio'] = 50 if args['ratio'] > 50 else args['ratio']
            if transform_function.name == 'replace_colour':
                args['initial_colour'] = obj.get_most_common_colour()
                args['final_colour'] = np.random.choice(np.arange(2, 11)[np.arange(2, 11) != args['initial_colour']])
            if transform_function.name == 'replace_all_colours':
                new_colours = np.arange(2, 11)
                np.random.shuffle(new_colours)
                args['colour_swap_hash'] = {2: new_colours[0], 3: new_colours[1], 4: new_colours[2], 5: new_colours[3],
                                            6: new_colours[4], 7: new_colours[5], 8: new_colours[6], 9: new_colours[7],
                                            10: new_colours[8]}
            if transform_function.name == 'fill':
                args['colour'] = np.random.randint(2, 10)

            transform_method = getattr(obj, transform_function.name)
            transform_method(**args)
            try:
                a = args['other']
                print(args['other'].id)
            except:
               print(args)

            print(f'New object pos and dims {obj.canvas_pos} // {obj.dimensions}')
            #print(f'New canvas size {for_canvas.size}')

        print('---')

    def get_list_of_output_transformations(self, num_of_output_transformations: int) -> List[int]:
        output_transformations_indices = []
        for _ in range(num_of_output_transformations):
            possible_transform_indices = range(len(Transformations))
            random_transform_index = np.random.choice(possible_transform_indices,
                                                      p=list(PROBS_OF_OUTPUT_TRANSFORMATIONS.values()))
            if random_transform_index in [4, 5] and \
                    (len(self.input_canvases[0].objects) < 2 or \
                     4 in output_transformations_indices or \
                     5 in output_transformations_indices):
                return self.get_list_of_output_transformations(num_of_output_transformations)
            output_transformations_indices.append(random_transform_index)

        return output_transformations_indices

    def randomise_canvas_pos(self, obj: Primitive, for_canvas: Canvas, fully_in_canvas: bool = False):

        too_much_overlap = True

        while too_much_overlap:
            too_much_overlap = False

            region_outside_canvas = Dimension2D(obj.dimensions.dx, obj.dimensions.dy) if fully_in_canvas else \
                Dimension2D(np.ceil(obj.dimensions.dx / 2), np.ceil(obj.dimensions.dx / 2))
            max_x = for_canvas.size.dx - region_outside_canvas.dx if \
                for_canvas.size.dx - region_outside_canvas.dx > 1 else 1
            max_y = for_canvas.size.dy - region_outside_canvas.dy if \
                for_canvas.size.dy - region_outside_canvas.dy > 1 else 1
            possible_canvas_pos = Point.random(min_x=0, max_x=int(max_x),
                                               min_y=0, max_y=int(max_y),
                                               min_z=-10, max_z=10)
            obj_temp = copy(obj)
            obj_temp.canvas_pos = possible_canvas_pos
            obj_area = obj_temp.dimensions.dx * obj_temp.dimensions.dy

            for o in for_canvas.objects:
                o_area = o.dimensions.dx * o.dimensions.dy

                x_intersection = max(obj_temp.bbox.top_left.x, o.bbox.top_left.x)
                y_intersection = max(obj_temp.bbox.bottom_right.y, o.bbox.bottom_right.y)
                w_intersection = min(obj_temp.bbox.top_left.x + obj_temp.dimensions.dx,
                                     o.bbox.top_left.x + o.dimensions.dx) - x_intersection
                h_intersection = min(obj_temp.bbox.bottom_right.y + obj_temp.dimensions.dy,
                                     o.bbox.bottom_right.y + o.dimensions.dy) - y_intersection

                intersection_area = w_intersection * h_intersection
                if intersection_area > MAX_OVERLAP_RATIO * np.min([obj_area, o_area]):
                    too_much_overlap = True

        obj.canvas_pos = possible_canvas_pos

    def get_min_dimension_of_all_canvasses(self):
        canvasses_dims = []
        for i in self.input_canvases:
            canvasses_dims.append(i.size.dx)
            canvasses_dims.append(i.size.dy)
        for o in self.output_canvases:
            canvasses_dims.append(o.size.dx)
            canvasses_dims.append(o.size.dy)

        return np.min(canvasses_dims)

    def place_new_objects_on_output_canvases(self, input_objects_with_transformations: List[List[Primitive, List[int]]]):
        """
        Takes an object together with a series of transformations that need to be done on that object and does those
        number of output_canvases times and puts each resulting object in an output canvas.
        :return:
        """
        for out_canvas in self.output_canvases:
            print(f'=== CANVAS {out_canvas.id} ======')
            print(f'CANVAS size {out_canvas.size}')

            for obj_and_trans in input_objects_with_transformations:

                output_obj = copy(obj_and_trans[0])
                output_transformations = obj_and_trans[1]

                print(f'Object {output_obj.get_str_type()} {output_obj.id} canvas pos = {output_obj.canvas_pos}'
                      ' and dims = {output_obj.dimensions}')
                self.do_output_transformations_with_random_parameters(obj=output_obj,
                                                                      transformations=output_transformations,
                                                                      for_canvas=out_canvas)
                output_obj.id = output_obj.id
                self.objects.append(output_obj)
                out_canvas.add_new_object(output_obj)

            dims_x = []
            dims_y = []
            for obj in out_canvas.objects:
                out_of_bounds = Point(obj.dimensions.dx // 3,
                                      obj.dimensions.dy // 3) if np.random.rand() > 0.8 else Point(0, 0)
                dims_x.append(obj.bbox.bottom_right.x - out_of_bounds.x)
                dims_y.append(obj.bbox.top_left.y - out_of_bounds.y)
            if out_canvas.size.dx < np.max(dims_x):
                target_size = int(np.max(dims_x)) if np.max(dims_x) <= MAX_PAD_SIZE else MAX_PAD_SIZE
                out_canvas.resize_canvas(Dimension2D(target_size, out_canvas.size.dy))
            if out_canvas.size.dy < np.max(dims_y):
                target_size = int(np.max(dims_y)) if np.max(dims_y) <= MAX_PAD_SIZE else MAX_PAD_SIZE
                out_canvas.resize_canvas(Dimension2D(out_canvas.size.dx, target_size))

            print('============')

    def generate_sample(self):
        """
        This is the main function to call to generate the Random Experiment.
        It generates a random number of objects (if experiment_type is 'Object') or just one object (if type is
        'Symmetry') and then places their transformations on all the canvases (in allowed positions)
        :return:
        """

        num_of_objects = np.random.randint(1, MAX_NUM_OF_DIFFERENT_PRIMITIVES + 1)

        input_objects = []

        for _ in range(num_of_objects):
            max_size_of_object = np.min([MAX_SIZE_OF_OBJECT, self.get_min_dimension_of_all_canvasses()])
            input_objects.append(self.create_object(debug=False,
                                                   max_size_of_obj=Dimension2D(max_size_of_object,
                                                                               max_size_of_object),
                                                   overlap_prob=1, far_away_prob=1))
        input_objects_with_transformations = []
        print(f'Number of input objects" {len(input_objects)}')

        for obj in input_objects:

            num_of_input_transformations = np.random.randint(1, NUMBER_OF_INPUT_TRANSFORMATIONS + 1)
            probs_of_input_transformations = list(PROBS_OF_INPUT_TRANSFORMATIONS.values())

            self.do_random_transformations(obj, num_of_transformations=num_of_input_transformations,
                                           debug=False,
                                           probs_of_transformations=probs_of_input_transformations)

            self.randomise_canvas_pos(obj, self.input_canvases[0], fully_in_canvas=True)

            obj.id = 0 if len(self.input_canvases[0].objects) == 0 else \
                self.input_canvases[0].objects[-1].id + 1
            self.add_object_on_canvasses(obj, [1])

            num_of_output_transformations = np.random.randint(1, NUMBER_OF_OUTPUT_TRANSFORMATIONS + 1)
            output_transformations = self.get_list_of_output_transformations(num_of_output_transformations)

            input_objects_with_transformations.append([obj, output_transformations])

        self.place_new_objects_on_output_canvases(input_objects_with_transformations)

        print('==========')
        for obj_and_trans in input_objects_with_transformations:
            output_obj = copy(obj_and_trans[0])
            output_transformations = obj_and_trans[1]
            trans = [Transformations(i).name for i in output_transformations]
            print(f'OBJECT {output_obj.get_str_type()} : {output_obj.id} TRANSFORMED: {trans}')

    def get_cnavasses_as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:

        n = self.num_of_outputs
        num_of_colours = 11
        x = np.zeros((n, 32, 32, num_of_colours))
        y = np.zeros((n, 32, 32, num_of_colours))

        for i in range(n):
            in_pixels = self.input_canvases[0].full_canvas.astype(int)
            x[i, :, :, :] = np.eye(num_of_colours)[in_pixels]
            out_pixels = self.output_canvases[i].full_canvas.astype(int)
            y[i, :, :, :] = np.eye(num_of_colours)[out_pixels]

        return x, y

    def show(self, canvas_index: int | str = 'all', save_as: str | None = None, two_cols: bool = False):
        thin_lines = True
        if save_as is None:
            thin_lines = False

        fig = plt.figure(figsize=(6, 16))
        index = 1
        ncoloumns = 4
        nrows = int(np.ceil((self.num_of_outputs + 1) / ncoloumns))

        self.input_canvases[0].show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=index,
                                    thin_lines=thin_lines)
        index += 1
        for i in range(self.num_of_outputs):
            self.output_canvases[i].show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=index,
                                         thin_lines=thin_lines)
            index += 1

