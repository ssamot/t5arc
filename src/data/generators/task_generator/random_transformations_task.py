

from __future__ import annotations

from typing import List, Dict

import numpy as np
from copy import copy

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
MAX_NUM_OF_SAME_LARGE_OBJECTS = 2
MIN_NUM_OF_SAME_SMALL_OBJECTS = 1
MAX_NUM_OF_SAME_SMALL_OBJECTS = 2
NUMBER_OF_INPUT_TRANSFORMATIONS = 5
NUMBER_OF_OUTPUT_TRANSFORMATIONS = 2
MAX_NUM_OF_SAME_OBJECTS_ON_CANVAS = 2

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


PROBS_OF_OUTPUT_TRANSFORMATIONS = {'translate_to_coordinates': 0.1,
                                   'translate_by': 0.1,
                                   'translate_along': 0.1,
                                   'translate_relative_point_to_point': 0,
                                   'translate_until_touch': 0.1,
                                   'translate_until_fit': 0.1,
                                   'rotate': 0.1,
                                   'scale': 0.1,
                                   'shear': 0,
                                   'mirror': 0.1,
                                   'flip': 0.1,
                                   'grow': 0,
                                   'randomise_colour': 0,
                                   'randomise_shape': 0,
                                   'replace_colour': 0.1,
                                   'replace_all_colours': 0,
                                   'delete': 0,
                                   'fill': 0}


class RandomTransformationsTask(Task):
    def __init__(self):
        super().__init__(min_canvas_size_for_background_object=MIN_CANVAS_SIZE_FOR_BACKGROUND_OBJ,
                         prob_of_background_object=0, number_of_io_pairs=1)

    def do_multiple_mirroring(self, obj: Primitive, number_of_mirrors: int | None = None) -> Primitive | None:
        """
        Mirror an object multiple times over random directions. Make sure the final size is not larger than the
        maximum canvas sise.
        :param obj: The object to mirror
        :return:
        """

        if number_of_mirrors is None:
            number_of_mirrors = np.random.randint(2, MAX_NUMBER_OF_MIRRORS)
        if number_of_mirrors == 0:
            return None

        obj_to_mirror = copy(obj)

        for i in range(number_of_mirrors):
            mirror_name = Transformations.get_transformation_from_name('mirror')
            args = mirror_name.get_random_parameters()
            mirror_method = getattr(obj_to_mirror, mirror_name.name)
            mirror_method(**args)

        if np.any(obj_to_mirror.dimensions.to_numpy() > MAX_PAD_SIZE):
            obj_to_mirror = self.do_multiple_mirroring(obj, number_of_mirrors=number_of_mirrors - 1)

        return obj_to_mirror

    @staticmethod
    def do_output_transformations_with_random_parameters(obj: Primitive, transformations: List[int],
                                                         for_canvas: Canvas,
                                                         to_other_obj: Primitive | None = None):

        for transform_index in transformations:
            try:
                transform_function = Transformations(transform_index)

                random_obj_or_not = 'Random' if obj.get_str_type() == 'Random' else 'Non-Random'

                print(obj.get_str_type(), obj.id, transform_function)

                args = {}
                if transform_function.name == 'translate_to_coordinates':
                    args['target_point'] = Point.random(min_x=0, max_x=for_canvas.size.dx - np.ceil(obj.dimensions.dx / 2),
                                                        min_y=0, max_y=for_canvas.size.dy - np.ceil(obj.dimensions.dy / 2),
                                                        min_z=-10, max_z=10)
                    args['object_point'] = Point.random(min_x=obj.canvas_pos.x, max_x=obj.canvas_pos.x + obj.dimensions.dx,
                                                        min_y=obj.canvas_pos.y, max_y=obj.canvas_pos.y + obj.dimensions.dy,
                                                        min_z=0, max_z=0)
                if transform_function.name == 'translate_by':
                    args['distance'] = Dimension2D.random(min_dx=-obj.canvas_pos.x + np.ceil(obj.dimensions.dx / 2),
                                                          max_dx=for_canvas.size.dx - obj.canvas_pos.x - np.ceil(obj.dimensions.dx / 2),
                                                          min_dy=-obj.canvas_pos.y + np.ceil(obj.dimensions.dy / 2),
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
                    if obj.dimensions.dx < 12 and obj.dimensions.dy < 12:
                        scale_probs_mask = scale_probs_mask * np.array([0, 1, 1, 1, 1, 1])
                    if obj.dimensions.dx < 9 and obj.dimensions.dy < 9:
                        scale_probs_mask = scale_probs_mask * np.array([0, 0, 1, 1, 1, 1])
                    if obj.dimensions.dx < 6 and obj.dimensions.dy < 6:
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
                print(args)
            except:
               pass

        print('---')

    def get_list_of_output_transformations(self, num_of_output_transformations: int) -> List[int]:
        output_transformations_indices = []
        for _ in range(num_of_output_transformations):
            possible_transform_indices = range(len(Transformations))
            random_transform_index = np.random.choice(possible_transform_indices,
                                                      p=list(PROBS_OF_OUTPUT_TRANSFORMATIONS.values()))
            if random_transform_index in [4, 5] and len(self.output_canvases[0].objects) == 0:
                return self.get_list_of_output_transformations(num_of_output_transformations)
            output_transformations_indices.append(random_transform_index)

        return output_transformations_indices

    @ staticmethod
    def randomise_canvas_pos(obj: Primitive, for_canvas: Canvas):

        too_much_overlap = True

        while too_much_overlap:
            too_much_overlap = False
            possible_canvas_pos = Point.random(min_x=0, max_x=for_canvas.size.dx - np.ceil(obj.dimensions.dx / 2),
                                               min_y=0, max_y=for_canvas.size.dy - np.ceil(obj.dimensions.dy / 2),
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

    def place_new_object_on_canvases(self):
        """
        Create a new object and put it on different canvases. The process is as follows.
        If the Task is of type 'Object:
        1) Randomly create a Primitive.
        2) Copy that Primitive random n number of times (all of these will have the same id)
        3) Randomly do a number of Transformations to every one of the cobjects.
        4) Randomly pick the canvasses to place each of the object in (of the possible ones given the other objects)
        If the Task is of type 'Symmetry':
        1) Randomly create a Primitive
        2) Mirror it random times (with random Orientations)
        3) Randomly pick the canvasses to place it
        :return:
        """
        if self.experiment_type == 'Object':

            max_size_of_object = np.min([MAX_SIZE_OF_OBJECT, self.get_min_dimension_of_all_canvasses()])
            self.temp_objects = [self.create_object(debug=False,
                                                    max_size_of_obj=Dimension2D(max_size_of_object, max_size_of_object),
                                                    overlap_prob=1, far_away_prob=1)]

            num_of_transformed_copies = np.random.randint(0, MAX_NUM_OF_SAME_LARGE_OBJECTS) \
                if np.any(self.temp_objects[-1].size.to_numpy() > LARGE_OBJECT_THRESHOLD) else \
                np.random.randint(MIN_NUM_OF_SAME_SMALL_OBJECTS - 1, MAX_NUM_OF_SAME_SMALL_OBJECTS)

            for _ in range(num_of_transformed_copies):
                self.temp_objects.append(copy(self.temp_objects[-1]))

            num_of_input_transformations = np.random.randint(1, NUMBER_OF_INPUT_TRANSFORMATIONS + 1)
            num_of_output_transformations = np.random.randint(1, NUMBER_OF_OUTPUT_TRANSFORMATIONS + 1)
            probs_of_input_transformations = list(PROBS_OF_INPUT_TRANSFORMATIONS.values())
            for k, obj in enumerate(self.temp_objects):

                self.randomise_canvas_pos(obj, self.input_canvases[0])

                if k > 0:  # Leave one object copy untransformed
                    self.do_random_transformations(obj, num_of_transformations=num_of_input_transformations, debug=False,
                                                   probs_of_transformations=probs_of_input_transformations)

                # Check that the object is not larger than the input canvas
                if self.input_canvases[0].size.dx > obj.dimensions.dx and\
                        self.input_canvases[0].size.dy > obj.dimensions.dy:
                    if obj.canvas_pos.x < 0:
                        obj.canvas_pos = Point(np.random.randint(0, self.input_canvases[0].size.dx - obj.dimensions.dx),
                                               obj.canvas_pos.y)
                    if obj.canvas_pos.y < 0:
                        obj.canvas_pos = Point(obj.canvas_pos.x,
                                               np.random.randint(0, self.input_canvases[0].size.dy - obj.dimensions.dy))

                    obj.id = 0 if len(self.input_canvases[0].objects) == 0 else \
                        self.input_canvases[0].objects[-1].id + 1
                    self.add_object_on_canvasses(obj, [1])

                    output_obj = copy(obj)

                    output_transformations = self.get_list_of_output_transformations(num_of_output_transformations)
                    self.do_output_transformations_with_random_parameters(obj=output_obj,
                                                                          transformations=output_transformations,
                                                                          for_canvas=self.output_canvases[0])
                    output_obj.id = obj.id
                    self.add_object_on_canvasses(output_obj, [2])

        elif self.experiment_type == 'Symmetry':

            # Make sure the base object of a symmetry object is not a Parallelogram
            base_object = Parallelogram(size=[2, 2])
            while base_object.get_str_type() == 'Parallelogram':
                base_object = self.create_object(debug=False)

            self.randomise_canvas_pos(base_object, self.input_canvases[0])
            base_object.id = 0 if len(self.input_canvases[0].objects) == 0 else self.input_canvases[0].objects[-1].id + 1
            self.add_object_on_canvasses(base_object, [1])
            output_obj = self.do_multiple_mirroring(base_object)
            if output_obj is not None:
                output_obj.id = base_object.id
                self.add_object_on_canvasses(output_obj, [2])
                print(output_obj.transformations)

    def generate_sample(self):
        """
        This is the main function to call to generate the Random Experiment.
        It generates a random number of objects (if experiment_type is 'Object') or just one object (if type is
        'Symmetry') and then places their transformations on all the canvases (in allowed positions)
        :return:
        """
        num_of_objects = 1
        if self.experiment_type == 'Object':
            num_of_objects = np.random.randint(1, MAX_NUM_OF_DIFFERENT_PRIMITIVES)

        for _ in range(num_of_objects):
            self.place_new_object_on_canvases()