

from __future__ import annotations

from enum import Enum

import numpy as np

from data_generators.object_recognition.object import Transformations
from data_generators.object_recognition.canvas import *
from data_generators.object_recognition.basic_geometry import Point, Vector, Bbox, Orientation, Dimension2D, OrientationZ

np.random.seed(const.RANDOM_SEED_FOR_NUMPY)
MAX_EXAMPLE_PAIRS = const.MAX_EXAMPLE_PAIRS
MIN_PAD_SIZE = const.MIN_PAD_SIZE
MAX_PAD_SIZE = const.MAX_PAD_SIZE

OVERLAP_PROB = 0.8
FAR_AWAY_PROB = 0.1
MIRROR_PROB = 0.05
MAX_NUM_OF_DIFFERENT_PRIMITIVES = 5
LARGE_OBJECT_THRESHOLD = 10
MAX_NUM_OF_LARGE_OBJECTS = 3
MIN_NUM_OF_SMALL_OBJECTS = 2
MAX_NUM_OF_SMALL_OBJECTS = 6
NUMBER_OF_TRANSFORMATIONS = 1
MAX_NUM_OF_SAME_OBJECTS_ON_CANVAS = 2

MIN_CANVAS_SIZE_FOR_BACKGROUND_OBJ = 10
PROB_OF_BACKGROUND_OBJ = 0.1

MAX_SIZE_OF_OBJECT = 15

MAX_NUMBER_OF_MIRRORS = 10


class Example:
    def __init__(self):
        self.number_of_io_pairs = np.random.randint(2, const.MAX_EXAMPLE_PAIRS)
        self.number_of_canvasses = self.number_of_io_pairs * 2 + 1

        self.input_canvases = []
        self.output_canvases = []
        self.test_canvas = None

        # TODO: Develop the Grd Primitive to be able to also create the Grid Experiment type
        self.experiment_type = np.random.choice(['Object', 'Symmetry', 'Grid'], p=[0.8, 0.2, 0])

        self.ids = []
        self.actual_pixel_ids = []
        self.objects = []
        self.temp_objects = []

        self.generate_canvasses()

    @staticmethod
    def get_random_colour(other_colour: int | None = None):
        colour = np.random.randint(2, len(const.COLOR_MAP))
        if other_colour is None:
            return colour

        else:
            while colour == other_colour:
                colour = np.random.randint(2, len(const.COLOR_MAP))
            return colour

    @staticmethod
    def get_random_position(obj: Primitive, canvas: Canvas) -> Point | None:
        """
        Finds all the positions an object would fit on the canvas (given where the other objects are) and
        gives back a random one of these.
        :param obj: The object to position
        :param canvas: The Canvas to position the object in
        :return:
        """
        available_positions = canvas.where_object_fits_on_canvas(obj=obj)
        if len(available_positions) > 0:
            return np.random.choice(available_positions)
        else:
            return None

    def do_random_transformations(self, obj: Primitive, debug: bool = False, num_of_transformations: int = 0,
                                  probs_of_transformations: List = (0.1, 0.2, 0.1, 0.1, 0.25, 0.25)):
        """
        Transform the obj Object random n times (0 to 3) with randomly selected Transformations (except mirror). The
        arguments of each Transformation are also chosen randomly
        :param obj: The Object to transform
        :param debug: If True print the transformations and their arguments
        :param num_of_transformations: How many transformations to do
        :param probs_of_transformations: The probabilities of the 6 possible transformations (except mirror) in the
        order of the Transformations enumeration
        :return:
        """
        if debug: print(type(obj))

        if debug: print(f'number of transforms = {num_of_transformations}')

        possible_transform_indices = [0, 1, 2, 4, 5, 6]

        for i in range(num_of_transformations):
            random_transform_index = np.random.choice(possible_transform_indices, p=probs_of_transformations)

            transform_name = Transformations(random_transform_index)
            if debug: print(f'Transform = {transform_name.name}')

            obj_type = 'Random' if obj.get_str_type() == 'Random' else 'Non-Random'
            args = transform_name.get_random_parameters(obj_type)
            if debug: print(f'Arguments = {args}')

            transform_method = getattr(obj, transform_name.name)

            # Do not allow Cross or InvertedCross to scale by an even factor
            if np.any(np.array(['Cross', 'InvertedCross']) == obj.get_str_type()) and \
                transform_name == Transformations.scale and args['factor'] % 2 == 0:
                continue

            transform_method(**args)

            obj.actual_pixels_id = self.actual_pixel_ids[-1] + 1
            self.actual_pixel_ids.append(obj.actual_pixels_id)

    def do_multiple_mirroring(self, obj: Primitive) -> Primitive:
        """
        Mirror an object multiple times over random directions. Make sure the final size is not larger than the
        maximum canvas sise.
        :param obj: The object to mirror
        :return:
        """
        number_of_mirrors = np.random.randint(2, MAX_NUMBER_OF_MIRRORS)
        transform_index = 3

        obj_to_mirror = copy(obj)

        for i in range(number_of_mirrors):
            mirror_name = Transformations(transform_index)
            args = mirror_name.get_random_parameters()
            mirror_method = getattr(obj_to_mirror, mirror_name.name)
            mirror_method(**args)

        if np.any(obj_to_mirror.dimensions.to_numpy() > MAX_PAD_SIZE):
            obj_to_mirror = self.do_multiple_mirroring(obj)

        return obj_to_mirror

    def create_random_object(self, debug: bool = False) -> Primitive:
        """
        Create a random Primitive
        :return: The Object generated
        """
        obj_type = ObjectType.random()

        if debug: print(obj_type)

        id = self.ids[-1] + 1 if len(self.ids) > 0 else 0
        actual_pixels_id = self.actual_pixel_ids[-1] + 1 if len(self.actual_pixel_ids) > 0 else 0
        args = {'colour': self.get_random_colour(),
                'border_size': Surround(0, 0, 0, 0),  # For now and then we will see
                'canvas_pos': Point(0, 0, 0),
                '_id': id,
                'actual_pixels_id': actual_pixels_id}
        self.ids.append(id)
        self.actual_pixel_ids.append(actual_pixels_id)

        if obj_type.name == 'InverseCross' or obj_type.name == 'Steps' or obj_type.name == 'Pyramid':  # These objects have height not size
            args['height'] = np.random.randint(2, MAX_SIZE_OF_OBJECT)
            if obj_type.name == 'InverseCross' and args['height'] % 2 == 0:  # Inverted Crosses need odd height
                args['height'] += 1

        if obj_type.name == 'InverseCross':
            fill_colour = self.get_random_colour(other_colour=args['colour'])
            args['fill_colour'] = fill_colour
            args['fill_height'] = np.random.randint(1, args['height'] - 2) if args['height'] > 3 else 0
            if args['fill_height'] % 2 == 0:  # Inverted Crosses need odd fill_height
                args['fill_height'] += 1

        if obj_type.name == 'Diagonal':  # Diagonal has length, not size
            args['length'] = np.random.randint(2, MAX_SIZE_OF_OBJECT)

        if obj_type.name == 'Steps':  # Steps also has depth
            args['depth'] = np.random.randint(1, args['height'])

        if not np.any(np.array(['InverseCross', 'Steps', 'Pyramid', 'Dot', 'Diagonal', 'Fish', 'Bolt', 'Tie'])
                      == obj_type.name):
            size = Dimension2D(np.random.randint(2, MAX_SIZE_OF_OBJECT), np.random.randint(2, MAX_SIZE_OF_OBJECT))
            if np.any(np.array(['Cross', 'InvertedCross']) == obj_type.name):   # Crosses need odd size
                if size.dx % 2 == 0:
                    size.dx += 1
                if size.dy % 2 == 0:
                    size.dy += 1
            args['size'] = size

        if obj_type.name == 'Hole':  # Hole has also thickness
            if args['size'].dx < 4:
                args['size'].dx = 4
            if args['size'].dy < 4:
                args['size'].dy = 4
            up = np.random.randint(1, args['size'].dy - 2)
            down = np.random.randint(1, args['size'].dy - up)
            left = np.random.randint(1, args['size'].dx - 2)
            right = np.random.randint(1, args['size'].dx - left)
            args['thickness'] = Surround(up, down, left, right)

        if debug: print(args)

        object = globals()[obj_type.name](**args)

        # Deal with the allowed minimum distance to other objects.
        # Create a standard distance and then check if the object allows for overlap or needs to be far away from
        # others and change the distance to others accordingly
        if 'length' in args:
            min_distance_to_others = Surround(Up=args['length'], Down=0, Left=0, Right=args['length'])
        elif 'height' in args:
            min_distance_to_others = Surround(Up=args['height'], Down=0, Left=0, Right=args['height'])
        elif 'size' in args:
            min_distance_to_others = Surround(Up=args['size'].dy, Down=0, Left=0, Right=args['size'].dx)
        else:
            min_distance_to_others = Surround(Up=1, Down=1, Left=1, Right=1)

        min_distance_to_others += args['border_size']

        allow_overlap = np.random.random() < OVERLAP_PROB
        if allow_overlap and np.all((min_distance_to_others - 2).to_numpy() > 1):
            min_distance_to_others -= Surround(Up=np.random.randint(1, min_distance_to_others.Up - 2),
                                               Down=np.random.randint(1, min_distance_to_others.Down - 2),
                                               Left=np.random.randint(1, min_distance_to_others.Left - 2),
                                               Right=np.random.randint(1, min_distance_to_others.Right - 2))
        else:
            far_away = np.random.random() < FAR_AWAY_PROB
            if far_away and np.all((min_distance_to_others - 2).to_numpy() > 1):
                min_distance_to_others += Surround(Up=np.random.randint(1, min_distance_to_others.Up - 2),
                                                   Down=np.random.randint(1, min_distance_to_others.Down - 2),
                                                   Left=np.random.randint(1, min_distance_to_others.Left - 2),
                                                   Right=np.random.randint(1, min_distance_to_others.Right - 2))

        object.required_dist_to_others = min_distance_to_others

        return object

    def generate_canvasses(self):
        """
        Generate random size canvases and add background (single colour) pixels to 10% of them if they are bigger than
        10x10
        :return:
        """
        min_pad_size = MIN_PAD_SIZE if self.experiment_type == 'Object' else MIN_PAD_SIZE + 10

        for c in range(self.number_of_io_pairs):
            input_size = Dimension2D(np.random.randint(min_pad_size, MAX_PAD_SIZE),
                                     np.random.randint(min_pad_size, MAX_PAD_SIZE))
            output_size = Dimension2D(np.random.randint(min_pad_size, MAX_PAD_SIZE),
                                      np.random.randint(min_pad_size, MAX_PAD_SIZE))

            input_canvas = Canvas(size=input_size, _id=2*c + 1)
            if np.all([input_size.dx > MIN_CANVAS_SIZE_FOR_BACKGROUND_OBJ,
                       input_size.dy > MIN_CANVAS_SIZE_FOR_BACKGROUND_OBJ]) \
                    and np.random.random() < PROB_OF_BACKGROUND_OBJ:
                background_object = Random(size=input_size, occupancy_prob=np.random.gamma(1, 0.05)+0.1)
                input_canvas.create_background_from_object(background_object)
            self.input_canvases.append(input_canvas)

            output_canvas = Canvas(size=output_size, _id=2*c + 2)
            if np.all([output_size.dx > MIN_CANVAS_SIZE_FOR_BACKGROUND_OBJ,
                       output_size.dy > MIN_CANVAS_SIZE_FOR_BACKGROUND_OBJ]) \
                    and np.random.random() < PROB_OF_BACKGROUND_OBJ:
                background_object = Random(size=output_size, occupancy_prob=np.random.gamma(1, 0.05)+0.1)
                output_canvas.create_background_from_object(background_object)
            self.output_canvases.append(output_canvas)

        test_canvas_size = Dimension2D(np.random.randint(min_pad_size, MAX_PAD_SIZE),
                                       np.random.randint(min_pad_size, MAX_PAD_SIZE))

        self.test_canvas = Canvas(size=test_canvas_size, _id=0)
        if np.all([test_canvas_size.dx > MIN_CANVAS_SIZE_FOR_BACKGROUND_OBJ,
                   test_canvas_size.dy > MIN_CANVAS_SIZE_FOR_BACKGROUND_OBJ]) \
                and np.random.random() < PROB_OF_BACKGROUND_OBJ:
            background_object = Random(size=test_canvas_size, occupancy_prob=np.random.gamma(1, 0.05) + 0.1)
            self.test_canvas.create_background_from_object(background_object)

    def randomly_position_object_in_all_canvases(self, obj: Primitive):
        """
        It takes an object and places it in random (but allowed) positions in all the Canvases of the Example
        :param obj: The object to position
        :return:
        """
        for input_canvas, output_canvas in zip(self.input_canvases, self.output_canvases):
            in_canvas_pos = self.get_random_position(obj, input_canvas)
            out_canvas_pos = self.get_random_position(obj, output_canvas)
            canvas_pos = None
            c = None
            if in_canvas_pos is not None and out_canvas_pos is not None:
                canvas_pos, c = (in_canvas_pos, input_canvas) if np.random.random() > 0.5 \
                    else (out_canvas_pos, output_canvas)
            elif in_canvas_pos is None:
                canvas_pos = out_canvas_pos
                c = output_canvas
            elif out_canvas_pos is None:
                canvas_pos = in_canvas_pos
                c = input_canvas
            if canvas_pos is not None:
                o = copy(obj)
                o.canvas_pos = canvas_pos
                c.add_new_object(o)
                self.objects.append(o)

        test_canvas_pos = self.get_random_position(obj, self.test_canvas)
        if test_canvas_pos is not None:
            o = copy(obj)
            o.canvas_pos = test_canvas_pos
            self.test_canvas.add_new_object(o)
            self.objects.append(o)

    def place_new_object_on_canvases(self):
        """
        Create a new object and put it on different canvases. The process is as follows.
        If the Example is of type 'Object:
        1) Randomly create a Primitive.
        2) Copy that Primitive random n number of times (all of these will have the same id)
        3) Randomly do a number of Transformations to every one of the cobjects.
        4) Randomly pick the canvasses to place each of the object in (of the possible ones given the other objects)
        If the Example is of type 'Symmetry':
        1) Randomly create a Primitive
        2) Mirror it random times (with random Orientations)
        3) Randomly pick the canvasses to place it
        :return:
        """
        if self.experiment_type == 'Object':

            self.temp_objects = [self.create_random_object(debug=False)]

            num_of_transformed_copies = np.random.randint(1, MAX_NUM_OF_LARGE_OBJECTS) \
                if np.any(self.temp_objects[-1].size.to_numpy() > LARGE_OBJECT_THRESHOLD) else \
                np.random.randint(MIN_NUM_OF_SMALL_OBJECTS, MAX_NUM_OF_SMALL_OBJECTS)

            for _ in range(num_of_transformed_copies):
                self.temp_objects.append(copy(self.temp_objects[-1]))

            num_of_transformations = NUMBER_OF_TRANSFORMATIONS
            probs_of_transformations = [0.1, 0.2, 0.05, 0.05, 0.3, 0.3]
            for k, obj in enumerate(self.temp_objects):

                if k > 0:  # Leave one object copy untransformed
                    self.do_random_transformations(obj, num_of_transformations=num_of_transformations, debug=False,
                                                   probs_of_transformations=probs_of_transformations)

                for _ in range(MAX_NUM_OF_SAME_OBJECTS_ON_CANVAS):
                    self.randomly_position_object_in_all_canvases(obj)

        elif self.experiment_type == 'Symmetry':

            # Make sure the base object of a symmetry object is not a Parallelogram
            base_object = Parallelogram(size=[2, 2])
            while base_object.get_str_type() == 'Parallelogram':
                base_object = self.create_random_object(debug=False)

            obj = self.do_multiple_mirroring(base_object)
            self.randomly_position_object_in_all_canvases(obj)

    def populate_canvases(self):
        """
        Generates a random number of objects (if experiment_type is 'Object') or just one object (if type is 'Symmetry')
        and then places their transformations on all the canvases (in allowed positions)
        :return:
        """
        num_of_objects = 1
        if self.experiment_type == 'Object':
            num_of_objects = np.random.randint(1, MAX_NUM_OF_DIFFERENT_PRIMITIVES)

        for _ in range(num_of_objects):
            self.place_new_object_on_canvases()

    def create_canvas_arrays_input(self) -> dict:
        result = {'test': [{'input': np.flipud(self.test_canvas.full_canvas).tolist()}],
                  'train': []}

        for input_canvas, output_canvas in zip(self.input_canvases, self.output_canvases):
            result['train'].append({'input': np.flipud(input_canvas.full_canvas).tolist(),
                                    'output': np.flipud(output_canvas.full_canvas).tolist()})

        return result

    def create_output(self):
        obj_pix_ids = []
        obj_with_unique_pix_ids = []
        positions_of_same_objects = {}
        actual_pixels_list = []
        unique_objects = []
        for obj in self.objects:
            obj_pix_id = (obj.id, obj.actual_pixels_id)
            if not obj_pix_id in obj_pix_ids:
                unique_objects.append(obj.json_output())
                actual_pixels_list.append(obj.actual_pixels)
                obj_pix_ids.append(obj_pix_id)
                obj_with_unique_pix_ids.append(obj)
                positions_of_same_objects[obj_pix_id] = [obj.canvas_id,
                                                         [obj.canvas_pos.x, obj.canvas_pos.y, obj.canvas_pos.z]]
            else:
                positions_of_same_objects[obj_pix_id].append([obj.canvas_id,
                                                              [obj.canvas_pos.x, obj.canvas_pos.y, obj.canvas_pos.z]])

        actual_pixels_array = np.zeros((MAX_PAD_SIZE, MAX_PAD_SIZE, len(actual_pixels_list)))
        for i, ap in enumerate(actual_pixels_list):
            actual_pixels_array[:ap.shape[0], :ap.shape[1], i] = ap

        return unique_objects, actual_pixels_array, positions_of_same_objects

    def show(self, canvas_index: int | str = 'all'):
        """
        Shows some (canvas_index is int or 'test') or all (canvas_index = 'all') the Canvases of the Experiment
        :param canvas_index: Which Canvases to show (int, 'test' or 'all')
        :return:
        """
        if type(canvas_index) == int:
            if canvas_index % 2 == 0:
                canvas = self.input_canvases[canvas_index // 2]
            else:
                canvas = self.output_canvases[canvas_index // 2]
            canvas.show()

        elif canvas_index == 'test':
            self.test_canvas.show()

        elif canvas_index == 'all':
            fig = plt.figure()
            index = 1
            nrows = self.number_of_io_pairs
            ncoloumns = 3
            for p in range(self.number_of_io_pairs):
                self.input_canvases[p].show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=index)
                self.output_canvases[p].show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=index + 1)
                if p == 0:
                    self.test_canvas.show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=index + 2)
                index += 3
            plt.tight_layout()


