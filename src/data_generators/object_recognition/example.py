

from __future__ import annotations

from enum import Enum

import numpy as np

from data_generators.object_recognition.canvas import *
from data_generators.object_recognition.basic_geometry import Point, Vector, Bbox, Orientation, Dimension2D, OrientationZ

np.random.seed(const.RANDOM_SEED_FOR_NUMPY)
MAX_EXAMPLE_PAIRS = const.MAX_EXAMPLE_PAIRS
MIN_PAD_SIZE = const.MIN_PAD_SIZE
MAX_PAD_SIZE = const.MAX_PAD_SIZE

OVERLAP_PROB = 0.8
FAR_AWAY_PROB = 0.1


class Transformations(Enum):
    scale: int = 0
    rotate: int = 1
    shear: int = 2
    mirror: int = 3
    flip: int = 4
    randomise_colour: int = 5
    randomise_shape: int = 6

    def get_random_parameters(self):
        args = {}
        if self.name == 'scale':
            args['factor'] = np.random.choice([-4, -3, -2, 2, 3, 4], p=[0.05, 0.15, 0.3, 0.3, 0.15, 0.05])
        if self.name == 'rotate':
            args['times'] = np.random.randint(1, 4)
        if self.name == 'shear':
            args['_shear'] = np.random.gamma(shape=1, scale=0.2) + 0.1  # Mainly between 0.1 and 0.75
        if self.name == 'mirror' or self.name == 'flip':
            args['axis'] = np.random.choice([Orientation.Up, Orientation.Down, Orientation.Left, Orientation.Right])
        if self.name == 'randomise_colour':
            args['ratio'] = np.random.gamma(shape=1, scale=0.1) + 0.1  # Mainly between 0.1 and 0.4
            args['ratio'] = 0.4 if args['ratio'] > 0.4 else args['ratio']
        if self.name == 'randomise_shape':
            args['add_or_subtract'] = 'add' if np.random.random() > 0.5 else 'subtract'
            args['ratio'] = np.random.gamma(shape=1, scale=0.07) + 0.1  # Mainly between 0.1 and 0.3
            args['ratio'] = 0.3 if args['ratio'] > 0.3 else args['ratio']

        return args


class Example:
    def __init__(self):
        self.number_of_io_pairs = np.random.randint(2, const.MAX_EXAMPLE_PAIRS)
        self.number_of_canvasses = self.number_of_io_pairs * 2 + 1

        self.input_canvases = []
        self.output_canvases = []
        self.test_canvas = None

        self.ids = []
        self.objects = []

        self.generate_canvasses()

    @staticmethod
    def get_random_colour(other_colour: int | None = None):

        colour = np.random.randint(1, len(const.COLOR_MAP))
        if other_colour is None:
            return colour

        else:
            while colour == other_colour:
                colour = np.random.randint(1, len(const.COLOR_MAP))
            return colour

    @staticmethod
    def get_random_position(obj: Primitive, canvas: Canvas) -> Point:
        available_positions = canvas.where_object_fits_on_canvas(obj=obj)
        return np.random.choice(available_positions)

    @staticmethod
    def do_random_transformations(obj: Primitive) -> Primitive:
        print(type(obj))
        number_of_transforms = np.random.choice([0, 1, 2, 3, 4], p=[0.05, 0.5, 0.3, 0.1, 0.05])
        print(f'number of transforms = {number_of_transforms}')
        possible_transform_indices = [0, 1, 2, 4, 5, 6]
        for i in range(number_of_transforms):
            random_transform_index = np.random.choice(possible_transform_indices)
            possible_transform_indices.remove(random_transform_index)
            transform_name = Transformations(random_transform_index)
            print(f'Transform = {transform_name.name}')
            args = transform_name.get_random_parameters()
            print(f'Arguments = {args}')
            transform_method = getattr(obj, transform_name.name)
            transform_method(**args)
        return obj

    @staticmethod
    def do_multiple_mirroring(obj: Primitive) -> Primitive:
        number_of_mirrors = np.random.randint(1, 5)
        transform_index = 3
        for i in range(number_of_mirrors):
            mirror_name = Transformations(transform_index)
            args = mirror_name.get_random_parameters()
            mirror_method = getattr(obj, mirror_name.name)
            mirror_method(**args)

        return obj

    def create_random_object(self) -> Primitive:
        obj_type = ObjectType.random()

        id = self.ids[-1] + 1 if len(self.ids) > 0 else 0
        args = {'colour': self.get_random_colour(),
                'border_size': Surround(0, 0, 0, 0),  # For now and then we will see
                'canvas_pos': Point(0, 0, 0),
                '_id': id}
        self.ids.append(id)

        if obj_type.name == 'InverseCross' or obj_type.name == 'Steps' or obj_type.name == 'Pyramid':  # These objects have height not size
            args['height'] = np.random.randint(2, 10)
            if obj_type.name == 'InverseCross' and args['height'] % 2 == 0:  # Inverted Crosses need odd height
                args['height'] += 1

        if obj_type.name == 'InverseCross':
            fill_colour = self.get_random_colour(other_colour=args['colour'])
            args['fill_colour'] = fill_colour
            args['fill_height'] = np.random.randint(0, args['height'])

        if obj_type.name == 'Diagonal':  # Diagonal has length, not size
            args['length'] = np.random.randint(2, 10)

        if not np.any(np.array(['InverseCross', 'Steps', 'Pyramid', 'Dot', 'Diagonal', 'Fish', 'Bolt', 'Tie'])
                      == obj_type.name):
            size = Dimension2D(np.random.randint(2, 10), np.random.randint(2, 10))
            if np.any(np.array(['Cross', '']) == obj_type.name):   # Crosses need odd size
                if size.dx % 2 == 0:
                    size.dx += 1
                if size.dy % 2 == 0:
                    size.dy += 1
            args['size'] = size

        if obj_type.name == 'Hole':  # Hole has also thickness
            up = np.random.randint(1, args['size'].dy - 2)
            down = np.random.randint(1, args['size'].dy - up)
            left = np.random.randint(1, args['size'].dx - 2)
            right = np.random.randint(1, args['size'].dx - left)
            args['thickness'] = Surround(up, down, left, right)

        object = globals()[obj_type.name](**args)
        object = self.do_random_transformations(object)

        # Deal with the allowed minimum distance to other objects
        if 'length' in args:
            min_distance_to_others = Dimension2D(args['length'], args['length'])
        elif 'height' in args:
            min_distance_to_others = Dimension2D(args['height'], args['height'])
        else:
            min_distance_to_others = args['size']

        min_distance_to_others += Dimension2D(args['border_size'].Right, args['border_size'].Up)

        allow_overlap = np.random.random() > OVERLAP_PROB
        if allow_overlap:
            min_distance_to_others -= Dimension2D(np.random.randint(1, min_distance_to_others.dx - 2),
                                                  np.random.randint(1, min_distance_to_others.dy - 2))
        else:
            far_away = np.random.random() > FAR_AWAY_PROB
            if far_away:
                min_distance_to_others -= Dimension2D(np.random.randint(1, min_distance_to_others.dx - 2),
                                                      np.random.randint(1, min_distance_to_others.dy - 2))

        object.required_distance_to_others = min_distance_to_others
        #object.canvas_pos = self.get_random_position(object)

        self.objects.append(object)

        return self.objects[-1]

    def generate_canvasses(self):
        for c in range(self.number_of_io_pairs):
            input_size = Dimension2D(np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE),
                                     np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE))
            output_size = Dimension2D(np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE),
                                      np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE))

            self.input_canvases.append(Canvas(size=input_size))
            self.output_canvases.append(Canvas(size=output_size))
        self.test_canvas = Canvas(size=Dimension2D(np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE),
                                                   np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE)))

    def place_new_object_on_canvases(self):
        """
        Create a new object and put it on different canvases. The process is as follows.
        1) Randomly create a Primitive.
        2) Randomly pick a number of transformations
        3) Do the transformations randomly picking their parameters
        4) Randomly pick the canvasses to place the object in (of the possible ones given the other objects)
        5) Randomly place the object in some of them (in allowed positions)
        6) Repeat steps 2 to 5 one more time so the same object appears on different canvases after some transformations
        :return:
        """
        obj = self.create_random_object()

    def populate_canvases(self):
        pass

    def show(self, canvas_index: int | str = 'all'):
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


