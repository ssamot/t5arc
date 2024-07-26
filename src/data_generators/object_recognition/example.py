

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
MIRROR_PROB = 0.05


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

        # TODO: Develop the Grd Primitive to be able to also create the Grid Experiment type
        self.experiment_type = np.random.choice(['Object', 'Symmetry', 'Grid'], p=[0.9, 0.1, 0])

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
    def get_random_position(obj: Primitive, canvas: Canvas) -> Point | None:
        available_positions = canvas.where_object_fits_on_canvas(obj=obj)
        if len(available_positions) > 0:
            return np.random.choice(available_positions)
        else:
            return None

    @staticmethod
    def do_random_transformations(obj: Primitive, debug: bool = False, num_of_transformations: int = 0,
                                  probs_of_transformations: List = (0.1, 0.2, 0.1, 0.1, 0.25, 0.25)):
        """
        Transform the obj Object random n times (0 to 3) with randomly selected Transformations (except mirror). The
        arguments of each Transformation are also chosen randomly
        :param obj: The Object to transform
        :param debug: If True print the transformations and their arguments
        :param probs_of_transformations: The probabilities of the 6 possible transformations (except mirror) in the
        order of the Transformations enumeration
        :return:
        """
        if debug: print(type(obj))

        if debug: print(f'number of transforms = {num_of_transformations}')

        possible_transform_indices = [0, 1, 2, 4, 5, 6]

        for i in range(num_of_transformations):
            random_transform_index = np.random.choice(possible_transform_indices, p=probs_of_transformations)
            #transform_index_index = np.where(possible_transform_indices == random_transform_index)[0][0]
            #possible_transform_indices.remove(random_transform_index)
            #del probs_of_transformations[transform_index_index]
            transform_name = Transformations(random_transform_index)
            if debug: print(f'Transform = {transform_name.name}')

            args = transform_name.get_random_parameters()
            if debug: print(f'Arguments = {args}')

            transform_method = getattr(obj, transform_name.name)
            transform_method(**args)

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
        """
        Create a random Primitive
        :return: The Object generated
        """
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

        if obj_type.name == 'Steps':  # Steps also has depth
            args['depth'] = np.random.randint(1, args['height'])

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

        # Deal with the allowed minimum distance to other objects
        if 'length' in args:
            min_distance_to_others = Surround(Up=args['length'], Down=0, Left=0, Right=args['length'])
        elif 'height' in args:
            min_distance_to_others = Surround(Up=args['height'], Down=0, Left=0, Right=args['height'])
        else:
            min_distance_to_others = Surround(Up=args['size'].dy, Down=0, Left=0, Right=args['size'].dx)

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
        self.objects.append(object)

        return self.objects[-1]

    def generate_canvasses(self):
        """
        Generate random size canvases and add background (single colour) pixels to 10% of them is they are bigger than
        10x10
        :return:
        """
        for c in range(self.number_of_io_pairs):
            input_size = Dimension2D(np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE),
                                     np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE))
            output_size = Dimension2D(np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE),
                                      np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE))

            input_canvas = Canvas(size=input_size)
            if np.all([input_size.dx > 10, input_size.dy > 10]) and np.random.random() < 0.1:
                background_object = Random(size=input_size, occupancy_prob=np.random.gamma(1, 0.05)+0.1)
                input_canvas.create_background_from_object(background_object)
            self.input_canvases.append(input_canvas)

            output_canvas = Canvas(size=output_size)
            if np.all([output_size.dx > 10, output_size.dy > 10]) and np.random.random() < 0.1:
                background_object = Random(size=output_size, occupancy_prob=np.random.gamma(1, 0.05)+0.1)
                output_canvas.create_background_from_object(background_object)
            self.output_canvases.append(output_canvas)

        test_canvas_size =Dimension2D(np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE),
                                                   np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE))
        self.test_canvas = Canvas(size=test_canvas_size)
        if np.all([test_canvas_size.dx > 10, test_canvas_size.dy > 10]) and np.random.random() < 0.1:
            background_object = Random(size=test_canvas_size, occupancy_prob=np.random.gamma(1, 0.05) + 0.1)
            self.test_canvas.create_background_from_object(background_object)

    def place_new_object_on_canvases(self):
        """
        Create a new object and put it on different canvases. The process is as follows.
        If the Example is of type 'Object:
        1) Randomly create a Primitive.
        2) Copy that Primitive random n number of times (all of these will have the same id)
        3) Randomly do a number of Transformations to every one of the cobjects.
        4) Randomly pick the canvasses to place each of the object in (of the possible ones given the other objects)
        :return:
        """
        if self.experiment_type == 'Object':
            objects = [self.create_random_object()]
            num_of_copies = np.random.randint(1, 5) if np.all(objects[-1].size.to_numpy() < 6) else \
                np.random.randint(4, 21)
            for _ in range(num_of_copies):
                objects.append(objects[-1].copy())

            num_of_transformations = np.random.choice([0, 1, 2, 3, 4], p=[0.05, 0.5, 0.3, 0.1, 0.05])
            probs_of_transformations = [0.1, 0.2, 0.05, 0.05, 0.3, 0.3]
            for obj in objects:
                self.do_random_transformations(obj, num_of_transformations=num_of_transformations,
                                               probs_of_transformations=probs_of_transformations)
                for input_canvas in self.input_canvases:
                    canvas_pos = self.get_random_position(obj, input_canvas)
                    if canvas_pos is not None:
                        obj.canvas_pos = canvas_pos
                        input_canvas.add_new_object(obj)

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


