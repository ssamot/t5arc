

from __future__ import annotations

from data_generators.object_recognition.canvas import *
from data_generators.object_recognition.basic_geometry import Point, Vector, Bbox, Orientation, Dimension2D, OrientationZ

np.random.seed(const.RANDOM_SEED_FOR_NUMPY)
MAX_EXAMPLE_PAIRS = const.MAX_EXAMPLE_PAIRS
MIN_PAD_SIZE = const.MIN_PAD_SIZE
MAX_PAD_SIZE = const.MAX_PAD_SIZE


class Example:
    def __init__(self):
        self.number_of_io_pairs = np.random.randint(2, const.MAX_EXAMPLE_PAIRS)
        self.number_of_canvasses = self.number_of_io_pairs * 2 + 1

        self.input_size = Dimension2D(np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE),
                                      np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE))
        self.output_size = Dimension2D(np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE),
                                       np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE))

        self.input_canvases = []
        self.output_canvases = []
        self.test_canvas = None

        self.ids = []
        self.objects = []

        #self.generate_canvasses()

    @staticmethod
    def get_random_colour(other_colour: int | None = None):

        colour = np.random.randint(1, len(const.COLOR_MAP))
        if other_colour is None:
            return colour

        else:
            while colour == other_colour:
                colour = np.random.randint(1, len(const.COLOR_MAP))
            return colour

    def create_random_object(self) -> Primitive:
        obj_type = ObjectType.random()

        id = self.ids[-1] + 1 if len(self.ids) > 0 else 0
        args = {'colour': self.get_random_colour(),
                'border_size': [0, 0, 0, 0],  # For now and then we will see
                'id': id,
                'canvas_pos': Point()}  # TODO: Deal with this given the other objects positions

        self.ids.append((id))

        if obj_type.name == 'InverseCross' or obj_type.name == 'Steps' or obj_type.name == 'Pyramid':
            args['height'] = np.random.randint(2, 10)
            if obj_type.name == 'InverseCross' and args['height'] % 2 == 0:
                args['height'] += 1

        if obj_type.name == 'InverseCross':
            fill_colour = self.get_random_colour(other_colour=args['colour'])
            args['fill_colour'] = fill_colour
            args['fill_height'] = np.random.randint(0, args['height'])

        if obj_type.name == 'Diagonal':
            args['length'] = np.random.randint((2, 10))

        if not np.any(np.array(['InverseCross', 'Steps', 'Pyramid', 'Dot', 'Diagonal', 'Fish', 'Bolt', 'Tie']) == obj_type.name):
            size = Dimension2D(np.random.randint(2, 10), np.random.randint(2, 10))
            if np.any(np.array(['Cross', '']) == obj_type.name):
                if size.dx % 2 == 0:
                    size.dx += 1
                if size.dy % 2 == 0:
                    size.dy += 1
            args['size'] = size

        if obj_type.name == 'Hole':
            up = np.random.randint(1, args['size'].dy - 2)
            down = np.random.randint(1, args['size'].dy - up)
            left = np.random.randint(1, args['size'].dx - 2)
            right = np.random.randint(1, args['size'].dx - left)
            args['thickness'] = [up, down, left, right]

        self.objects.append(globals()[obj_type.name](**args))

        return self.objects[-1]

    def generate_canvasses(self):
        for c in range(self.number_of_io_pairs):
            self.input_canvases.append(Canvas(size=self.input_size))
            self.output_canvases.append(Canvas(size=self.output_size))

    def show(self):
        pass