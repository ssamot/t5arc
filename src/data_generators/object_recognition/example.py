

from __future__ import annotations

from data_generators.object_recognition.canvas import *
from data_generators.object_recognition.basic_geometry import Point, Vector, Bbox, Orientation, Dimension2D, OrientationZ

np.random.seed(const.RANDOM_SEED_FOR_NUMPY)
MAX_EXAMPLE_PAIRS = const.MAX_EXAMPLE_PAIRS
MIN_PAD_SIZE = const.MIN_PAD_SIZE
MAX_PAD_SIZE = const.MAX_PAD_SIZE

OVERLAP_PROB = 0.8
FAR_AWAY_PROB = 0.1

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

    def get_random_position(self, obj: Primitive, canvas: Canvas) -> Point:
        available_positions = canvas.where_object_fits_on_canvas(obj=obj)
        return np.random.choice(available_positions)
    
    def do_random_transformations(self, obj: Primitive) -> Primitive:
        pass

    def create_random_object(self) -> Primitive:
        obj_type = ObjectType.random()

        id = self.ids[-1] + 1 if len(self.ids) > 0 else 0
        args = {'colour': self.get_random_colour(),
                'border_size': Surround(0, 0, 0, 0),  # For now and then we will see
                'canvas_pos': Point(0, 0, 0),
                'id': id}
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
            args['length'] = np.random.randint((2, 10))

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
            args['thickness'] = [up, down, left, right]

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
        object.canvas_pos = self.get_random_position()

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


