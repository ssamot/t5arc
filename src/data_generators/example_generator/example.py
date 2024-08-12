
from __future__ import annotations

from data_generators.object_recognition.canvas import *
from data_generators.object_recognition.basic_geometry import Point, Dimension2D

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


