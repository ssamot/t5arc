

from __future__ import annotations
import numpy as np
from copy import copy
from matplotlib import pyplot as plt

from data_generators.example_generator.example import Example
from data_generators.object_recognition.basic_geometry import Dimension2D
from data_generators.object_recognition.canvas import Canvas
from data_generators.object_recognition.primitives import Primitive, Parallelogram, Random
from src import constants as const
from data_generators.object_recognition.object import Transformations

MIN_PAD_SIZE = 3
MAX_PAD_SIZE = const.MAX_PAD_SIZE  # 32

MIRROR_PROB = 0.05
MAX_NUM_OF_DIFFERENT_PRIMITIVES = 5
LARGE_OBJECT_THRESHOLD = 20
MAX_NUM_OF_LARGE_OBJECTS = 3
MIN_NUM_OF_SMALL_OBJECTS = 2
MAX_NUM_OF_SMALL_OBJECTS = 6
NUMBER_OF_TRANSFORMATIONS = 1
MAX_NUM_OF_SAME_OBJECTS_ON_CANVAS = 2

MIN_CANVAS_SIZE_FOR_BACKGROUND_OBJ = 10
PROB_OF_BACKGROUND_OBJ = 0.1

OVERLAP_PROB = 0.8
FAR_AWAY_PROB = 0.1
MAX_SIZE_OF_OBJECT = 30

MAX_NUMBER_OF_MIRRORS = 6


class AutoEncoderDataExample(Example):
    def __init__(self, number_of_canvases):
        super().__init__(min_canvas_size_for_background_object=MIN_CANVAS_SIZE_FOR_BACKGROUND_OBJ,
                         prob_of_background_object=PROB_OF_BACKGROUND_OBJ)
        self.number_of_io_pairs = number_of_canvases
        self.number_of_canvasses = number_of_canvases

        self.input_canvases = []
        self.generate_canvasses()
        self.randomly_populate_canvases()

    def generate_canvasses(self):
        """
        Generate random size canvases and add background (single colour) pixels to 10% of them if they are bigger than
        MIN_CANVAS_SIZE_FOR_BACKGROUND_OBJ. This generates only self.input_canvases
        :return:
        """
        min_pad_size = MIN_PAD_SIZE if self.experiment_type == 'Object' else MIN_PAD_SIZE + 10
        for c in range(self.number_of_canvasses):
            input_size = Dimension2D(np.random.randint(min_pad_size, MAX_PAD_SIZE),
                                     np.random.randint(min_pad_size, MAX_PAD_SIZE))

            input_canvas = Canvas(size=input_size, _id=c)
            if np.all([input_size.dx > self.min_canvas_size_for_background_object,
                       input_size.dy > self.min_canvas_size_for_background_object]) \
                    and np.random.random() < self.prob_of_background_object:
                background_object = Random(size=input_size, occupancy_prob=np.random.gamma(1, 0.05)+0.1)
                input_canvas.create_background_from_object(background_object)
            self.input_canvases.append(input_canvas)

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
        transform_index = 3

        obj_to_mirror = copy(obj)

        for i in range(number_of_mirrors):
            mirror_name = Transformations(transform_index)
            args = mirror_name.get_random_parameters()
            mirror_method = getattr(obj_to_mirror, mirror_name.name)
            mirror_method(**args)

        if np.any(obj_to_mirror.dimensions.to_numpy() > MAX_PAD_SIZE):
            obj_to_mirror = self.do_multiple_mirroring(obj, number_of_mirrors=number_of_mirrors - 1)

        return obj_to_mirror

    def randomly_position_object_in_canvas(self, obj: Primitive, canvas: Canvas) -> bool:
        """
        It takes an object and places it in random (but allowed) positions in a Canvas of the Example
        :param canvas: The Canvas to place the object in.
        :param obj: The object to position.
        :return: True if successful. False otherwise.
        """
        in_canvas_pos = self.get_random_position(obj, canvas)
        if in_canvas_pos is not None:
            o = copy(obj)
            o.canvas_pos = in_canvas_pos
            canvas.add_new_object(o)
            self.objects.append(o)
            return True

        return False

    def place_new_object_on_canvas(self, canvas: Canvas):
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

        success = False

        if self.experiment_type == 'Object':

            self.temp_objects = [self.create_object(debug=False, max_size_of_obj=canvas.size,
                                                    overlap_prob=OVERLAP_PROB, far_away_prob=FAR_AWAY_PROB)]

            num_of_transformed_copies = np.random.randint(1, MAX_NUM_OF_LARGE_OBJECTS) \
                if np.any(self.temp_objects[-1].size.to_numpy() > LARGE_OBJECT_THRESHOLD) else \
                np.random.randint(MIN_NUM_OF_SMALL_OBJECTS, MAX_NUM_OF_SMALL_OBJECTS)

            for _ in range(num_of_transformed_copies):
                self.temp_objects.append(copy(self.temp_objects[-1]))

            num_of_transformations = NUMBER_OF_TRANSFORMATIONS
            probs_of_transformations = [0.1, 0.2, 0.05, 0, 0.05, 0.3, 0.3]  # No mirroring
            for k, obj in enumerate(self.temp_objects):
                if k > 0:  # Leave one object copy untransformed
                    try:
                        self.do_random_transformations(obj, num_of_transformations=num_of_transformations, debug=False,
                                                       probs_of_transformations=probs_of_transformations)
                    except:
                        pass

                for _ in range(MAX_NUM_OF_SAME_OBJECTS_ON_CANVAS):
                    success = self.randomly_position_object_in_canvas(obj, canvas)

        elif self.experiment_type == 'Symmetry':

            # Make sure the base object of a symmetry object is not a Parallelogram
            base_object = Parallelogram(size=[2, 2])
            while base_object.get_str_type() == 'Parallelogram':
                base_object = self.create_object(debug=False)

            obj = self.do_multiple_mirroring(base_object)
            if obj is not None:
                success = self.randomly_position_object_in_canvas(obj, canvas)

        if (not success) and (len(canvas.objects) == 0):
            self.place_new_object_on_canvas(canvas=canvas)

    def randomly_populate_canvases(self):
        """
        This is the main function to call to generate the Random Experiment.
        It generates a random number of objects (if experiment_type is 'Object') or just one object (if type is
        'Symmetry') and then places their transformations on all the canvases (in allowed positions)
        :return:
        """

        for canvas in self.input_canvases:
            num_of_objects = 1
            if self.experiment_type == 'Object':
                num_of_objects = np.random.randint(1, MAX_NUM_OF_DIFFERENT_PRIMITIVES)
            for _ in range(num_of_objects):
                self.place_new_object_on_canvas(canvas=canvas)

    def get_canvases_as_numpy_array(self):
        result = np.zeros((self.number_of_canvasses,
                           self.input_canvases[0].full_canvas.shape[0],
                           self.input_canvases[0].full_canvas.shape[1]))

        for i, c in enumerate(self.input_canvases):
            result[i, :, :] = c.full_canvas

        return result

    def show(self, canvas_index: int | str = 'all', save_as: str | None = None):
        """
        Shows some (canvas_index is int or 'test') or all (canvas_index = 'all') the Canvases of the Experiment
        :param save_as:  If not None then save the figure generated as save_as file (but do not show it).
        :param canvas_index: Which Canvases to show (int, 'test' or 'all')
        :return:
        """
        if type(canvas_index) == int:
            canvas = self.input_canvases[canvas_index]
            canvas.show(save_as=save_as)

        elif canvas_index == 'all':
            fig = plt.figure()
            nrows = self.number_of_canvasses // 4
            ncoloumns = 4
            for p in range(self.number_of_canvasses):
                self.input_canvases[p].show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=p + 1)
            plt.tight_layout()

        plt.show()


