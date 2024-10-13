
from __future__ import annotations
import numpy as np
from copy import copy

from data.generators.task_generator.task import Task
from data.generators.object_recognition.basic_geometry import Dimension2D
from data.generators.object_recognition.primitives import Primitive, Parallelogram
from data.generators import constants as const
from data.generators.object_recognition.object import Transformations

MAX_EXAMPLE_PAIRS = const.MAX_EXAMPLE_PAIRS
MIN_PAD_SIZE = const.MIN_PAD_SIZE
MAX_PAD_SIZE = const.MAX_PAD_SIZE

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

OVERLAP_PROB = 0.8
FAR_AWAY_PROB = 0.1
MAX_SIZE_OF_OBJECT = 15

MAX_NUMBER_OF_MIRRORS = 10


class RandomObjectsTask(Task):
    def __init__(self, number_of_io_pairs: int | None = None):
        super().__init__(min_canvas_size_for_background_object=MIN_CANVAS_SIZE_FOR_BACKGROUND_OBJ,
                         prob_of_background_object=PROB_OF_BACKGROUND_OBJ, number_of_io_pairs=number_of_io_pairs)

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

            self.temp_objects = [self.create_random_object(debug=False,
                                                           max_size_of_obj=Dimension2D(MAX_SIZE_OF_OBJECT, MAX_SIZE_OF_OBJECT),
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
            if obj is not None:
                self.randomly_position_object_in_all_canvases(obj)

    def randomly_populate_canvases(self):
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

