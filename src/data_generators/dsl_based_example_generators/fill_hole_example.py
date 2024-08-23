

import numpy as np

from data_generators.example_generator.example import Example
from data_generators.object_recognition.primitives import Primitive

class FillHoleExample(Example):
    def __init__(self):
        super().__init__()

        self.experiment_type = 'Object'

    def create_random_object_with_holes(self) -> Primitive | None:

        obj_probs = np.array([0.4, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0, 0.1, 0.1])
        transformations_probs = [1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7]

        def new_obj():
            obj = self.create_object(obj_probs=obj_probs, max_size_of_obj=Dimension2D(20,20))
            self.do_random_transformations(obj=obj, num_of_transformations=1,
                                           probs_of_transformations=transformations_probs)
            _, n_holes = obj.detect_holes()
            if n_holes == 0:
                obj.create_random_hole(hole_size=np.random.randint(2, 10))

            if n_holes > 0:
                return obj
            return None

        obj = new_obj()
        tries = 0
        while obj is None and tries < 5:
            obj = new_obj()
            tries += 1

        return obj


        