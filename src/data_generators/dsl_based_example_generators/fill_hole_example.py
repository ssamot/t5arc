

from data_generators.example_generator.example import *


class FillHoleExample(Example):
    def __init__(self):
        super().__init__()

        self.experiment_type = 'Object'

    def create_random_object_with_holes(self):

        obj_type = ObjectType.random(_probabilities=np.array([0.4, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0, 0.1, 0.1]))

        id = self.ids[-1] + 1 if len(self.ids) > 0 else 0
        actual_pixels_id = self.actual_pixel_ids[-1] + 1 if len(self.actual_pixel_ids) > 0 else 0

        args = {'colour': self.get_random_colour(),
                'border_size': Surround(0, 0, 0, 0),  # For now and then we will see
                'canvas_pos': Point(0, 0, 0),
                '_id': id,
                'actual_pixels_id': actual_pixels_id}
        self.ids.append(id)
        self.actual_pixel_ids.append(actual_pixels_id)

        