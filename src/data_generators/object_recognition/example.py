

from __future__ import annotations

from visualization import visualize_data as vis
from data_generators.object_recognition.primitives import *
from data_generators.object_recognition.basic_geometry import Point, Vector, Bbox, Orientation, Dimension2D, OrientationZ

np.random.seed(const.RANDOM_SEED_FOR_NUMPY)
MAX_EXAMPLE_PAIRS = const.MAX_EXAMPLE_PAIRS
MIN_PAD_SIZE = const.MIN_PAD_SIZE
MAX_PAD_SIZE = const.MAX_PAD_SIZE


class Example:
    def __init__(self):
        self.number_of_canvasses = np.random.randint(4, const.MAX_EXAMPLE_PAIRS + 2, 2)
        self.input_size = Dimension2D(np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE),
                                      np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE))
        self.output_size = Dimension2D(np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE),
                                       np.random.randint(MIN_PAD_SIZE, MAX_PAD_SIZE))
        print(self.input_size, self.output_size)

    def generate_canvasses(self):
        pass