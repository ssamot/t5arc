
from copy import copy

from data.generators.example_generator.arc_example_generator import ARCExample
from data.generators.object_recognition.basic_geometry import Dimension2D, Point
from dsls.our_dsl.functions import dsl_functions as dsl

example = ARCExample('05269061')

unique_objects = [
    {'primitive': 'Diagonal', 'colour': 3, 'id': 0, 'actual_pixels_id': 0, 'dimensions': Dimension2D(1, 1),
     'canvases_positions': [[0, Point(0, 6, 10)], [1, Point(0, 6, 10)], [1, Point(6, 0, 10)],
                            [3, Point(0, 6, 10)], [3, Point(6, 0, 10)]],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Diagonal', 'colour': 9, 'id': 1, 'actual_pixels_id': 1, 'dimensions': Dimension2D(2, 2),
     'canvases_positions': [[0, Point(0, 5, 9)], [1, Point(0, 5, 9)]],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Diagonal', 'colour': 4, 'id': 2, 'actual_pixels_id': 2, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[0, Point(0, 4, 8)], [1, Point(0, 4, 8)], [6, Point(0, 4, 8)]],
     'transformations': [],
     'symmetries': []},
]

example.generate_objects_from_output(unique_objects=unique_objects)
