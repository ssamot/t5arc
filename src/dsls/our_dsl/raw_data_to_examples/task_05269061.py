
from copy import copy

from data.generators.task_generator.arc_task_generator import ARCTask
from data.generators.object_recognition.basic_geometry import Dimension2D, Point
from dsls.our_dsl.functions import dsl_functions as dsl

task = ARCTask('05269061')

# [1, Point(0, 6, 10)], [1, Point(6, 0, 10)],
#                             [3, Point(0, 6, 10)], [3, Point(6, 0, 10)]

unique_objects = [
    {'primitive': 'Diagonal', 'colour': 3, 'id': 0, 'actual_pixels_id': 0, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [0, Point(0, 6, 10)],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Diagonal', 'colour': 3, 'id': 0, 'actual_pixels_id': 0, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [1, Point(0, 6, 10)],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Diagonal', 'colour': 3, 'id': 0, 'actual_pixels_id': 0, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [1, Point(6, 0, 10)],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Diagonal', 'colour': 3, 'id': 0, 'actual_pixels_id': 0, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [3, Point(0, 6, 10)],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Diagonal', 'colour': 3, 'id': 0, 'actual_pixels_id': 0, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [3, Point(6, 0, 10)],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Diagonal', 'colour': 9, 'id': 1, 'actual_pixels_id': 1, 'dimensions': Dimension2D(2, 2),
     'canvas_and_position': [0, Point(0, 5, 9)],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Diagonal', 'colour': 9, 'id': 1, 'actual_pixels_id': 1, 'dimensions': Dimension2D(2, 2),
     'canvas_and_position': [1, Point(0, 5, 9)],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Diagonal', 'colour': 4, 'id': 2, 'actual_pixels_id': 2, 'dimensions': Dimension2D(3, 3),
     'canvas_and_position': [0, Point(0, 4, 8)],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Diagonal', 'colour': 4, 'id': 2, 'actual_pixels_id': 2, 'dimensions': Dimension2D(3, 3),
     'canvas_and_position':  [1, Point(0, 4, 8)],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Diagonal', 'colour': 4, 'id': 2, 'actual_pixels_id': 2, 'dimensions': Dimension2D(3, 3),
     'canvas_and_position':  [6, Point(0, 4, 8)],
     'transformations': [],
     'symmetries': []},
]

task.generate_objects_from_output(unique_objects=unique_objects)
