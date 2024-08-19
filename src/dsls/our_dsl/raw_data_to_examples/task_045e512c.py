
from copy import copy

from data_generators.example_generator.arc_example_generator import ARCExample
from data_generators.object_recognition.basic_geometry import Dimension2D, Point
from dsls.our_dsl.functions import dsl_functions as dsl

example = ARCExample('045e512c')

unique_objects = [
    {'primitive': 'Hole', 'colour': 9, 'id': 0, 'actual_pixels_id': 0,'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[0, Point(6, 12, 0)], [1, Point(6, 12, 0)]],
     'transformations': [],
     'symmetries': [],
     'hole_bbox':[[1, 1], [1, 11]],
     'thickness': [1, 1, 1, 1]},
    {'primitive': 'Hole', 'colour': 3, 'id': 0, 'actual_pixels_id': 1, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[1, Point(6, 8, 0)], [1, Point(6, 4, 0)], [1, Point(6, 0, 0)]],
     'transformations': [],
     'symmetries': [],
     'hole_bbox':[[1, 1], [1, 11]],
     'thickness': [1, 1, 1, 1]},
    {'primitive': 'Hole', 'colour': 4, 'id': 0, 'actual_pixels_id': 2, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[1, Point(10, 12, 0)], [1, Point(14, 12, 0)], [1, Point(18, 12, 0)]],
     'transformations': [],
     'symmetries': [],
     'hole_bbox':[[1, 1], [1, 11]],
     'thickness': [1, 1, 1, 1]},
    {'primitive': 'Parallelogram', 'colour': 3, 'id': 1, 'actual_pixels_id': 3, 'dimensions': Dimension2D(3, 1),
     'canvases_positions': [[0, Point(6, 10, 0)]],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Parallelogram', 'colour': 4, 'id': 1, 'actual_pixels_id': 4, 'dimensions': Dimension2D(1, 3),
     'canvases_positions': [[0, Point(10, 12, 0)]],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Parallelogram', 'colour': 3, 'id': 1, 'actual_pixels_id': 5, 'dimensions': Dimension2D(3, 1),
     'canvases_positions': [[6, Point(6, 9, 0)]],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Parallelogram', 'colour': 3, 'id': 1, 'actual_pixels_id': 6, 'dimensions': Dimension2D(1, 3),
     'canvases_positions': [[6, Point(10, 11, 0)]],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Cross', 'colour': 2, 'id': 2, 'actual_pixels_id': 7, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[2, Point(11, 11, 0)], [3, Point(11, 11, 0)]],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Cross', 'colour': 3, 'id': 2, 'actual_pixels_id': 8, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[3, Point(7, 11, 0)], [3, Point(3, 11, 0)], [3, Point(-1, 11, 0)]],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Cross', 'colour': 5, 'id': 2, 'actual_pixels_id': 9, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[3, Point(15, 11, 0)], [3, Point(19, 11, 0)], [3, Point(11, 15, 0)], [3, Point(11, 19, 0)]],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 3, 'id': 3, 'actual_pixels_id': 10, 'dimensions': Dimension2D(1, 1),
     'canvases_positions': [[2, Point(9, 12, 0)]],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 5, 'id': 3, 'actual_pixels_id': 11, 'dimensions': Dimension2D(1, 1),
     'canvases_positions': [[2, Point(12, 15, 0)], [2, Point(15, 12, 0)]],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Bolt', 'colour': 6, 'id': 4, 'actual_pixels_id': 12, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[4, Point(6, 11, 0)], [5, Point(6, 11, 0)]], 'center_on': False,
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Bolt', 'colour': 2, 'id': 4, 'actual_pixels_id': 13, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[5, Point(10, 7, 0)], [5, Point(14, 3, 0)], [5, Point(18, -1, 0)]], 'center_on': False,
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Bolt', 'colour': 10, 'id': 4, 'actual_pixels_id': 14, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[5, Point(10, 15, 0)], [5, Point(14, 19, 0)]], 'center_on': False,
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Diagonal', 'colour': 7, 'id': 5, 'actual_pixels_id': 15, 'dimensions': Dimension2D(2, 2),
     'canvases_positions': [[4, Point(11, 15, 0)]],
     'transformations': [[1, 1]],
     'symmetries': []},
    {'primitive': 'Angle', 'colour': 2, 'id': 6, 'actual_pixels_id': 16, 'dimensions': Dimension2D(2, 2),
     'canvases_positions': [[4, Point(10, 9, 0)]],
     'transformations': [[1, 3]],
     'symmetries': []},
    {'primitive': 'Pi', 'colour': 9, 'id': 7, 'actual_pixels_id': 17, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[6, Point(6, 11, 0)]],
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Parallelogram', 'colour': 5, 'id': 8, 'actual_pixels_id': 18, 'dimensions': Dimension2D(1, 2),
     'canvases_positions': [[6, Point(10, 15, 0)]],
     'transformations': [],
     'symmetries': []},
]

example.generate_objects_from_output(unique_objects=unique_objects)

# example.show()

# Solution

canvas = example.input_canvases[1]
canvas = example.test_input_canvas

largest_object = canvas.sort_objects_by_size(used_dim='area')[-1]
other_objects = canvas.sort_objects_by_size(used_dim='area')[:-1]
for oo in other_objects:
    match_positions = largest_object.match(oo, match_shape_only=True)
    eucl_dist = dsl.furthest(largest_object.canvas_pos, match_positions)

    so = copy(largest_object)
    for _ in range(3):
        no = copy(so)
        no.set_new_colour(oo.colour)
        no.canvas_pos.transform(translation=eucl_dist)
        canvas.add_new_object(no)
        so = no

