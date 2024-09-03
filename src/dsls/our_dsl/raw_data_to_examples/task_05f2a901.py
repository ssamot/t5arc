
from data.generators.example_generator.arc_example_generator import ARCExample
from data.generators.object_recognition.basic_geometry import Dimension2D, Point
from dsls.our_dsl.functions import task_solving_utils as utils
from dsls.our_dsl.solutions import solutions as sols

example = ARCExample('05f2a901')
example.generate_canvasses()

unique_objects = [
    {'primitive': 'Parallelogram', 'colour': 9, 'id': 0, 'actual_pixels_id': 0, 'dimensions': Dimension2D(2, 2),
     'canvases_positions': [[0, Point(3, 2, 0)], [2, Point(6, 3, 0)], [4, Point(3, 8, 0)], [6, Point(1, 3, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(0, Point(3, 2, 0), Dimension2D(2, 2)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 3, 'id': 1, 'actual_pixels_id': 1, 'dimensions': Dimension2D(4, 3),
     'canvases_positions': [[0, Point(0, 10, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(0, Point(0, 10, 0), Dimension2D(4, 3)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 3, 'id': 2, 'actual_pixels_id': 2, 'dimensions': Dimension2D(3, 4),
     'canvases_positions': [[2, Point(0, 4, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(2, Point(0, 4, 0), Dimension2D(3, 4)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 3, 'id': 3, 'actual_pixels_id': 3, 'dimensions': Dimension2D(5, 3),
     'canvases_positions': [[4, Point(1, 2, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(4, Point(1, 2, 0), Dimension2D(5, 3)),
     'transformations': [],
     'symmetries': []},
{'primitive': 'Random', 'colour': 3, 'id': 3, 'actual_pixels_id': 3, 'dimensions': Dimension2D(2, 4),
     'canvases_positions': [[6, Point(5, 3, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(6, Point(5, 3, 0), Dimension2D(2, 4)),
     'transformations': [],
     'symmetries': []},
    ]

example.generate_objects_from_output(unique_objects=unique_objects)
example = utils.solve_canvas_pairs(example=example, solution=sols.solution_05f2a901, which_pair='all')


#example.show()

