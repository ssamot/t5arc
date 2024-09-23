
from data.generators.task_generator.arc_task_generator import ARCTask
from data.generators.object_recognition.basic_geometry import Dimension2D, Point
from dsls.our_dsl.functions import task_solving_utils as utils
from dsls.our_dsl.solutions import solutions as sols

task = ARCTask('05f2a901')
task.generate_canvasses()

unique_objects = [
    {'primitive': 'Parallelogram', 'colour': 9, 'id': 0, 'actual_pixels_id': 0, 'dimensions': Dimension2D(2, 2),
     'canvas_and_position': [0, Point(3, 2, 0)],
     'actual_pixels': task.get_object_pixels_from_data(0, Point(3, 2, 0), Dimension2D(2, 2)),
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Parallelogram', 'colour': 9, 'id': 0, 'actual_pixels_id': 0, 'dimensions': Dimension2D(2, 2),
     'canvas_and_position': [2, Point(6, 3, 0)],
     'actual_pixels': task.get_object_pixels_from_data(0, Point(3, 2, 0), Dimension2D(2, 2)),
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Parallelogram', 'colour': 9, 'id': 0, 'actual_pixels_id': 0, 'dimensions': Dimension2D(2, 2),
     'canvas_and_position': [4, Point(3, 8, 0)],
     'actual_pixels': task.get_object_pixels_from_data(0, Point(3, 2, 0), Dimension2D(2, 2)),
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Parallelogram', 'colour': 9, 'id': 0, 'actual_pixels_id': 0, 'dimensions': Dimension2D(2, 2),
     'canvas_and_position': [6, Point(1, 3, 0)],
     'actual_pixels': task.get_object_pixels_from_data(0, Point(3, 2, 0), Dimension2D(2, 2)),
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 3, 'id': 1, 'actual_pixels_id': 1, 'dimensions': Dimension2D(4, 3),
     'canvas_and_position': [0, Point(0, 10, 0)],
     'actual_pixels': task.get_object_pixels_from_data(0, Point(0, 10, 0), Dimension2D(4, 3)),
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 3, 'id': 2, 'actual_pixels_id': 2, 'dimensions': Dimension2D(3, 4),
     'canvas_and_position': [2, Point(0, 4, 0)],
     'actual_pixels': task.get_object_pixels_from_data(2, Point(0, 4, 0), Dimension2D(3, 4)),
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 3, 'id': 3, 'actual_pixels_id': 3, 'dimensions': Dimension2D(5, 3),
     'canvas_and_position': [4, Point(1, 2, 0)],
     'actual_pixels': task.get_object_pixels_from_data(4, Point(1, 2, 0), Dimension2D(5, 3)),
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 3, 'id': 3, 'actual_pixels_id': 3, 'dimensions': Dimension2D(2, 4),
     'canvas_and_position': [6, Point(5, 3, 0)],
     'actual_pixels': task.get_object_pixels_from_data(6, Point(5, 3, 0), Dimension2D(2, 4)),
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    ]

task.generate_objects_from_output(unique_objects=unique_objects)
task = utils.solve_canvas_pairs(task=task, solution=sols.solution_05f2a901, which_pair='all')


task.show()

