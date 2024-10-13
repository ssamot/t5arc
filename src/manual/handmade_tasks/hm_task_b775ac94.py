
import numpy as np
from copy import copy
from data.generators.object_recognition.object import Object
from data.generators.task_generator.arc_task_generator import ARCTask
from data.generators.object_recognition.basic_geometry import Dimension2D, Point
from manual import manual_object_generator_functions as man_funcs

task = ARCTask('b775ac94')
task.generate_canvasses()

'''
unique_objects = [
    {'primitive': 'Random', 'colour': 2, 'id': 0, 'actual_pixels_id': 0, 'dimensions': Dimension2D(5, 5),
     'canvas_and_position': [0, Point(3, 14, 0)],
     'actual_pixels': task.get_object_pixels_from_data(0, Point(3, 14, 0), Dimension2D(5, 5)),
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 2, 'id': 1, 'actual_pixels_id': 1, 'dimensions': Dimension2D(4, 4),
     'canvas_and_position': [0, Point(14, 17, 0)],
     'actual_pixels': task.get_object_pixels_from_data(0, Point(14, 17, 0), Dimension2D(4, 4)),
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 2, 'id': 2, 'actual_pixels_id': 2, 'dimensions': Dimension2D(4, 4),
     'canvas_and_position': [0, Point(13, 5, 0)],
     'actual_pixels': task.get_object_pixels_from_data(0, Point(13, 5, 0), Dimension2D(4, 4)),
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 2, 'id': 3, 'actual_pixels_id': 3, 'dimensions': Dimension2D(6, 4),
     'canvas_and_position': [2, Point(3, 10, 0)],
     'actual_pixels': task.get_object_pixels_from_data(2, Point(3, 10, 0), Dimension2D(6, 4)),
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 2, 'id': 4, 'actual_pixels_id': 4, 'dimensions': Dimension2D(5, 4),
     'canvas_and_position': [4, Point(1, 3, 0)],
     'actual_pixels': task.get_object_pixels_from_data(4, Point(1, 3, 0), Dimension2D(5, 4)),
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 2, 'id': 5, 'actual_pixels_id': 5, 'dimensions': Dimension2D(4, 4),
     'canvas_and_position': [4, Point(7, 9, 0)],
     'actual_pixels': task.get_object_pixels_from_data(4, Point(7, 9, 0), Dimension2D(4, 4)),
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 2, 'id': 6, 'actual_pixels_id': 6, 'dimensions': Dimension2D(5, 5),
     'canvas_and_position': [6, Point(4, 16, 0)],
     'actual_pixels': task.get_object_pixels_from_data(6, Point(4, 16, 0), Dimension2D(5, 5)),
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 2, 'id': 7, 'actual_pixels_id': 7, 'dimensions': Dimension2D(4, 4),
     'canvas_and_position': [6, Point(11, 4, 0)],
     'actual_pixels': task.get_object_pixels_from_data(6, Point(11, 4, 0), Dimension2D(4, 4)),
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 2, 'id': 8, 'actual_pixels_id': 8, 'dimensions': Dimension2D(4, 3),
     'canvas_and_position': [6, Point(15, 15, 0)],
     'actual_pixels': task.get_object_pixels_from_data(6, Point(15, 15, 0), Dimension2D(4, 3)),
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
]

task.generate_objects_from_json_description(unique_objects=unique_objects)
'''

task.generate_objects_from_data(man_funcs.same_colour)


'''
for i, (in_c, out_c) in enumerate(zip(task.input_canvases, task.output_canvases)):
    data = np.flipud(task.task_data['train'][i]['output']) + 1
    full_canvas_object = Object(actual_pixels=data)
    out_c.add_new_object(full_canvas_object)

    num_of_in_c_objects = len(in_c.objects)
    for obj in in_c.objects[:num_of_in_c_objects]:
        in_c.split_object_by_colour(obj)


obj = task.input_canvases[0].objects[4]
a = obj.match(task.output_canvases[0].objects[0], match_shape_only=True, try_unique=True)
'''