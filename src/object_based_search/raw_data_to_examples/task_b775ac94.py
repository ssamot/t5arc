
from data.generators.example_generator.arc_example_generator import ARCExample
from data.generators.object_recognition.basic_geometry import Dimension2D, Point
from dsls.our_dsl.functions import task_solving_utils as utils
from dsls.our_dsl.solutions import solutions as sols


example = ARCExample('b775ac94')
example.generate_canvasses()

unique_objects = [
    {'primitive': 'Random', 'colour': 2, 'id': 0, 'actual_pixels_id': 0, 'dimensions': Dimension2D(5, 5),
     'canvas_and_position': [0, Point(3, 14, 0)],
     'actual_pixels': example.get_object_pixels_from_data(0, Point(3, 14, 0), Dimension2D(4, 4)),
     'on_canvas_transformations': [],
     'in_out_transformations': [
                                [['translate_by', [0, 0]]],
                                [['flip', 'Right'], ['translate_by', [4, 0]], ['replace_colour', [2, 3]]],
                                [['flip', 'Up'], ['flip', 'Right'], ['translate_by', [4, 4]], ['replace_colour', [2, 5]]],
                                [['flip', 'Up'], ['translate_by', [0, 4]], ['replace_colour', [2, 4]]]
                               ],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 4, 'id': 1, 'actual_pixels_id': 1, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [0, Point(6, 18, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 5, 'id': 2, 'actual_pixels_id': 2, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [0, Point(7, 18, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 3, 'id': 3, 'actual_pixels_id': 3, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [0, Point(7, 17, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},

    {'primitive': 'Random', 'colour': 8, 'id': 4, 'actual_pixels_id': 4, 'dimensions': Dimension2D(3, 3),
     'canvas_and_position': [0, Point(14, 17, 0)],
     'actual_pixels': example.get_object_pixels_from_data(0, Point(14, 17, 0), Dimension2D(3, 3)),
     'on_canvas_transformations': [],
     'in_out_transformations': [
                                [['translate_by', [0, 0]]],
                                [['flip', 'Right'], ['translate_by', [3, 0]], ['replace_colour', [8, 5]]],
                                [['flip', 'Up'], ['flip', 'Right'], ['translate_by', [3, 3]], ['replace_colour', [8, 2]]],
                                [['flip', 'Up'], ['translate_by', [0, 3]], ['replace_colour', [8, 3]]]
                               ],
     'symmetries': []},
   {'primitive': 'Dot', 'colour': 3, 'id': 5, 'actual_pixels_id': 5, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [0, Point(16, 20, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
   {'primitive': 'Dot', 'colour': 2, 'id': 6, 'actual_pixels_id': 6, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [0, Point(17, 20, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
   {'primitive': 'Dot', 'colour': 5, 'id': 6, 'actual_pixels_id': 6, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [0, Point(17, 19, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},

    {'primitive': 'Random', 'colour': 4, 'id': 7, 'actual_pixels_id': 7, 'dimensions': Dimension2D(3, 3),
     'canvas_and_position': [0, Point(14, 5, 0)],
     'actual_pixels': example.get_object_pixels_from_data(0, Point(14, 5, 0), Dimension2D(3, 3)),
     'on_canvas_transformations': [],
     'in_out_transformations': [
                                [['translate_by', [0, 0]]],
                                [['flip', 'Left'], ['translate_by', [-3, 0]], ['replace_colour', [4, 3]]],
                                [['flip', 'Up'], ['flip', 'Left'], ['translate_by', [-3, 3]], ['replace_colour', [4, 9]]]
                               ],
     'symmetries': []},
   {'primitive': 'Dot', 'colour': 3, 'id': 8, 'actual_pixels_id': 8, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [0, Point(13, 7, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
   {'primitive': 'Dot', 'colour': 9, 'id': 9, 'actual_pixels_id': 9, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [0, Point(13, 8, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},

    {'primitive': 'Random', 'colour': 3, 'id': 10, 'actual_pixels_id': 10, 'dimensions': Dimension2D(5, 3),
     'canvas_and_position': [2, Point(3, 11, 0)],
     'actual_pixels': example.get_object_pixels_from_data(2, Point(3, 11, 0), Dimension2D(5, 3)),
     'on_canvas_transformations': [],
     'in_out_transformations': [
                                [['translate_by', [0, 0]]],
                                [['flip', 'Right'], ['translate_by', [5, 0]], ['replace_colour', [3, 9]]],
                                [['flip', 'Down'], ['flip', 'Right'], ['translate_by', [5, -3]], ['replace_colour', [3, 4]]],
                                [['flip', 'Down'], ['translate_by', [0, -3]], ['replace_colour', [3, 5]]]
                               ],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 5, 'id': 11, 'actual_pixels_id': 11, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [2, Point(7, 10, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 4, 'id': 12, 'actual_pixels_id': 12, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [2, Point(8, 10, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 9, 'id': 13, 'actual_pixels_id': 13, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [2, Point(8, 11, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},

    {'primitive': 'Random', 'colour': 3, 'id': 14, 'actual_pixels_id': 14, 'dimensions': Dimension2D(4, 3),
     'canvas_and_position': [4, Point(1, 3, 0)],
     'actual_pixels': example.get_object_pixels_from_data(4, Point(1, 3, 0), Dimension2D(4, 3)),
     'on_canvas_transformations': [],
     'in_out_transformations': [
                                [['translate_by', [0, 0]]],
                                [['flip', 'Right'], ['translate_by', [4, 0]], ['replace_colour', [3, 4]]],
                                [['flip', 'Up'], ['flip', 'Right'], ['translate_by', [4, 3]], ['replace_colour', [3, 2]]]
                               ],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 5, 'id': 15, 'actual_pixels_id': 15, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [4, Point(5, 5, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 2, 'id': 16, 'actual_pixels_id': 16, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [4, Point(5, 6, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},

    {'primitive': 'Tie', 'colour': 9, 'id': 17, 'actual_pixels_id': 17, 'dimensions': Dimension2D(3, 3),
     'canvas_and_position': [4, Point(9, 12, 0)],
     'on_canvas_transformations': [['rotate', 2]],
     'in_out_transformations': [
                                [['translate_by', [0, 0]]],
                                [['flip', 'Right'], ['translate_by', [3, 0]], ['replace_colour', [9, 5]]],
                                [['flip', 'Down'], ['flip', 'Right'], ['translate_by', [3, -3]], ['replace_colour', [9, 7]]]
                               ],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 7, 'id': 18, 'actual_pixels_id': 18, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [4, Point(10, 9, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 5, 'id': 19, 'actual_pixels_id': 19, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [4, Point(10, 10, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},

    {'primitive': 'Random', 'colour': 9, 'id': 20, 'actual_pixels_id': 20, 'dimensions': Dimension2D(4, 4),
     'canvas_and_position': [6, Point(4, 17, 0)],
     'actual_pixels': example.get_object_pixels_from_data(6, Point(4, 17, 0), Dimension2D(4, 4)),
     'on_canvas_transformations': [],
     'in_out_transformations': [
                                [['translate_by', [0, 0]]],
                                [['flip', 'Right'], ['translate_by', [4, 0]], ['replace_colour', [9, 3]]],
                                [['flip', 'Down'], ['flip', 'Right'], ['translate_by', [4, -4]], ['replace_colour', [9, 4]]],
                                [['flip', 'Down'], ['translate_by', [0, -4]], ['replace_colour', [9, 5]]]
                               ],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 5, 'id': 21, 'actual_pixels_id': 21, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [6, Point(7, 16, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 4, 'id': 22, 'actual_pixels_id': 22, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [6, Point(8, 16, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 3, 'id': 23, 'actual_pixels_id': 23, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [6, Point(8, 17, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},

    {'primitive': 'Random', 'colour': 3, 'id': 24, 'actual_pixels_id': 24, 'dimensions': Dimension2D(3, 3),
     'canvas_and_position': [6, Point(11, 4, 0)],
     'actual_pixels': example.get_object_pixels_from_data(6, Point(11, 4, 0), Dimension2D(3, 3)),
     'on_canvas_transformations': [],
     'in_out_transformations': [
                                [['translate_by', [0, 0]]],
                                [['flip', 'Right'], ['translate_by', [3, 0]], ['replace_colour', [3, 5]]],
                                [['flip', 'Up'], ['translate_by', [0, 3]], ['replace_colour', [3, 2]]]
                               ],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 2, 'id': 25, 'actual_pixels_id': 25, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [6, Point(13, 7, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 5, 'id': 26, 'actual_pixels_id': 26, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [6, Point(14, 6, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},

    {'primitive': 'Random', 'colour': 2, 'id': 27, 'actual_pixels_id': 27, 'dimensions': Dimension2D(3, 2),
     'canvas_and_position': [6, Point(15, 15, 0)],
     'actual_pixels': example.get_object_pixels_from_data(6, Point(15, 15, 0), Dimension2D(3, 2)),
     'on_canvas_transformations': [],
     'in_out_transformations': [
                                [['translate_by', [0, 0]]],
                                [['flip', 'Right'], ['translate_by', [3, 0]], ['replace_colour', [2, 9]]],
                                [['flip', 'Up'], ['flip', 'Right'], ['translate_by', [3, 2]], ['replace_colour', [2, 3]]],
                                [['flip', 'Up'], ['translate_by', [0, 2]], ['replace_colour', [2, 4]]]
                               ],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 4, 'id': 28, 'actual_pixels_id': 28, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [6, Point(17, 17, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 3, 'id': 29, 'actual_pixels_id': 29, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [6, Point(18, 17, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
    {'primitive': 'Dot', 'colour': 9, 'id': 29, 'actual_pixels_id': 29, 'dimensions': Dimension2D(1, 1),
     'canvas_and_position': [6, Point(18, 16, 0)],
     'on_canvas_transformations': [],
     'in_out_transformations': [],
     'symmetries': []},
]

example.generate_objects_from_output(unique_objects=unique_objects)

example.show()








