
from data_generators.example_generator.arc_example_generator import ARCExample
from data_generators.object_recognition.basic_geometry import Dimension2D, Point
from dsls.our_dsl.solutions import solutions as sols

example = ARCExample('97a05b5b')

unique_objects = [
    {'primitive': 'Random', 'colour': 1, 'id': 0, 'actual_pixels_id': 0, 'dimensions': Dimension2D(9, 17),
     'canvases_positions': [[0, Point(2, 6, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(0, Point(2, 6, 0), Dimension2D(9, 17)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 1, 'id': 1, 'actual_pixels_id': 1, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[0, Point(3, 1, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(0, Point(3, 1, 0), Dimension2D(3, 3)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 1, 'id': 2, 'actual_pixels_id': 2, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[0, Point(8, 1, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(0, Point(8, 1, 0), Dimension2D(3, 3)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 1, 'id': 3, 'actual_pixels_id': 3, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[0, Point(14, 4, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(0, Point(14, 4, 0), Dimension2D(3, 3)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 1, 'id': 4, 'actual_pixels_id': 4, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[0, Point(14, 14, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(0, Point(14, 14, 0), Dimension2D(3, 3)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 1, 'id': 5, 'actual_pixels_id': 5, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[0, Point(13, 19, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(0, Point(13, 19, 0), Dimension2D(3, 3)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 1, 'id': 6, 'actual_pixels_id': 6, 'dimensions': Dimension2D(8, 8),
     'canvases_positions': [[2, Point(1, 11, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(2, Point(1, 11, 0), Dimension2D(8, 8)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 1, 'id': 7, 'actual_pixels_id': 7, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[2, Point(1, 6, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(2, Point(1, 6, 0), Dimension2D(3, 3)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 1, 'id': 8, 'actual_pixels_id': 8, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[2, Point(5, 4, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(2, Point(5, 4, 0), Dimension2D(3, 3)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 1, 'id': 9, 'actual_pixels_id': 9, 'dimensions': Dimension2D(9, 9),
     'canvases_positions': [[4, Point(1, 1, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(4, Point(1, 1, 0), Dimension2D(9, 9)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 1, 'id': 10, 'actual_pixels_id': 10, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[4, Point(5, 12, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(4, Point(5, 12, 0), Dimension2D(3, 3)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 3, 'id': 11, 'actual_pixels_id': 11, 'dimensions': Dimension2D(12, 8),
     'canvases_positions': [[6, Point(2, 10, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(6, Point(2, 10, 0), Dimension2D(12, 8)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 1, 'id': 12, 'actual_pixels_id': 12, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[6, Point(1, 5, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(6, Point(1, 5, 0), Dimension2D(3, 3)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 1, 'id': 13, 'actual_pixels_id': 13, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[6, Point(3, 0, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(6, Point(3, 0, 0), Dimension2D(3, 3)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 1, 'id': 12, 'actual_pixels_id': 12, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[6, Point(8, 5, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(6, Point(8, 5, 0), Dimension2D(3, 3)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 1, 'id': 12, 'actual_pixels_id': 12, 'dimensions': Dimension2D(3, 3),
     'canvases_positions': [[6, Point(11, 1, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(6, Point(11, 1, 0), Dimension2D(3, 3)),
     'transformations': [],
     'symmetries': []},
]


example.generate_objects_from_output(unique_objects=unique_objects)
example.reset_object_colours()


#canvas_in = example.input_canvases[2]
canvas_in = example.test_input_canvas


canvas_out = sols.solution_97a05b5b(canvas_in)

canvas_out.show()