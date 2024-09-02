
from data.generators.example_generator.arc_example_generator import ARCExample
from data.generators.object_recognition.basic_geometry import Dimension2D, Point
from dsls.our_dsl.functions import task_solving_utils as utils
from dsls.our_dsl.solutions import solutions as sols


example = ARCExample('b775ac94')
example.generate_canvasses()

unique_objects = [
    {'primitive': 'Random', 'colour': 2, 'id': 0, 'actual_pixels_id': 0, 'dimensions': Dimension2D(5, 5),
     'canvases_positions': [[0, Point(3, 14, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(0, Point(3, 14, 0), Dimension2D(5, 5)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 2, 'id': 1, 'actual_pixels_id': 1, 'dimensions': Dimension2D(4, 4),
     'canvases_positions': [[0, Point(14, 17, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(0, Point(14, 17, 0), Dimension2D(4, 4)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 2, 'id': 2, 'actual_pixels_id': 2, 'dimensions': Dimension2D(4, 4),
     'canvases_positions': [[0, Point(13, 5, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(0, Point(13, 5, 0), Dimension2D(4, 4)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 2, 'id': 3, 'actual_pixels_id': 3, 'dimensions': Dimension2D(6, 4),
     'canvases_positions': [[2, Point(3, 10, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(2, Point(3, 10, 0), Dimension2D(6, 4)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 2, 'id': 4, 'actual_pixels_id': 4, 'dimensions': Dimension2D(5, 4),
     'canvases_positions': [[4, Point(1, 3, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(4, Point(1, 3, 0), Dimension2D(5, 4)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 2, 'id': 5, 'actual_pixels_id': 5, 'dimensions': Dimension2D(4, 4),
     'canvases_positions': [[4, Point(7, 9, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(4, Point(7, 9, 0), Dimension2D(4, 4)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 2, 'id': 6, 'actual_pixels_id': 6, 'dimensions': Dimension2D(5, 5),
     'canvases_positions': [[6, Point(4, 16, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(6, Point(4, 16, 0), Dimension2D(5, 5)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 2, 'id': 7, 'actual_pixels_id': 7, 'dimensions': Dimension2D(4, 4),
     'canvases_positions': [[6, Point(11, 4, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(6, Point(11, 4, 0), Dimension2D(4, 4)),
     'transformations': [],
     'symmetries': []},
    {'primitive': 'Random', 'colour': 2, 'id': 8, 'actual_pixels_id': 8, 'dimensions': Dimension2D(4, 3),
     'canvases_positions': [[6, Point(15, 15, 0)]],
     'actual_pixels': example.get_object_pixels_from_data(6, Point(15, 15, 0), Dimension2D(4, 3)),
     'transformations': [],
     'symmetries': []},
]

example.generate_objects_from_output(unique_objects=unique_objects)
example = utils.solve_canvas_pairs(example=example, solution=sols.solution_b775ac94, which_pair='all')

example.show()


#canvas = example.input_canvases[0]
#canvas = example.test_input_canvas
#out_canvas = sols.solution_b775ac94(canvas)
#out_canvas.show()






