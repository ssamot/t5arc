
import numpy as np
from copy import copy
from data.generators.object_recognition.object import Object
from data.generators.object_recognition.primitives import Predefined
from data.generators.task_generator.arc_task_generator import ARCTask
from data.generators.object_recognition.basic_geometry import Dimension2D, Point
from manual import manual_object_generator_functions as man_funcs

task = ARCTask('b775ac94')
#task = ARCTask('00d62c1b')
task.generate_canvasses()

task.generate_objects_from_data(man_funcs.same_colour_connected_pixels)

task.show()


in_canvas = task.input_canvases[0]
out_canvas = task.output_canvases[0]

for obj in in_canvas.objects:
    full_canvas_out_pixels = task.get_object_pixels_from_data(out_canvas.id, canvas_pos=Point(0, 0, 0), size=None)
    full_canvas_out_object = Predefined(actual_pixels=full_canvas_out_pixels)

    matchings = obj.match(full_canvas_out_object)
