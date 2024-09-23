
from copy import copy
import json

import numpy as np

from data.generators.task_generator.task import Task
from data.generators.object_recognition.basic_geometry import Dimension2D, Point
from data.generators.object_recognition.canvas import Canvas
from data.generators.object_recognition.primitives import Random, Dot

randint = np.random.randint

number_of_pairs = 6

e = Task(number_of_io_pairs=number_of_pairs, prob_of_background_object=0, run_generate_canvasses=False)
e.experiment_type = 'other'
e.generate_canvasses()

for i in range(len(e.input_canvases)):
    e.output_canvases[i] = Canvas(size=e.input_canvases[i].size, _id=e.output_canvases[i].id)

start_points = []
colour_to_trans = {int(2): Point(5, 0), int(4): Point(0, 6), int(6): Point(4, 4)}
transition_colours = [2, 4, 2, 4, 2, 6]
for p in range(number_of_pairs):
    start_points.append(Point(randint(0, 8), randint(0, 10)))
    #transition_colours.append(np.random.choice([2, 4, 6], replace=True).tolist())
end_points = [p + colour_to_trans[transition_colours[i]] for i, p in enumerate(start_points)]

input_objs = []
output_objs = []
for p in range(number_of_pairs):
    random_in = Random(size=Dimension2D(randint(3, 8), randint(3, 8)), _id=3*p, actual_pixels_id=3*p,
                       canvas_pos=start_points[p], canvas_id=2*p + 1)
    random_in.randomise_colour(30)
    dot_in = Dot(colour=transition_colours[p], _id=3 * p + 1, actual_pixels_id=3 * p + 1, canvas_pos=Point(0, 0),
                 canvas_id=2*p + 1)
    input_objs.append(random_in)
    input_objs.append(dot_in)

    random_out = copy(random_in)
    random_out.canvas_id = 2*p + 2
    random_out.translate_to_coordinates(target_point=end_points[p])
    dot_out = Dot(colour=transition_colours[p], _id=3 * p + 2, actual_pixels_id=3 * p + 2, canvas_pos=Point(0, 0),
                  canvas_id=2*p + 2)
    output_objs.append(random_out)
    output_objs.append(dot_out)


for o1, o2 in zip(input_objs, output_objs):
    e.add_object_on_canvasses(o1, [o1.canvas_id])
    e.add_object_on_canvasses(o2, [o2.canvas_id])


dot_colour = 2
test_in_obj = Random(size=Dimension2D(7, 4), canvas_pos=Point(9, 2), _id=20, actual_pixels_id=20,
                     canvas_id=e.output_canvases[-1].id + 1)
test_dot = Dot(colour=dot_colour, canvas_pos=Point(0, 0), _id=20, actual_pixels_id=20, canvas_id=9)
e.add_object_on_canvasses(test_in_obj, [test_in_obj.canvas_id])
e.add_object_on_canvasses(test_dot, [test_in_obj.canvas_id])

e.test_output_canvas = Canvas(size=e.test_input_canvas.size, _id=e.test_input_canvas.id + 1)
test_out_obj = copy(test_in_obj)
test_out_obj.canvas_pos += colour_to_trans[dot_colour]
test_out_obj.canvas_id = e.test_input_canvas.id + 1
e.add_object_on_canvasses(test_out_obj, [test_out_obj.canvas_id])
e.add_object_on_canvasses(test_dot, [test_out_obj.canvas_id])

e.show()


with_pixels = False
for j, (i, o) in enumerate(zip(e.input_canvases, e.output_canvases)):
    ic = i.json_output(with_pixels=with_pixels)
    oc = o.json_output(with_pixels=with_pixels)
    pair = {'Task': j, 'Input': ic, 'Output': oc}
    json.dump(pair, fp=open(fr'E:/tmp/canvasses/example_{j}.json', mode='w'), indent=2)

test_input = e.test_input_canvas.json_output(with_pixels=with_pixels)
json.dump(test_input, fp=open(fr'E:/tmp/canvasses/test_input.json', mode='w'), indent=2)
test_output = e.test_output_canvas.json_output(with_pixels=with_pixels)
json.dump(test_output, fp=open(fr'E:/tmp/canvasses/test_output.json', mode='w'), indent=2)
