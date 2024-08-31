
from copy import copy
import json

from data.generators.example_generator.example import Example
from data.generators.object_recognition.basic_geometry import Dimension2D, Point
from data.generators.object_recognition.canvas import Canvas
from data.generators.object_recognition.primitives import Random, Dot

e = Example(number_of_io_pairs=4, prob_of_background_object=0, run_generate_canvasses=False)
e.experiment_type = 'other'
e.generate_canvasses()

for i in range(len(e.input_canvases)):
    e.output_canvases[i] = Canvas(size=e.input_canvases[i].size, _id=e.output_canvases[i].id)

start_points = [Point(3, 4), Point(7, 3), Point(2, 10), Point(9, 8)]
colour_to_trans = {2: Point(4, 2), 4: Point(3, 0)}
transitions = [2, 4, 4, 2]
end_points = [p + colour_to_trans[transitions[i]] for i, p in enumerate(start_points)]

objs1 = [Random(size=Dimension2D(5, 7), _id=0, actual_pixels_id=0, canvas_pos=start_points[0], canvas_id=1),
         Dot(colour=transitions[0], _id=1, actual_pixels_id=1, canvas_pos=Point(0, 0), canvas_id=1),
         Random(size=Dimension2D(6, 7), _id=2, actual_pixels_id=2, canvas_pos=start_points[1], canvas_id=3),
         Dot(colour=transitions[1], _id=3, actual_pixels_id=3, canvas_pos=Point(0, 0), canvas_id=3),
         Random(size=Dimension2D(5, 8), _id=4, actual_pixels_id=4, canvas_pos=start_points[2], canvas_id=5),
         Dot(colour=transitions[2], _id=5, actual_pixels_id=5, canvas_pos=Point(0, 0), canvas_id=5),
         Random(size=Dimension2D(5, 8), _id=6, actual_pixels_id=6, canvas_pos=start_points[3], canvas_id=7),
         Dot(colour=transitions[3], _id=7, actual_pixels_id=7, canvas_pos=Point(0, 0), canvas_id=7)
         ]

for o in objs1[0: 6: 2]:
    o.randomise_colour(30)

objs2 = [copy(objs1[0]),
         Dot(colour=transitions[0], _id=9, actual_pixels_id=9, canvas_pos=Point(0, 0), canvas_id=2),
         copy(objs1[2]),
         Dot(colour=transitions[1], _id=11, actual_pixels_id=11, canvas_pos=Point(0, 0), canvas_id=4),
         copy(objs1[4]),
         Dot(colour=transitions[2], _id=13, actual_pixels_id=13, canvas_pos=Point(0, 0), canvas_id=6),
         copy(objs1[6]),
         Dot(colour=transitions[3], _id=15, actual_pixels_id=15, canvas_pos=Point(0, 0), canvas_id=8),
         ]

for i in [0, 2, 4, 6]:
    objs2[i].canvas_pos = end_points[i//2]
    objs2[i].canvas_id = i + 2

for o1, o2 in zip(objs1, objs2):
    e.add_object_on_canvasses(o1, [o1.canvas_id])
    e.add_object_on_canvasses(o2, [o2.canvas_id])

dot_colour = 2
test_in_obj = Random(size=Dimension2D(7, 4), canvas_pos=Point(9, 2), _id=20, actual_pixels_id=20, canvas_id=9)
test_dot = Dot(colour=dot_colour, canvas_pos=Point(0, 0), _id=20, actual_pixels_id=20, canvas_id=9)
e.add_object_on_canvasses(test_in_obj, [test_in_obj.canvas_id])
e.add_object_on_canvasses(test_dot, [test_in_obj.canvas_id])

e.test_output_canvas = Canvas(size=e.test_input_canvas.size, _id=e.test_input_canvas.id + 1)
test_out_obj = copy(test_in_obj)
test_out_obj.canvas_pos += transitions[dot_colour]
test_out_obj.canvas_id = 10
e.add_object_on_canvasses(test_out_obj, [test_out_obj.canvas_id])
e.add_object_on_canvasses(test_dot, [test_out_obj.canvas_id])

e.show()

objects, arrays = e.json_output_of_all_objects(lean=False)

with_pixels = False
for j, (i, o) in enumerate(zip(e.input_canvases, e.output_canvases)):
    ic = i.json_output(with_pixels=with_pixels)
    oc = o.json_output(with_pixels=with_pixels)
    pair = {'Example': j, 'Input': ic, 'Output': oc}
    json.dump(pair, fp=open(fr'E:/tmp/canvasses/example_{j}.json', mode='w'), indent=2)

test_input = e.test_input_canvas.json_output(with_pixels=with_pixels)
json.dump(test_input, fp=open(fr'E:/tmp/canvasses/test_input.json', mode='w'), indent=2)
test_output = e.test_output_canvas.json_output(with_pixels=with_pixels)
json.dump(test_output, fp=open(fr'E:/tmp/canvasses/test_output.json', mode='w'), indent=2)
