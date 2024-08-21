
from copy import copy
import numpy as np

from data_generators.object_recognition.basic_geometry import Point, Surround
from data_generators.object_recognition.canvas import Canvas
from dsls.our_dsl.functions import dsl_functions as dsl

def solution_045e512c(canvas: Canvas) -> Canvas:
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
    return canvas


def solution_97a05b5b(canvas: Canvas):
    largest_object = canvas.sort_objects_by_size(used_dim='area')[-1]
    other_objects = canvas.sort_objects_by_size(used_dim='area')[:-1]
    neg = copy(largest_object)
    neg.negative_colour()

    canvas_out = Canvas(size=largest_object.size)
    new_lo = copy(largest_object)
    new_lo.canvas_pos = Point(0, 0, -1)
    canvas_out.add_new_object(new_lo)

    for oo in other_objects:
        o3 = copy(oo)
        o3.actual_pixels[np.where(o3.actual_pixels != largest_object.colour)] = 1

        match_positions = o3.match(neg, after_rotation=True, padding=Surround(1, 1, 1, 1))[0]

        new = copy(oo)
        new.rotate(match_positions[1])
        new.canvas_pos = match_positions[0] - largest_object.canvas_pos
        canvas_out.add_new_object(new)

    return canvas_out