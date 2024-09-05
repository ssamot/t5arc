

from data.generators.object_recognition.basic_geometry import Point, Surround
from data.generators.object_recognition.canvas import Canvas
from dsls.our_dsl.functions import dsl_functions as dsl


#  12 funcs
def solution_045e512c(canvas: Canvas) -> Canvas:
    largest_object = dsl.select_largest_object_by_area(canvas)
    other_objects = dsl.select_rest_of_the_objects(canvas=canvas, obj=largest_object)
    canvas_pos_lo = dsl.get_object_feature_canvas_pos(largest_object)
    for oo in other_objects:
        point = dsl.get_point_for_match_shape_furthest(largest_object, oo, match_shape_only=True, padding=Surround(0, 0, 0, 0))
        dist, _ = dsl.furthest_point_to_point(canvas_pos_lo, point)
        colour = dsl.get_object_feature_colour(oo)
        obj = dsl.copy_object(largest_object)
        for _ in range(3):
            obj = dsl.object_transform_new_colour(obj, colour)
            obj = dsl.object_transform_translate_along_direction(obj, dist)
            canvas = dsl.add_object_to_canvas(canvas, obj)
    return canvas


#  14 funcs
def solution_97a05b5b(canvas: Canvas) -> Canvas:
    largest_object = dsl.select_largest_object_by_area(canvas)
    other_objects = dsl.select_rest_of_the_objects(canvas=canvas, obj=largest_object)

    neg_object = dsl.object_transform_negate(largest_object)
    lo_canvas_pos = dsl.get_object_feature_canvas_pos(largest_object)

    new_oo = dsl.object_transform_translate_to_point(largest_object, Point(0, 0))
    canvas_size = dsl.get_object_feature_size(new_oo)
    canvas_out = dsl.make_new_canvas(size=canvas_size)
    canvas_out = dsl.add_object_to_canvas(canvas_out, new_oo)

    for oo in other_objects:
        match_position, rotation = dsl.get_point_and_rotation_for_match_shape_furthest(oo, neg_object,
                                                                                       match_shape_only=False,
                                                                                       padding=Surround(1, 1, 1, 1))
        oo = dsl.object_transform_rotate(oo, rotation=rotation)
        oo_pos = dsl.subtract_points(match_position, lo_canvas_pos)
        oo = dsl.object_transform_translate_to_point(oo, oo_pos)
        canvas_out = dsl.add_object_to_canvas(canvas_out, oo)

    return canvas_out


#  14 funcs
def solution_b775ac94(canvas: Canvas) -> Canvas:
    object_list = dsl.select_rest_of_the_objects(canvas, obj=None)
    canvas_out = dsl.copy_canvas(canvas)

    for init_obj in object_list:
        canvas_temp = dsl.make_new_canvas_as(canvas)
        canvas_temp = dsl.add_object_to_canvas(canvas_temp, init_obj)
        canvas_temp = dsl.split_object_by_colour_on_canvas(canvas=canvas_temp, obj=init_obj)
        largest_object = dsl.select_largest_object_by_area(canvas=canvas_temp)
        other_objects = dsl.select_rest_of_the_objects(canvas=canvas_temp, obj=largest_object)

        for oo in other_objects:
            colour = dsl.get_object_feature_colour(oo)
            dist = dsl.get_distance_min_between_objects(largest_object, oo)
            largest_object = dsl.object_transform_flip_and_translate(largest_object, dist)
            largest_object = dsl.object_transform_new_colour(largest_object, colour)
            canvas_out.add_new_object(largest_object)

    return canvas_out


#  7 funcs
def solution_05f2a901(canvas: Canvas) -> Canvas:
    blue = dsl.select_object_of_colour(canvas, 9)
    red = dsl.select_object_of_colour(canvas, 3)
    dist = dsl.get_distance_touching_between_objects(red, blue)

    red = dsl.object_transform_translate_along_direction(red, dist)
    out_canvas = dsl.make_new_canvas_as(canvas)
    out_canvas = dsl.add_object_to_canvas(out_canvas, red)
    out_canvas = dsl.add_object_to_canvas(out_canvas, blue)

    return out_canvas
