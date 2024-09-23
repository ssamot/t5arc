
from copy import copy
from typing import List, Tuple

import numpy as np

from data.generators.object_recognition.basic_geometry import Point, Vector, Dimension2D, RelativePoint, Orientation, \
    Surround
from data.generators.object_recognition.canvas import Canvas
from data.generators.object_recognition.primitives import Primitive


#  Funcs on Points
def furthest_point_to_point(origin: Point, targets: List[Point] | Point) -> Tuple[Vector, int]:
    if type(targets) == Point:
        return origin.euclidean_distance(targets), 0
    result = targets[0]
    index = 0
    for i, t in enumerate(targets):
        if origin.euclidean_distance(result) is None:
            result = t
            index = i
        elif origin.euclidean_distance(t) is not None:
            if origin.euclidean_distance(t).length > origin.euclidean_distance(result).length:
                result = t
                index = i

    return origin.euclidean_distance(result), index


def closest_point_to_point(origin: Point, targets: List[Point] | Point) -> Tuple[Vector, int]:
    if type(targets) == Point:
        return origin.euclidean_distance(targets), 0
    result = targets[0]
    index = 0
    for i, t in enumerate(targets):
        if origin.euclidean_distance(result) is None:
            result = t
            index = i
        elif origin.euclidean_distance(t) is not None:
            if origin.euclidean_distance(t).length < origin.euclidean_distance(result).length:
                result = t
                index = i

    return origin.euclidean_distance(result), index


def sum_points(first: Point, second: Point) -> Point:
    f = copy(first)
    s = copy(second)
    return f + s


def subtract_points(first: Point, second: Point) -> Point:
    f = copy(first)
    s = copy(second)
    return f - s


#  Funcs on Canvasses
def copy_canvas(canvas: Canvas) -> Canvas:
    return copy(canvas)


def copy_object(obj: Primitive) -> Primitive:
    return copy(obj)


def make_new_canvas_as(canvas: Canvas) -> Canvas:
    new_canvas = Canvas(size=canvas.size)
    return new_canvas


def make_new_canvas(size: Dimension2D) -> Canvas:
    return Canvas(size=size)


def add_object_to_canvas(canvas: Canvas, obj: Primitive) -> Canvas:
    new_canvas = copy(canvas)
    new_obj = copy(obj)
    new_canvas.add_new_object(new_obj)
    return new_canvas


# Funcs to get Primitive features
def get_distance_min_between_objects(first: Primitive, second: Primitive) -> Vector:
    return first.distance_to_object(other=second, dist_type='min')


def get_distance_max_between_objects(first: Primitive, second: Primitive) -> Vector:
    return first.distance_to_object(other=second, dist_type='max')


def get_distance_origin_to_origin_between_objects(first: Primitive, second: Primitive) -> Vector:
    return first.distance_to_object(other=second, dist_type='canvas_pos')


def get_distance_touching_between_objects(first: Primitive, second: Primitive) -> Vector:
    dist = get_distance_min_between_objects(first=first, second=second)
    dist.length -= 1
    return dist


def get_point_for_match_shape_furthest(base_obj: Primitive, target_obj: Primitive, match_shape_only: bool,
                                       padding: Surround = Surround(0, 0, 0, 0)) -> Point:
    match_positions, _ = base_obj.match(target_obj, after_rotation=True,
                                                match_shape_only=match_shape_only, padding=padding)

    _, index = furthest_point_to_point(base_obj.canvas_pos, match_positions)

    return match_positions[index]


def get_point_and_rotation_for_match_shape_furthest(base_obj: Primitive, target_obj: Primitive, match_shape_only: bool,
                                                    padding: Surround = Surround(0, 0, 0, 0)) -> Tuple[Point, int]:
    match_positions, rotations = base_obj.match(target_obj, after_rotation=True,
                                                match_shape_only=match_shape_only, padding=padding)

    _, index = furthest_point_to_point(base_obj.canvas_pos, match_positions)

    return match_positions[index], rotations[index]


def get_point_for_match_shape_nearest(base_obj: Primitive, target_obj: Primitive, match_shape_only: bool,
                                      padding: Surround = Surround(0, 0, 0, 0)) -> Point:
    match_positions, _ = base_obj.match(target_obj, after_rotation=True,
                                                match_shape_only=match_shape_only, padding=padding)

    _, index = closest_point_to_point(base_obj.canvas_pos, match_positions)

    return match_positions[index]


def get_point_and_rotation_for_match_shape_nearest(base_obj: Primitive, target_obj: Primitive, match_shape_only: bool,
                                                   padding: Surround = Surround(0, 0, 0, 0)) -> Tuple[Point, int]:
    match_positions, rotations = base_obj.match(target_obj, after_rotation=True,
                                                match_shape_only=match_shape_only, padding=padding)

    _, index = closest_point_to_point(base_obj.canvas_pos, match_positions)

    return match_positions[index], rotations[index]


def get_object_feature_colour(obj: Primitive) -> int:
    return obj.colour


def get_object_feature_size(obj: Primitive) -> Dimension2D:
    return obj.size


def get_object_feature_canvas_pos(obj: Primitive) -> Point:
    return obj.canvas_pos


# Funcs to select Primitives
def select_largest_object_by_area(canvas: Canvas) -> Primitive:
    new_canvas = copy(canvas)
    return new_canvas.sort_objects_by_size(used_dim='area')[-1]


def select_largest_object_by_height(canvas: Canvas) -> Primitive:
    new_canvas = copy(canvas)
    return new_canvas.sort_objects_by_size(used_dim='height')[-1]


def select_largest_object_by_width(canvas: Canvas) -> Primitive:
    new_canvas = copy(canvas)
    return new_canvas.sort_objects_by_size(used_dim='width')[-1]


def select_smallest_object_by_area(canvas: Canvas) -> Primitive:
    new_canvas = copy(canvas)
    return new_canvas.sort_objects_by_size(used_dim='area')[0]


def select_smallest_object_by_height(canvas: Canvas) -> Primitive:
    new_canvas = copy(canvas)
    return new_canvas.sort_objects_by_size(used_dim='height')[0]


def select_smallest_object_by_width(canvas: Canvas) -> Primitive:
    new_canvas = copy(canvas)
    return new_canvas.sort_objects_by_size(used_dim='width')[0]


def select_rest_of_the_objects(canvas: Canvas, obj: Primitive ) -> List[Primitive]:
    temp_obj_list = [copy(o) for o in canvas.objects]
    if obj is not None:
        temp_obj_list.remove(obj)
    return temp_obj_list


def select_all_objects_of_colour(canvas: Canvas, colour: int) -> List[Primitive]:
    new_canvas = copy(canvas)
    return new_canvas.find_objects_of_colour(colour)


def select_only_object_of_colour(canvas: Canvas, colour: int) -> Primitive:
    return select_all_objects_of_colour(canvas, colour=colour)[0]


def select_objects_of_type(canvas: Canvas, primitive_type: type[Primitive]) -> List[Primitive]:
    new_canvas = copy(canvas)
    objs_of_type = []
    for obj in new_canvas.objects:
        if isinstance(obj, primitive_type):
            objs_of_type.append(copy(obj))

    return objs_of_type


# Funcs to transform Primitives
def object_transform_rotate(obj: Primitive, rotation: int) -> Primitive:
    new_obj = copy(obj)
    new_obj.rotate(times=rotation)
    return new_obj


def object_transform_translate_to_point(obj: Primitive, target_point: Point,
                                        object_point: Point | None = None) -> Primitive:
    new_obj = copy(obj)
    new_obj.translate_to(target_point=target_point, object_point=object_point)
    return new_obj


def object_transform_translate_by_distance(obj: Primitive, distance: Dimension2D) -> Primitive:
    new_obj = copy(obj)
    new_obj.translate_by(distance=distance)
    return new_obj


def object_transform_translate_along_direction(obj: Primitive, direction: Vector) -> Primitive:
    new_obj = copy(obj)
    new_obj.translate_along(direction=direction)
    return new_obj


def object_transform_translate_relative_point_to_point(obj: Primitive, relative_point: RelativePoint,
                                                       other_point: Point) -> Primitive:
    new_obj = copy(obj)
    new_obj.translate_relative_point_to_point(relative_point=relative_point, other_point=other_point)
    return new_obj


def object_transform_mirror(obj: Primitive, axis: Orientation):
    new_obj = copy(obj)
    new_obj.mirror(axis=axis, on_axis=False)
    return new_obj


def object_transform_mirror_on_axis(obj: Primitive, axis: Orientation):
    new_obj = copy(obj)
    new_obj.mirror(axis=axis, on_axis=True)
    return new_obj


def object_transform_flip_only(obj: Primitive, axis: Orientation | Vector):
    new_obj = copy(obj)
    if isinstance(axis, Vector):
        axis = axis.orientation
    new_obj.flip(axis=axis, translate=False)
    return new_obj


def object_transform_flip_and_translate(obj: Primitive, axis: Orientation | Vector):
    new_obj = copy(obj)
    if isinstance(axis, Vector):
        axis = axis.orientation
    new_obj.flip(axis=axis, translate=True)
    return new_obj


def object_transform_new_colour(obj: Primitive, colour: int) -> Primitive:
    new_obj = copy(obj)
    new_obj.set_new_colour(new_colour=colour)
    return new_obj


def object_transform_negate(obj: Primitive) -> Primitive:
    new_obj = copy(obj)
    new_obj.negative_colour()
    return  new_obj


def object_transform_delete_colour(obj: Primitive, colour: int) -> Primitive:
    new_obj = copy(obj)
    new_obj.actual_pixels[np.where(new_obj.actual_pixels == colour)] = 1
    return new_obj


def split_object_by_colour(obj: Primitive) -> List[Primitive]:
    temp_canvas = Canvas(size=obj.size)
    o = copy(obj)
    temp_canvas.add_new_object(o)
    temp_canvas.split_object_by_colour(o)
    objects = [ob for ob in temp_canvas.objects]
    return objects


def split_object_by_colour_on_canvas(canvas: Canvas, obj: Primitive) -> Canvas:
    new_canvas = copy(canvas)
    new_canvas.split_object_by_colour(obj)
    return new_canvas
