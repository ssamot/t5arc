
from copy import copy
from typing import List

from data_generators.object_recognition.basic_geometry import Point, Vector
from data_generators.object_recognition.canvas import Canvas
from data_generators.object_recognition.object import Object


def furthest(origin: Point, targets: List[Point] | Point) -> Vector | None:
    if type(targets) == Point:
        return origin.euclidean_distance(targets)
    result = targets[0]
    for t in targets[1:]:
        if origin.euclidean_distance(result) is None:
            result = t
        elif origin.euclidean_distance(t) is not None:
            if origin.euclidean_distance(t).length > origin.euclidean_distance(result).length:
                result = t

    return origin.euclidean_distance(result)


def closest(origin: Point, targets: List[Point]) -> Vector | None:
    if type(targets) == Point:
        return origin.euclidean_distance(targets)
    result = targets[0]
    for t in targets[1:]:
        if origin.euclidean_distance(result) is None:
            result = t
        elif origin.euclidean_distance(t) is not None:
            if origin.euclidean_distance(t).length < origin.euclidean_distance(result).length:
                result = t
    return origin.euclidean_distance(result)


def largest_object_by_area(canvas: Canvas) -> Object:
    return canvas.sort_objects_by_size(used_dim='area')[-1]


def largest_object_by_height(canvas: Canvas) -> Object:
    return canvas.sort_objects_by_size(used_dim='height')[-1]


def largest_object_by_width(canvas: Canvas) -> Object:
    return canvas.sort_objects_by_size(used_dim='width')[-1]


def smallest_object_by_area(canvas: Canvas) -> Object:
    return canvas.sort_objects_by_size(used_dim='area')[0]


def smallest_object_by_height(canvas: Canvas) -> Object:
    return canvas.sort_objects_by_size(used_dim='height')[0]


def smallest_object_by_width(canvas: Canvas) -> Object:
    return canvas.sort_objects_by_size(used_dim='width')[0]


def rest_of_the_objects(canvas: Canvas, obj: Object):
    temp_obj_list = copy(canvas.objects)
    temp_obj_list.remove(obj)
    return temp_obj_list

