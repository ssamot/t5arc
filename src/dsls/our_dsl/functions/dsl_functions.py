
from typing import List

from data_generators.object_recognition.basic_geometry import Point, Vector, Dimension2D


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


def closest(origin: Point, targets: List[Point]):
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


def loop():
    pass
