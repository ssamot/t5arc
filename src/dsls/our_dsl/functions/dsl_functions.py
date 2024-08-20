
from typing import List

from data_generators.object_recognition.basic_geometry import Point, Vector, Dimension2D


def furthest(origin: Point, targets: List[Point]) -> Vector | None:
    result = targets[0]
    for t in targets[1:]:
        if origin.manhattan_direction(result) is None:
            result = t
        elif origin.manhattan_direction(t) is not None:
            if origin.manhattan_direction(t).length > origin.manhattan_direction(result).length:
                result = t

    return origin.manhattan_direction(result)


def closest(origin: Point, targets: List[Point]):
    result = targets[0]
    for t in targets[1:]:
        if origin.manhattan_direction(result) is None:
            result = t
        elif origin.manhattan_direction(t) is not None:
            if origin.manhattan_direction(t).length < origin.manhattan_direction(result).length:
                result = t

    return origin.manhattan_direction(result)


def loop():
    pass
