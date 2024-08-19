
from typing import List

from data_generators.object_recognition.basic_geometry import Point, Vector, Dimension2D


def furthest(origin: Point, targets: List[Point]):
    result = targets[0]
    if origin.euclidean_dist_2d(origin) is not None:
        for t in targets[1:]:
            if origin.euclidean_dist_2d(result) is None:
                result = t
            elif origin.euclidean_dist_2d(t) is not None:
                if origin.euclidean_dist_2d(t).length > origin.euclidean_dist_2d(result).length:
                    result = t
    else:
        return None

    return origin.euclidean_dist_2d(result)


def closest(origin: Point, targets: List[Point]):
    result = targets[0]
    if origin.euclidean_dist_2d(origin) is not None:
        for t in targets[1:]:
            if origin.euclidean_dist_2d(result) is None:
                result = t
            elif origin.euclidean_dist_2d(t) is not None:
                if origin.euclidean_dist_2d(t).length < origin.euclidean_dist_2d(result).length:
                    result = t
    else:
        return None

    return origin.euclidean_dist_2d(result)
