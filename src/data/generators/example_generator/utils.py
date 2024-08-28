import numpy as np
import os
import json
from data.generators.object_recognition.primitives import Primitive, Object, Point


def do_two_objects_overlap(object_a: Primitive | Object, object_b: Primitive | Object) -> bool:

    top_left_a = Point(object_a.bbox.top_left.x, object_a.bbox.top_left.y)
    top_left_a.x -= object_a.required_dist_to_others.Left
    top_left_a.y += object_a.required_dist_to_others.Up
    bottom_right_a = Point(object_a.bbox.bottom_right.x, object_a.bbox.bottom_right.y)
    bottom_right_a.x += object_a.required_dist_to_others.Right
    bottom_right_a.y -= object_a.required_dist_to_others.Down

    top_left_b = Point(object_b.bbox.top_left.x, object_b.bbox.top_left.y)
    top_left_b.x -= object_b.required_dist_to_others.Left
    top_left_b.y += object_b.required_dist_to_others.Up
    bottom_right_b = Point(object_b.bbox.bottom_right.x, object_b.bbox.bottom_right.y)
    bottom_right_b.x += object_b.required_dist_to_others.Right
    bottom_right_b.y -= object_b.required_dist_to_others.Down

    # if rectangle has area 0, no overlap
    if top_left_a.x == bottom_right_a.x or top_left_a.y == bottom_right_a.y or bottom_right_b.x == top_left_b.x or top_left_b.y == bottom_right_b.y:
        return False

    # If one rectangle is on left side of other
    if top_left_a.x > bottom_right_b.x or top_left_b.x > bottom_right_a.x:
        return False

    # If one rectangle is above other
    if bottom_right_a.y > top_left_b.y or bottom_right_b.y > top_left_a.y:
        return False

    return True

