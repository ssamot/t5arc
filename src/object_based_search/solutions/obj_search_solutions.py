from typing import List

from data.generators.object_recognition.basic_geometry import Point, Surround, Vector
from data.generators.object_recognition.canvas import Canvas
from data.generators.object_recognition.primitives import Primitive, Random
from dsls.our_dsl.functions import dsl_functions as dsl


def solution_05f2a901(canvas: Canvas) -> Canvas:
    def translate_by_params(canvas: Canvas) -> Point:
        return Point(0, 0)

    def translate_along_params(canvas: Canvas) -> Vector:
        blue = dsl.select_only_object_of_colour(canvas, 9)
        red = dsl.select_only_object_of_colour(canvas, 3)
        dist = dsl.get_distance_touching_between_objects(red, blue)
        return dist

    def objects_to_transform(canvas: Canvas) -> List[Primitive]:
        return dsl.select_objects_of_type(canvas, Random)



