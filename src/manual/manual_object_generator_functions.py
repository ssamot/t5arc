from typing import List

import numpy as np
from data.generators.object_recognition.primitives import Primitive, Predefined


def get_array_from_object(obj: Primitive) -> np.array:
    return np.flipud(obj.actual_pixels)


def generate_primitive_from_array(array: np.ndarray) -> Primitive:
    primitive = Predefined(actual_pixels=array)
    return primitive


def template(obj: Primitive) -> List[Primitive]:
    array = get_array_from_object(obj)

    new_object_arrays = []
    # Do stuff

    new_objects = []
    for a in new_object_arrays:
        new_objects.append(generate_primitive_from_array(a))


def same_colour_connected_pixels(in_obj: Primitive, out_obj: Primitive, for_canvas: str):
    base_obj = in_obj if for_canvas == 'input' else out_obj

    colours = base_obj.get_used_colours()

    all_objects = []
    for colour in colours:
        one_colour_objects = base_obj.create_new_primitives_from_pixels_of_colour(colour)

        for oco in one_colour_objects:
            all_objects.append(oco)

    return all_objects
