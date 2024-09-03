
import json
from copy import copy

from data.generators.object_recognition.basic_geometry import Dimension2D
from dsls.our_dsl.raw_data_to_examples.task_05f2a901 import example

json_dict = example.create_example_json()

json.dump(json_dict, fp=open(fr'E:/tmp/example_05f2a901.json', mode='w'), indent=2)


# CLAUDE's solution

# <editor-fold desc="CLAUDE's Code verbatim.">
'''
def transform_canvas(input_canvas):
    output_canvas = input_canvas.copy()  # Create a copy to modify

    # Find the objects with colour 9 and 3
    obj_9 = None
    obj_3 = None
    for obj in output_canvas['objects']:
        if obj['colour'] == 9:
            obj_9 = obj
        elif obj['colour'] == 3:
            obj_3 = obj

    if obj_9 is None or obj_3 is None:
        return output_canvas  # No transformation needed

    # Calculate the required translation
    if obj_3['dimensions'][0] > obj_3['dimensions'][1]:  # Object 3 is wider than tall
        # Align left edge of obj_3 with left edge of obj_9
        translation_x = obj_9['canvas_pos'][0] - obj_3['canvas_pos'][0]
        translation_y = 0
    else:  # Object 3 is taller than wide or square
        # Align top edge of obj_3 with bottom edge of obj_9
        translation_x = 0
        translation_y = (obj_9['canvas_pos'][1] + obj_9['dimensions'][1]) - obj_3['canvas_pos'][1]

    # Apply the translation
    obj_3['transformations'] = [
        ['translate', {'distance': [translation_x, translation_y, 0]}]
    ]

    # Update the bounding box and canvas position
    obj_3['bbox'] = [
        [obj_3['bbox'][0][0] + translation_x, obj_3['bbox'][0][1] + translation_y],
        [obj_3['bbox'][1][0] + translation_x, obj_3['bbox'][1][1] + translation_y]
    ]
    obj_3['canvas_pos'] = [
        obj_3['canvas_pos'][0] + translation_x,
        obj_3['canvas_pos'][1] + translation_y,
        obj_3['canvas_pos'][2]
    ]

    return output_canvas


def transform_all_canvases(input_canvases):
    return [transform_canvas(canvas) for canvas in input_canvases]
'''
# </editor-fold>


# <editor-fold desc="CLAUDE's Code but with API changes to make it run straight on the Classes and not on the json.">
def transform_canvas(input_canvas):
    output_canvas = copy(input_canvas)  # Create a copy to modify

    # Find the objects with colour 9 and 3
    obj_9 = None
    obj_3 = None
    for obj in output_canvas.objects:
        if obj.colour == 9:
            obj_9 = obj
        elif obj.colour == 3:
            obj_3 = obj

    if obj_9 is None or obj_3 is None:
        return output_canvas  # No transformation needed

    '''
    # First implementation
    if obj_3.dimensions.dx > obj_3.dimensions.dy:  # Object 3 is wider than tall
        # Align left edge of obj_3 with left edge of obj_9
        translation_x = obj_9.canvas_pos.x - obj_3.canvas_pos.x
        translation_y = 0
    else:  # Object 3 is taller than wide or square
        # Align top edge of obj_3 with bottom edge of obj_9
        translation_x = 0
        translation_y = (obj_9.canvas_pos.y + obj_9.dimensions.dy) - obj_3.canvas_pos.y
    '''
    '''
    #Second implementation
    translation_y = (obj_9.canvas_pos.y + obj_9.dimensions.dy) - obj_3.canvas_pos.y
    translation_x = 0
    '''
    '''
    # Third inplementation
    is_vertical = obj_3.dimensions.dy > obj_3.dimensions.dx

    if is_vertical:
        # Align top edge of obj_3 with bottom edge of obj_9
        translation_y = (obj_9.canvas_pos.y + obj_9.dimensions.dy) - obj_3.canvas_pos.y
        translation_x = 0
    else:
        # Align left edge of obj_3 with left edge of obj_9
        translation_x = obj_9.canvas_pos.x - obj_3.canvas_pos.x
        translation_y = 0
    '''
    '''
    # Forth implementation
    distance_x = obj_9.canvas_pos.x- obj_3.canvas_pos.x
    distance_y = obj_9.canvas_pos.y- obj_3.canvas_pos.y

    # Determine the direction of movement
    if abs(distance_x) > abs(distance_y):
        # Move horizontally
        translation_x = distance_x
        translation_y = 0
    else:
        # Move vertically
        translation_x = 0
        translation_y = distance_y
    '''

    # Fifth implementation
    diff_x = obj_9.canvas_pos.x - obj_3.canvas_pos.x
    diff_y = obj_9.canvas_pos.y - obj_3.canvas_pos.y

    # Determine which direction to move based on which difference is larger
    if abs(diff_x) > abs(diff_y):
        translation = Dimension2D(diff_x, 0)
    else:
        translation = Dimension2D(0, diff_y)

    # Apply the translation
    #obj_3['transformations'] = [
    #    ['translate', {'distance': [translation_x, translation_y, 0]}]
    #]
    obj_3.translate_by(translation)

    # Update the bounding box and canvas position
    # This is taken care of automatically
    '''
    obj_3['bbox'] = [
        [obj_3['bbox'][0][0] + translation_x, obj_3['bbox'][0][1] + translation_y],
        [obj_3['bbox'][1][0] + translation_x, obj_3['bbox'][1][1] + translation_y]
    ]
    obj_3['canvas_pos'] = [
        obj_3['canvas_pos'][0] + translation_x,
        obj_3['canvas_pos'][1] + translation_y,
        obj_3['canvas_pos'][2]
    ]
    '''
    output_canvas.embed_objects()
    return output_canvas


# </editor-fold>


i = 2
example.input_canvases[i].show()
output_canvas = transform_canvas(example.input_canvases[i])
output_canvas.show()