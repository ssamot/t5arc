
from copy import copy

from data.generators.example_generator.example import Example
from dsls.our_dsl.solutions import solutions as sols


def solve_canvas_pairs(example: Example, solution, which_pair: int | str = 'all'):
    input_canvasses = []
    if which_pair == 'all':
        for i in range(len(example.input_canvases)):
            input_canvasses.append(i)
        input_canvasses.append('input')
    else:
        input_canvasses.append(which_pair)

    for task_index in input_canvasses:
        if type(task_index) == int:
            canvas = example.input_canvases[task_index]
            canvas = copy(canvas)
            canvas = solution(canvas)
            example.output_canvases[task_index] = canvas
        else:
            canvas = example.test_input_canvas
            canvas = copy(canvas)
            canvas = solution(canvas)
            example.test_output_canvas = canvas

    return example
