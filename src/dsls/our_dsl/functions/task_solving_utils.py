
from copy import copy

from data.generators.task_generator.task import Task
from dsls.our_dsl.solutions import solutions as sols


def solve_canvas_pairs(task: Task, solution, which_pair: int | str = 'all') -> Task:
    input_canvasses = []
    if which_pair == 'all':
        for i in range(len(task.input_canvases)):
            input_canvasses.append(i)
        input_canvasses.append('input')
    else:
        input_canvasses.append(which_pair)

    for task_index in input_canvasses:
        if type(task_index) == int:
            canvas = task.input_canvases[task_index]
            canvas = copy(canvas)
            canvas = solution(canvas)
            task.output_canvases[task_index] = canvas
        else:
            canvas = task.test_input_canvas
            canvas = copy(canvas)
            canvas = solution(canvas)
            task.test_output_canvas = canvas

    return task


def solve_canvas_pairs_for_object_based_search(example, solution, which_pair: int | str = 'all') -> Task:
    input_canvasses = []
    if which_pair == 'all':
        for i in range(len(example.input_canvases)):
            input_canvasses.append(i)
        input_canvasses.append('input')
    else:
        input_canvasses.append(which_pair)

    new_example = Task(number_of_io_pairs=example.number_of_io_pairs)

    for task_index in input_canvasses:
        if type(task_index) == int:
            canvas = example.input_canvases[task_index]
            canvas = copy(canvas)
            new_example.input_canvases[task_index] = canvas
            canvas = solution(canvas)
            new_example.output_canvases[task_index] = canvas
        else:
            canvas = example.test_input_canvas
            canvas = copy(canvas)
            new_example.test_input_canvas = canvas
            canvas = solution(canvas)
            new_example.test_output_canvas = canvas

    return new_example