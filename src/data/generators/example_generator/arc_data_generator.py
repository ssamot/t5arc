
import numpy as np

from data import load_data as ld
from data.generators.object_recognition.canvas import Canvas


def get_all_arc_data(group: str = 'train') -> np.ndarray:
    """
    Return all Canvasses of the group ARC set.
    :param group: Whether the data set is the 'train' or the 'eval' ones.
    :return: A 3d array with (number of Canvasses, MAX_PAD_SIZE, MAX_PAD_SIZE) dimensions.
    """
    challenges_names, challenges_tasks, solutions_tasks = ld.from_json(group)
    data_size = len(challenges_tasks)
    result = []
    for index in range(data_size):
        task = challenges_tasks[index]

        for pair in range(len(task['train'])):
            for in_out in ['input', 'output']:
                data = np.flipud(np.array(task['train'][pair][in_out])) + 1
                canvas = Canvas(actual_pixels=data, _id=index)
                result.append(canvas.full_canvas)

        data = np.flipud(np.array(task['test'][0]['input'])) + 1
        canvas = Canvas(actual_pixels=data, _id=index)
        result.append(canvas.full_canvas)

        data = np.flipud(np.array(solutions_tasks[index][0])) + 1

        canvas = Canvas(actual_pixels=data, _id=index)
        result.append(canvas.full_canvas)
    result = np.array(result)
    return result
