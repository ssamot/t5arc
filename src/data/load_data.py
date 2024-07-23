
from os.path import join, dirname, relpath
import json
from typing import List
import numpy as np

path = dirname(relpath(__file__))
path = r'E:\Code\Competitions\ARC2024\t5arc\src'
raw_data_path = join(path, 'data', 'raw_data')

train_challenges_path = join(raw_data_path, 'arc-agi_training_challenges.json')
train_solutions_path = join(raw_data_path, 'arc-agi_training_solutions.json')

eval_challenges_path = join(raw_data_path, 'arc-agi_evaluation_challenges.json')
eval_solutions_path = join(raw_data_path, 'arc-agi_evaluation_solutions.json')

test_path = join(raw_data_path, 'arc-agi_test_challenges.json')
sample_path = join(raw_data_path, 'sample_submission.json')

names: list


def from_json(data_type: str = 'train') -> tuple[List, List, List] | tuple[List, List]:
    """
    Loads the required data from the json files
    :param data_type: str => 'train' OR 'eval' OR 'test'
    :return: names of set, input images of set, (output images of set if data_type is 'train' or 'eval')
    """
    global names

    if data_type == 'train':
        challenges_path = train_challenges_path
        solutions_path = train_solutions_path
    elif data_type == 'eval':
        challenges_path = eval_challenges_path
        solutions_path = eval_solutions_path
    elif data_type == 'test':
        challenges_path = test_path

        # Load challenges and solutions
    with open(challenges_path, 'r') as f:
        challenges_names = list(json.load(f).keys())

    names = challenges_names

    if data_type != 'test':
        with open(solutions_path, 'r') as f:
            solutions_names = list(json.load(f).keys())

        assert challenges_names == solutions_names, \
            print('ERROR: Challenges and Solutions names are not the same list')

    print(f'Number of items in {data_type} set {len(challenges_names)}')

    with open(challenges_path, 'r') as f:
        challenges_tasks = list(json.load(f).values())
    if data_type != 'test':
        with open(solutions_path, 'r') as f:
            solutions_tasks = list(json.load(f).values())

        return challenges_names, challenges_tasks, solutions_tasks

    else:
        return challenges_names, challenges_tasks


def get_index_from_name(name):
    result = [i for i, j in enumerate(names) if j == name]
    return result[0]


def get_name_from_index(index):
    return names[index]


def get_canvas_sizes(tasks: List) -> tuple[List, List, List]:
    input_canvases_size = []
    output_canvases_size = []
    test_canvases_size = []

    for task in tasks:
        canvas = task['test'][0]['input']
        test_canvases_size.append([len(canvas), len(canvas[0])])

        for tr in task['train']:
            canvas = tr['input']
            input_canvases_size.append([len(canvas), len(canvas[0])])
            canvas = tr['output']
            output_canvases_size.append([len(canvas), len(canvas[0])])

    return input_canvases_size, output_canvases_size, test_canvases_size


def hist_of_canvas_sizes(canvas_sizes: List[List]) -> dict:

    unique_sizes = {}

    for s in canvas_sizes:
        if str(s) not in unique_sizes:
            unique_sizes[str(s)] = 1
        else:
            unique_sizes[str(s)] += 1

    return unique_sizes


def generate_json_of_canvas_sizes():

    train_challenges_names, train_challenges_tasks, train_solutions_tasks = from_json('train')

    train_input_canvases_size, train_output_canvases_size, train_test_canvases_size = \
        get_canvas_sizes(train_challenges_tasks)

    eval_challenges_names, eval_challenges_tasks, eval_solutions_tasks = from_json('eval')

    eval_input_canvases_size, eval_output_canvases_size, eval_test_canvases_size = \
        get_canvas_sizes(eval_challenges_tasks)

    all_unique_sizes = hist_of_canvas_sizes((train_input_canvases_size + train_output_canvases_size +
                                             train_test_canvases_size + eval_output_canvases_size +
                                             eval_input_canvases_size + eval_test_canvases_size))

    all_unique_sizes = {k: v for k, v in sorted(all_unique_sizes.items(), key=lambda item: item[1], reverse=True)}
    json_file = join(path, 'data_generators', 'object_recognition', 'canvas_sizes.json')

    with open(json_file, "w") as outfile:
        json.dump(all_unique_sizes, outfile, indent=3)