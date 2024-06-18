


import os
import gc
import json
import random
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

from colorama import Style, Fore
import matplotlib.pyplot as plt
from matplotlib import colors


train_challenges_path = '../arc-prize-2024/arc-agi_training_challenges.json'
train_solutions_path = '../arc-prize-2024/arc-agi_training_solutions.json'

eval_challenges_path = '../arc-prize-2024/arc-agi_evaluation_challenges.json'
eval_solutions_path = '../arc-prize-2024/arc-agi_evaluation_solutions.json'

test_path = '../arc-prize-2024/arc-agi_test_challenges.json'
sample_path = '../arc-prize-2024/sample_submission.json'


def from_json(data_type='train'):
    """
    Loads the required data from the json files
    :param data_type: str => 'train' OR 'eval' OR 'test'
    :return: names of set, input images of set, (output images of set)
    """
    if data_type == 'train':
        challenges_path = train_challenges_path
        solutions_path = train_solutions_path
    elif data_type == 'eval':
        challenges_path = eval_challenges_path
        solutions_path = eval_solutions_path
    elif data_type == 'test_path':
        challenges_path = test_path

        # Load challenges and solutions
    with open(challenges_path, 'r') as f:
        challenges_names = list(json.load(f).keys())
    if data_type != 'test_path':
        with open(solutions_path, 'r') as f:
            solutions_names = list(json.load(f).keys())

        assert challenges_names == solutions_names, \
            print('ERROR: Challenges and Solutions names are not the same list')

    print(f' Number of items in train set {len(challenges_names)}')

    with open(challenges_path, 'r') as f:
        challenges_tasks = list(json.load(f).values())
    if data_type != 'test_path':
        with open(solutions_path, 'r') as f:
            solutions_tasks = list(json.load(f).values())

        return challenges_names, challenges_tasks, solutions_tasks

    else:
        return challenges_names, challenges_tasks


