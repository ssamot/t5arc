from multiprocessing import  Pool
from functools import partial
from typing import List

import numpy as np
from collections.abc import Iterator
from data import load_data as ld
from data.generators.task_generator.arc_task_generator import ARCTask


class ArcTaskData(Iterator):
    def __init__(self, group: str = 'train', augment_with: List[str] | None = None,
                 max_samples: int = 10000, load_step: int = 100, with_black: bool = True):
        """
        The Iterator that generates numpy arrays from the Canvasses of the ARC tasks. It returns a dict with the name
        of the task, the augmentation index, the input numpy array of size (number of task pairs + 1 x 32 x 32) and
        the output array (same size as input). If augment_with is None it iterates per ARC Task. If it is a list of
        strings (e.g. ['colour', 'rotation']) it then augments the datawith the appropriate logic invariant operations
        and it then iterates over each augmented data set. In this case it will return all of the augmented tasks of
        each ARC Task before it moves to the next one. The number of augmented tasks per ARC Task is variable
        (it is specified by the ARC Task itself).
        :param group: 'train' or 'eval'
        :param augment_with: None for no augmented data or a List of strings. Can be 'colour' and 'rotation'
        :param max_samples: The maximum samples for each augmentation
        :param load_step: The number of examples to parallel load in a single go before their augmentations are returned
        :param with_black: If True (default) then the colour permutations are done using the value 1 (black) also. Otherwise the black colour doesn't change
        """
        challenges_names, challenges_tasks, solutions_tasks = ld.from_json(group)
        self.names = challenges_names
        self.tasks = challenges_tasks
        self.solutions = solutions_tasks
        self.current_example = None
        self._index = 0
        self._augmented_index = 0
        self._previous_index = -1
        self.group = group
        self.augment_with = augment_with
        self.max_samples = max_samples

        self.example_names = [None]*len(self.tasks)
        self.example_augment_sizes = [None]*len(self.tasks)
        self.buffer_inputs = [None]*len(self.tasks)
        self.buffer_outputs = [None]*len(self.tasks)

        self.load_step = load_step
        self.with_black = with_black

    def fill_buffer(self, example_index):
        current_example = ARCTask(arc_data=[self.names[example_index], self.tasks[example_index],
                                               self.solutions[example_index]])
        current_example.generate_canvasses(empty=False, augment_with=self.augment_with, max_samples=self.max_samples,
                                           with_black=self.with_black)
        if self.augment_with is None:
            inputs = []
            outputs = []
            for p in range(current_example.number_of_io_pairs):
                inputs.append(current_example.input_canvases[p].full_canvas)
                outputs.append(current_example.output_canvases[p].full_canvas)
            inputs.append(current_example.test_input_canvas.full_canvas)
            outputs.append(current_example.test_output_canvas.full_canvas)
        else:
            inputs = []
            outputs = []
            for augment_index in range(len(current_example.colour_mappings_for_augmentation) *
                                       current_example.number_of_rotations_for_augmentation):
                temp_im = []
                temp_out = []
                for p in range(current_example.number_of_io_pairs):
                    temp_im.append(current_example.input_canvases_augmented[augment_index][p].full_canvas)
                    temp_out.append(current_example.output_canvases_augmented[augment_index][p].full_canvas)
                temp_im.append(current_example.test_input_canvas_augmented[augment_index].full_canvas)
                temp_out.append(current_example.test_output_canvas_augmented[augment_index].full_canvas)
                inputs.append(np.array(temp_im))
                outputs.append(np.array(temp_out))

        #print(f'Finished example {current_example.name} with index {example_index} and size of inputs {np.array(inputs).shape}')
        return example_index, current_example.name, len(current_example.test_input_canvas_augmented), np.array(inputs, dtype=np.int8),\
            np.array(outputs, dtype=np.int8)

    def load_data(self, example_indices):

        with Pool(processes=10) as pool:
            results = pool.map(partial(self.fill_buffer), example_indices)

        for r in results:
            index = r[0]
            self.example_names[index] = r[1]
            self.example_augment_sizes[index] = r[2]
            self.buffer_inputs[index] = r[3]
            self.buffer_outputs[index] = r[4]
            #print(np.array(self.buffer_inputs[index]).shape)

    def __next__(self):
        if self._index < len(self.tasks):

            if self._previous_index != self._index:
                self._augmented_index = 0

                if self._index % self.load_step == 0:
                    example_indices = [i for i in range(self._index, self.load_step + self._index)]
                    self.buffer_inputs = [None]*len(self.tasks)
                    self.buffer_outputs = [None]*len(self.tasks)
                    self.load_data(example_indices=example_indices)

            if self.augment_with is None:
                inputs = self.buffer_inputs[self._index]
                outputs = self.buffer_outputs[self._index]
                self._index += 1
                self._previous_index += 1
            else:
                inputs = self.buffer_inputs[self._index][self._augmented_index]
                outputs = self.buffer_outputs[self._index][self._augmented_index]
                self._augmented_index += 1
                self._previous_index = self._index
                if self._augmented_index == self.example_augment_sizes[self._index]:
                    self._index += 1

            return {'name': self.example_names[self._index - 1], 'augmentation_index': self._augmented_index - 1,
                    'augmented_size': self.example_augment_sizes[self._index - 1],
                    'input': np.array(inputs).tolist(), 'output': np.array(outputs).tolist()}
        else:
            raise StopIteration
