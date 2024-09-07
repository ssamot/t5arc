from typing import List

import numpy as np
from collections.abc import Iterator
from data import load_data as ld
from data.generators.example_generator.arc_example_generator import ARCExample


class ArcExampleData(Iterator):
    def __init__(self, group: str = 'train', augment_with: List[str] | None = None):
        """
        The Iterator that generates numpy arrays from the Canvasses of the ARC examples. It returns a dict with the name
        of the example, the augmentation index, the input numpy array of size (number of example pairs + 1 x 32 x 32) and
        the output array (same size as input). If augment_with is None it iterates per ARC Example. If it is a list of
        strings (e.g. ['colour', 'rotation']) it then augments the datawith the appropriate logic invariant operations
        and it then iterates over each augmented data set. In this case it will return all of the augmented examples of
        each ARC Example before it moves to the next one. The number of augmented examples per ARC Example is variable
        (it is specified by the ARC Example itself).
        :param group: 'train' or 'eval'
        :param augment_with: None for no augmented data or a List of strings. Can be 'colour' and 'rotation'
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

    def __next__(self):
        if self._index < len(self.tasks):
            if self._previous_index != self._index:
                self._augmented_index = 0
                i = self._index
                self.current_example = ARCExample(arc_data=[self.names[i], self.tasks[i], self.solutions[i]])
                self.current_example.generate_canvasses(empty=False, augment_with=self.augment_with)
            if self.augment_with is None:
                inputs = []
                outputs = []
                for p in range(self.current_example.number_of_io_pairs):
                    inputs.append(self.current_example.input_canvases[p].full_canvas)
                    outputs.append(self.current_example.output_canvases[p].full_canvas)
                inputs.append(self.current_example.test_input_canvas.full_canvas)
                outputs.append(self.current_example.test_output_canvas.full_canvas)
                self._index += 1
                self._previous_index += 1
            else:
                inputs = []
                outputs = []
                a = self._augmented_index
                for p in range(self.current_example.number_of_io_pairs):
                    inputs.append(self.current_example.input_canvases_augmented[a][p].full_canvas)
                    outputs.append(self.current_example.output_canvases_augmented[a][p].full_canvas)
                inputs.append(self.current_example.test_input_canvas_augmented[a].full_canvas)
                outputs.append(self.current_example.test_output_canvas_augmented[a].full_canvas)
                self._augmented_index += 1
                self._previous_index = self._index
                if self._augmented_index == len(self.current_example.test_input_canvas_augmented):
                    self._index += 1

            return {'name': self.current_example.name, 'augmentation_index': self._augmented_index - 1,
                    'augmented_size': len(self.current_example.test_input_canvas_augmented),
                    'input': np.array(inputs), 'output': np.array(outputs)}
        else:
            raise StopIteration
