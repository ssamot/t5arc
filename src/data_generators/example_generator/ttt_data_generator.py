
import numpy as np

from data import load_data as ld
from data_generators.example_generator.arc_example_generator import ARCExample
from constants import constants as const


class ArcExampleData:
    def __init__(self, group: str = 'train'):
        challenges_names, challenges_tasks, solutions_tasks = ld.from_json(group)
        self.names = challenges_names
        self.tasks = challenges_tasks
        self.solutions = solutions_tasks
        self._index = 0
        self.group = group

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self.tasks):
            i = self._index
            example = ARCExample(arc_data=[self.names[i], self.tasks[i], self.solutions[i]])
            example.generate_canvasses(empty=False)
            inputs = []
            outputs = []
            for p in range(example.number_of_io_pairs):
                inputs.append(example.input_canvases[p].full_canvas)
                outputs.append(example.output_canvases[p].full_canvas)
            inputs.append(example.test_input_canvas.full_canvas)
            outputs.append(example.test_output_canvas.full_canvas)
            self._index += 1

            return {'name': example.name, 'input': np.array(inputs), 'output': np.array(outputs)}
        else:
            raise StopIteration