
from typing import List

import numpy as np
from data import load_data as ld
from data_generators.example_generator.example import Example
from data_generators.object_recognition.basic_geometry import Dimension2D, Point
from data_generators.object_recognition.canvas import Canvas


class ARCExample(Example):

    def __init__(self, arc_name: str | None = None, arc_group_and_index: List | None = None):

        assert not(arc_name is None and arc_group_and_index is None), print(f'Making ARC Example. Both arc_name and '
                                                                            f'arc_group_and_index cannot be None!')

        if arc_group_and_index is not None:
            group = arc_group_and_index[0]
            index = arc_group_and_index[1]
            challenges_names, challenges_tasks, solutions_tasks = ld.from_json(group)
            solution = solutions_tasks[index]
            task = challenges_tasks[index]

        if arc_name is not None:
            train_challenges_names, train_challenges_tasks, train_solutions_tasks = ld.from_json('train')
            eval_challenges_names, eval_challenges_tasks, eval_solutions_tasks = ld.from_json('eval')

            for i, names in enumerate([train_challenges_names, eval_challenges_names]):
                if arc_name in names:
                    index = [k for k, j in enumerate(names) if j == arc_name][0]
                    task = [train_challenges_tasks, eval_challenges_tasks][i][index]
                    solution = [train_solutions_tasks, eval_solutions_tasks][i][index]

        self.task_data = task
        self.solution_data = solution

        super().__init__()

        self.experiment_type = 'ARC'

        self.test_output_canvas = None

    def generate_canvasses(self, empty: bool = True):
        self.number_of_io_pairs = len(self.task_data['train'])
        self.number_of_canvasses = self.number_of_io_pairs * 2 + 2

        for pair in range(self.number_of_io_pairs):
            input_data = np.flipud(np.array(self.task_data['train'][pair]['input']) + 1)
            output_date = np.flipud(np.array(self.task_data['train'][pair]['output']) + 1)
            if not empty:
                self.input_canvases.append(Canvas(actual_pixels=input_data, _id=pair * 2))
                self.output_canvases.append(Canvas(actual_pixels=output_date, _id=pair * 2 + 1))
            else:
                self.input_canvases.append(Canvas(size=Dimension2D(input_data.shape[1], input_data.shape[0]), _id=pair * 2))
                self.output_canvases.append(Canvas(size=Dimension2D(output_date.shape[1], output_date.shape[0]), _id=pair * 2 + 1))

        test_input_date = np.flipud(np.array(self.task_data['test'][0]['input']) + 1)
        test_output_data = np.flipud(np.array(self.solution_data[0]) + 1)  # TODO: Fix this!
        if not empty:
            self.test_input_canvas = Canvas(actual_pixels=test_input_date, _id=2 * self.number_of_io_pairs)
            self.test_output_canvas = Canvas(actual_pixels=test_output_data, _id=2 * self.number_of_io_pairs + 1)
        else:
            self.test_input_canvas = Canvas(size=Dimension2D(test_input_date.shape[1], test_input_date.shape[0]),
                                            _id=2 * self.number_of_io_pairs)
            self.test_output_canvas = Canvas(size=Dimension2D(test_output_data.shape[1], test_output_data.shape[0]),
                                             _id=2 * self.number_of_io_pairs + 1)

    def get_object_pixels_from_data(self, canvas_id: int, canvas_pos: Point, size: Dimension2D):

        group = 'train'
        from_in_or_out = 'input'
        if canvas_id % 2 == 1:
            from_in_or_out = 'output'
        from_canvas = canvas_id // 2
        if canvas_id == self.number_of_io_pairs * 2:
            group = 'test'
            from_canvas = 0
            from_in_or_out = 'input'

        actual_pixels = np.flipud(self.task_data[group][from_canvas][from_in_or_out])  \
                                    [canvas_pos.y:canvas_pos.y + size.dy, canvas_pos.x:canvas_pos.x + size.dx] + 1

        return actual_pixels

    def reset_object_colours(self):
        for o in self.objects:
            o.set_colour_to_most_common()
