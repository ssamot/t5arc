
from copy import deepcopy, copy
from typing import List, Dict
import numpy as np

from data import load_data as ld
from data.generators.example_generator.example import Example
from data.generators.object_recognition.basic_geometry import Dimension2D, Point
from data.generators.object_recognition.canvas import Canvas
from data.generators.example_generator import utils
from data.generators import constants as const

from_json = ld.from_json


class ARCExample(Example):

    def __init__(self, arc_name: str | None = None, arc_group_and_index: List | None = None,
                 arc_data: List | Dict | None = None):
        """
        Generates an Example using specific data from an ARC task
        :param arc_name: The name of the task to be loaded
        :param arc_group_and_index: OR the group ('train' or 'eval') and the index of the task in that group
        :param arc_data: OR the actual data sets (a List of [name, task_data, solution_data] as returned from ld.from_json()
        """
        assert not(arc_name is None and arc_group_and_index is None and arc_data is None), \
            print(f'Making ARC Example. arc_name, arc_group_and_index and arc_data cannot all be None!')

        if arc_group_and_index is not None:
            group = arc_group_and_index[0]
            index = arc_group_and_index[1]
            challenges_names, challenges_tasks, solutions_tasks = from_json(group)
            self.name = challenges_names[index]
            self.solution_data = solutions_tasks[index]
            self.task_data = challenges_tasks[index]

        if arc_name is not None:
            train_challenges_names, train_challenges_tasks, train_solutions_tasks = ld.from_json('train')
            eval_challenges_names, eval_challenges_tasks, eval_solutions_tasks = ld.from_json('eval')

            for i, names in enumerate([train_challenges_names, eval_challenges_names]):
                if arc_name in names:
                    index = [k for k, j in enumerate(names) if j == arc_name][0]
                    task = [train_challenges_tasks, eval_challenges_tasks][i][index]
                    solution = [train_solutions_tasks, eval_solutions_tasks][i][index]
            try:
                self.name = arc_name
                self.task_data = task
                self.solution_data = solution
            except:
                print(f'No name {arc_name} in the ARC data set. The ARCExample Example will be empty.')

        if arc_data is not None:
            if type(arc_data) == list:
                self.name = arc_data[0]
                self.task_data = arc_data[1]
                self.solution_data = arc_data[2]
            elif type(arc_data) == dict:
                arc_data = deepcopy(arc_data)
                self.name = arc_data['name']
                self.task_data = {'test': [], 'train': []}
                for in_data, out_data in zip(arc_data['input'][:-1], arc_data['output'][:-1]):
                    in_data -= 1
                    out_data -= 1
                    self.task_data['train'].append({'input': [list(d.astype(int)) for d in in_data]})
                    self.task_data['train'][-1]['output'] = [list(d.astype(int)) for d in out_data]
                test_in = arc_data['input'][-1] - 1
                self.task_data['test'].append({'input': [list(d.astype(int)) for d in test_in]})
                test_out = arc_data['output'][-1] - 1
                self.solution_data = [[list(d.astype(int)) for d in test_out]]

        super().__init__(run_generate_canvasses=False)

        self.experiment_type = 'ARC'
        self.input_canvases_augmented = []
        self.output_canvases_augmented = []
        self.test_input_canvas_augmented = []
        self.test_output_canvas_augmented = []
        self.colour_mappings_for_augmentation = []
        self.number_of_rotations_for_augmentation = 1

    def generate_canvasses(self, empty: bool = True, augment_with: List[str] | None = None, max_samples: int = 10000,
                           with_black: bool = True):
        """
        Generate the ARC task Canvasses using the self.task_name, self.task_data and self.solution_data.
        :param with_black: If True augment the data by permuting also the black (0) colour.
        :param max_samples: The maximum sample to create if augmentation is on.
        :param empty: If empty is True then make the Canvasses the correct size but keep them empty (canvas.actual_pixels = 1).
        If False then copy onto the Canvasses the loaded data (this generates the correct looking Canvasses but they
        carry no Objects).
        :param augment_with: Choose what type of augmentation to apply to the data. List of string with possible values
         'colour', 'rotation', 'translation'
        :return:
        """
        self.number_of_io_pairs = len(self.task_data['train'])
        self.number_of_canvasses = self.number_of_io_pairs * 2 + 2

        for pair in range(self.number_of_io_pairs):
            input_data = np.flipud(np.array(self.task_data['train'][pair]['input']) + 1)
            output_data = np.flipud(np.array(self.task_data['train'][pair]['output']) + 1)
            if not empty:
                self.input_canvases.append(Canvas(actual_pixels=input_data, _id=pair * 2))
                self.output_canvases.append(Canvas(actual_pixels=output_data, _id=pair * 2 + 1))
            else:
                self.input_canvases.append(Canvas(size=Dimension2D(input_data.shape[1], input_data.shape[0]), _id=pair * 2))
                self.output_canvases.append(Canvas(size=Dimension2D(output_data.shape[1], output_data.shape[0]), _id=pair * 2 + 1))

        test_input_data = np.flipud(np.array(self.task_data['test'][0]['input']) + 1)
        test_output_data = np.flipud(np.array(self.solution_data[0]) + 1)
        if not empty:
            self.test_input_canvas = Canvas(actual_pixels=test_input_data, _id=2 * self.number_of_io_pairs)
            self.test_output_canvas = Canvas(actual_pixels=test_output_data, _id=2 * self.number_of_io_pairs + 1)
        else:
            self.test_input_canvas = Canvas(size=Dimension2D(test_input_data.shape[1], test_input_data.shape[0]),
                                            _id=2 * self.number_of_io_pairs)
            self.test_output_canvas = Canvas(size=Dimension2D(test_output_data.shape[1], test_output_data.shape[0]),
                                             _id=2 * self.number_of_io_pairs + 1)

        if augment_with is not None and not empty:
            self.generate_augmented_canvasses(augment_with, max_samples, with_black)

    def generate_augmented_canvasses(self, augment_with: List[str], max_samples: int = 10000, with_black: bool = True):

        if 'colour' in augment_with:
            used_colours = self.get_all_colours()
            self.colour_mappings_for_augmentation = utils.colours_permutations(used_colours, max_samples, with_black)
            for map in self.colour_mappings_for_augmentation:
                self.augment_with_colour(map)
        if 'rotation' in augment_with:
            self.augment_with_rotation()

    def augment_with_colour(self, colour_map: dict[int, int]):
        temp_inputs = []
        temp_outputs = []
        for i, o in zip(self.input_canvases, self.output_canvases):
            a = copy(i)
            a.swap_colours(colour_map)
            temp_inputs.append(a)
            b = copy(o)
            b.swap_colours(colour_map)
            temp_outputs.append(b)
        self.input_canvases_augmented.append(temp_inputs)
        self.output_canvases_augmented.append(temp_outputs)
        a = copy(self.test_input_canvas)
        a.swap_colours(colour_map)
        self.test_input_canvas_augmented.append(a)
        a = copy(self.test_output_canvas)
        a.swap_colours(colour_map)
        self.test_output_canvas_augmented.append(a)

    def augment_with_rotation(self):
        self.number_of_rotations_for_augmentation = 4
        j = 0
        while j < len(self.input_canvases_augmented):

            for k in range(3):
                p_in = []
                p_out = []
                for p in range(self.number_of_io_pairs):
                    ir = np.rot90(self.input_canvases_augmented[j + k][p].actual_pixels)
                    ir_c = Canvas(size=Dimension2D(const.MAX_PAD_SIZE, const.MAX_PAD_SIZE), actual_pixels=ir)
                    p_in.append(ir_c)

                    outr = np.rot90(self.output_canvases_augmented[j + k][p].actual_pixels)
                    outr_c = Canvas(size=Dimension2D(const.MAX_PAD_SIZE, const.MAX_PAD_SIZE), actual_pixels=outr)
                    p_out.append(outr_c)

                self.input_canvases_augmented.insert(j + k + 1, p_in)
                self.output_canvases_augmented.insert(j + k + 1, p_out)

                tir = np.rot90(self.test_input_canvas_augmented[j + k].actual_pixels)
                tir_c = Canvas(size=Dimension2D(const.MAX_PAD_SIZE, const.MAX_PAD_SIZE), actual_pixels=tir)
                self.test_input_canvas_augmented.insert(j + k + 1, tir_c)

                tor = np.rot90(self.test_output_canvas_augmented[j + k].actual_pixels)
                tor_c = Canvas(size=Dimension2D(const.MAX_PAD_SIZE, const.MAX_PAD_SIZE), actual_pixels=tor)
                self.test_output_canvas_augmented.insert(j + k + 1, tor_c)

            j += 4

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

    def show_augmented(self, index):
        temp_example = Example(run_generate_canvasses=False, number_of_io_pairs=self.number_of_io_pairs)
        temp_example.generate_canvasses()
        for i in range(self.number_of_io_pairs):
            temp_example.input_canvases[i] = self.input_canvases_augmented[index][i]
            temp_example.output_canvases[i] = self.output_canvases_augmented[index][i]
        temp_example.test_input_canvas = self.test_input_canvas_augmented[index]
        temp_example.test_output_canvas = self.test_output_canvas_augmented[index]

        temp_example.show()
