
from abc import ABC, abstractmethod

import networkx as nx
from typing import List
from data.generators.object_recognition.object import Object
from data.generators.object_recognition.primitives import Predefined
from data.generators.object_recognition.canvas import Canvas
from data.generators.task_generator.task import Task


def get_manual_object_linker_subclass_by_name(manual_linker_name: str):
    return [cls for cls in ManualObjectLinker.__subclasses__() if cls.__name__ == manual_linker_name][0]


class ManualObjectLinker(ABC):

    def __init__(self, task: Task):

        self.task = task
        self.detected_objects_per_canvas = {}

        for i in task.canvas_ids:
            self.detected_objects_per_canvas[i] = []

    @abstractmethod
    def run_linker(self):
        pass


class MatchWithAffineLinker(ManualObjectLinker):

    def __init__(self, task: Task):
        super().__init__(task=task)
        self.min_obj_dimension = 3
        self.match_shape_only = True

    def do_a_match(self, input_object:Object, out_pixels_object: Object,
                   out_canvas: Canvas, example_graph: nx.Graph, transformations: List[str]):
        best_matches = \
            input_object.match_to_background(out_pixels_object, try_unique=True, transformations=transformations,
                                             match_shape_only=self.match_shape_only)
        num_of_matched_objects = 0
        # print('-----------------')
        # print(f'Input object at {input_object.canvas_pos}')
        for bm in best_matches:
            all_pos = bm['canvas_pos']
            for pos in all_pos:
                output_object = out_canvas.find_object_at_canvas_pos(pos)
                # print(f'Output object at {output_object.canvas_pos}')
                if output_object is not None:
                    example_graph.add_edge(input_object, output_object)
                    num_of_matched_objects += 1

        if num_of_matched_objects > 0:
            return True
        else:
            return False

    def run_linker(self):
        task = self.task

        # For every example
        for p in range(task.number_of_io_pairs):
            in_canvas = task.input_canvases[p]
            out_canvas = task.output_canvases[p]
            out_pixels_object = Predefined(actual_pixels=out_canvas.actual_pixels)

            example_graph = task.objects_transformations_in_example_graphs[p]
            # For every object in the input canvas
            for input_object in in_canvas.objects:
                if input_object.dimensions.dx >= self.min_obj_dimension and input_object.dimensions.dy >= self.min_obj_dimension:
                    # match that object to the whole of the output canvas

                    transformations = ['rotate', 'scale', 'flip', 'invert']
                    if not self.do_a_match(input_object=input_object, out_pixels_object=out_pixels_object,
                                           out_canvas=out_canvas, example_graph=example_graph,
                                           transformations=transformations):
                        transformations = ['rotate', 'scale', 'flip']
                        self.do_a_match(input_object=input_object, out_pixels_object=out_pixels_object,
                                        out_canvas=out_canvas, example_graph=example_graph,
                                        transformations=transformations)








