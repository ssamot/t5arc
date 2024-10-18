

from abc import ABC, abstractmethod
import numpy as np

from data.generators.object_recognition.primitives import Primitive, Predefined
from data.generators.task_generator.task import Task


class ObjectDetector(ABC):

    def __init__(self, task: Task):

        self.task = task
        self.detected_objects_per_canvas = {}

        for i in task.canvas_ids:
            self.detected_objects_per_canvas[i] = []

    @abstractmethod
    def detector(self):
        pass

    def embed_objects_in_canvasses(self):
        for i in self.detected_objects_per_canvas:
            self.task.get_canvas_by_id(i).objects = self.detected_objects_per_canvas[i]


class SameColourConnectedPixels(ObjectDetector):

    def __init__(self, task: Task):
        super().__init__(task=task)

    def detector(self):

        for i in range(self.task.number_of_canvasses):
            base_obj = Predefined(actual_pixels=self.task.get_canvas_by_id(i).actual_pixels)

            colours = base_obj.get_used_colours()

            all_objects = []
            for colour in colours:
                one_colour_objects = base_obj.create_new_primitives_from_pixels_of_colour(colour)

                for oco in one_colour_objects:
                    all_objects.append(oco)

            self.detected_objects_per_canvas[i] = all_objects

        self.embed_objects_in_canvasses()

