
from abc import ABC, abstractmethod

from data.generators.task_generator.task import Task


def get_manual_object_detector_subclass_by_name(manual_detector_name: str):
    return [cls for cls in ManualObjectDetector.__subclasses__() if cls.__name__ == manual_detector_name][0]


class ManualObjectDetector(ABC):

    def __init__(self, task: Task):

        self.task = task
        self.detected_objects_per_canvas = {}

        for i in task.canvas_ids:
            self.detected_objects_per_canvas[i] = []

    @abstractmethod
    def run_detection(self):
        pass

    def embed_objects_in_canvasses(self):
        for i in self.detected_objects_per_canvas:
            self.task.get_canvas_by_id(i).objects = self.detected_objects_per_canvas[i]


class SameColourConnectedPixels(ManualObjectDetector):

    def __init__(self, task: Task):
        super().__init__(task=task)

    def run_detection(self):

        for i in range(self.task.number_of_canvasses - 1):
            base_obj = self.task.get_canvas_by_id(i).objects[0]
            colours = base_obj.get_used_colours()

            all_objects = []
            for colour in colours:
                one_colour_objects = base_obj.create_new_primitives_from_pixels_of_colour(colour)

                for oco in one_colour_objects:
                    all_objects.append(oco)

            self.detected_objects_per_canvas[i] = all_objects

