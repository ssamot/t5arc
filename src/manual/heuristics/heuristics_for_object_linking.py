
from abc import ABC, abstractmethod
import numpy as np

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

