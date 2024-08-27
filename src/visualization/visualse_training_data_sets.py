

import matplotlib.pyplot as plt
from typing import Dict
from data_generators.example_generator.arc_example_generator import ARCExample


def visualise_training_data(data: Dict, save_as: str | None = None):
    example = ARCExample(arc_data=data)
    example.generate_canvasses(empty=False)
    example.show(two_cols=True, save_as=save_as)
    plt.tight_layout(rect=(1.0, 0.015, 0.01, 0.99), h_pad=0.088, w_pad=0.0)
