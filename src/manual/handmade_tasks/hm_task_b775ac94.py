import networkx as nx
from matplotlib import pyplot as plt

from data.generators.task_generator.arc_task_generator import ARCTask


task = ARCTask('b775ac94')
#task = ARCTask('00d62c1b')
task.generate_canvasses()

task.generate_objects_from_heuristic(manual_detector_name='SameColourConnectedPixels')
task.link_detected_objects_with_heuristic(manual_linker_name='MatchWithAffineLinker')

task.show()

f = plt.figure()

for p in range(task.number_of_io_pairs):
    ax = f.add_subplot(1, 3, p+1, frame_on=True)
    nx.draw(task.objects_transformations_in_example_graphs[p], ax=ax, node_size=50, node_color=[(0, p/2, p/3)],
            hide_ticks=False, margins=0.1)


