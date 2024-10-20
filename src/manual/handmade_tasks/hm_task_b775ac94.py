
from data.generators.task_generator.arc_task_generator import ARCTask


task = ARCTask('b775ac94')
#task = ARCTask('00d62c1b')
task.generate_canvasses()

task.generate_objects_from_data(manual_detector_name='SameColourConnectedPixels')
task.populate_object_transformations_graphs_with_nodes()

task.show()


