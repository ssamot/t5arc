
from data_generators.example_generator.auto_encoder_data_generator import AutoEncoderDataExample
from data_generators.example_generator.arc_data_generator import get_all_arc_data
from data_generators.example_generator.ttt_data_generator import ArcExampleData
from visualization.visualse_training_data_sets import visualise_training_data


# Usage of Random Data Generation for the auto-encoder training
e = AutoEncoderDataExample(number_of_canvases=20, percentage_of_arc_canvases=1, train_or_eval_arc='train')
array = e.get_canvases_as_numpy_array()
string = e.get_canvasses_as_string()

#  Save all canvasses as separate images (for crying out loud)
png_or_pdf = 'pdf'
for i, c in enumerate(e.input_canvases):
    file = fr'E:\tmp\canvasses\canvas_{i}.' + png_or_pdf
    c.show(save_as=file)

#  Show the whole example at once
e.show()

#  Or show just one Canvas
e.show(canvas_index=1)


# Usage of the ARC data generation for the auto-encoder training
arc_array = get_all_arc_data(group='train')  # group can be 'train' or 'eval'

# Usage of using the ARC data iterator
it = ArcExampleData('train')
result = []
for c in it:
    result.append(c)

# Visualise the results of the iterator
save = r'E:\tmp\canvasses\test.pdf'
visualise_training_data(result[0], save_as=save)
