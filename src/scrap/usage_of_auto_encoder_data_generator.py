
from data_generators.example_generator.auto_encoder_data_generator import AutoEncoderDataExample

e = AutoEncoderDataExample(number_of_canvases=20)
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