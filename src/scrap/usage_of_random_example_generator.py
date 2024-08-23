
import numpy as np
from data_generators.example_generator.random_objects_example import RandomObjectsExample
np.random.seed(11)

#  How many, different, Primitives are allowed in the Example. These will be the base Objects
MAX_NUM_OF_DIFFERENT_PRIMITIVES = 4

# This is the maximum number of transformed Objects each base Object is allowed to generate if the base object is LARGE
# (bigger than LARGE_OBJECT_THRESHOLD in both size dimensions)
MAX_NUM_OF_LARGE_OBJECTS = 3

# These define the minimum and maximum number of transformed Objects each base Object is allowed to have transformations
# of if the base Object is small (smaller than LARGE_OBJECT_THRESHOLD)
MIN_NUM_OF_SMALL_OBJECTS = 2
MAX_NUM_OF_SMALL_OBJECTS = 6

# The number of transformations every base Object can have in order to create a transformed Object.
NUMBER_OF_TRANSFORMATIONS = 2

# How many possible copies of the exact same transformed Object can coexist on the same Canvas
MAX_NUM_OF_SAME_OBJECTS_ON_CANVAS = 2

# Each Canvas can have MAX_NUM_OF_SAME_OBJECTS_ON_CANVAS of each transformed Object and the same transformed Object can
# appear in multiple Canvases.

for i in range(1):
    # Create an Example
    print(i)

    e = RandomObjectsExample()

    # Populate the Canvases of the Example with Objects drawn randomly
    e.randomly_populate_canvases()

    # Generate the Input arrays. These are a dictionary with the same structure as the ARC examples
    arc_style_input = e.create_canvas_arrays_input()

    # Generate the Output info
    # unique_objects is a list of dictionaries, each describing one unique transformed Object in the Example
    # actual_pixels_array is a 3D numpy array (MAX_PAD_SIZE, MAX_PAD_SIZE, len(unique_objects)) where each
    # (MAX_PAD_SIZE, MAX_PAD_SIZE) 2D slice has the actual pixels of a transformed Object. The actual_pixels_id of the
    # transformed Object (found in the unique_objects representation) is the index of the actual_pixels slice in the 3D
    # array.
    # The positions_of_same_objects is a dict with keys the tuple (object.id, object.actual_pixels_id) for each
    # transformed Object and value a List of all the [Canvas id, [canvas_pos.x, canvas_pos.y]] lists that show on which
    # Canvases and on which locations the transformed Object appears.
    unique_objects, actual_pixels_array = e.create_output()


# VISUALISATION
# To show the whole example do
e.show()

# To show a specific canvas (e.g. the 1st input canvas) do
e.input_canvases[0].show()

# One can access the individual objects either from the Example directly
len(e.objects)
e.objects[5].show()

# or from a specific Canvas
len(e.output_canvases[0].objects)
e.output_canvases[0].objects[0].show()
