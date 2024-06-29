
Formatting for object detection DataConstructor
----------------------------------------------

#. Add 1 to all colours (so that padding is 0)
#. Image is numpy (32x32x1) and add actual picture randomly in it
#. Check how many pairs of images are in each example
#. Format Input: list([np.array(samples, x, y, 1), ...]) for 2*number of pairs +1 numpy arrays. The order is Input0, Output0, Input1, Output1,...,TestInput, TestOutput (TestOutput is empty)
#. Format Output:
    #. Make two hash tables for dictionary of tokens (string tokens <-> integers). 0 is masking. Start token, end token must exist.
    #. The output is either a string or a list of integer
