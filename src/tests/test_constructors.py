
import pytest
import matplotlib.pyplot as plt
import numpy as np
from data_generators.object_recognition import object_data_generator as odg
import constants as const


class Test:

    def test_pic_embedding_in_image(self):
        show_two_examples = 0
        for x in range(const.MAX_PAD_SIZE - 3):
            for y in range(const.MAX_PAD_SIZE -3):
                pic = np.ones((x, y))
                image = odg.embed_pic_in_image(pic)
                if image.sum() != x*y:
                    plt.imshow(image)
                    plt.show()
                    show_two_examples += 1
                if show_two_examples < 2 and x>10 and y>10:
                    plt.imshow(image)
                    plt.show()
                    show_two_examples += 1
                assert image.sum() == x * y

