
import matplotlib.pyplot as plt
import numpy as np
from data_generators.object_recognition import object_data_generator as odg
import constants as const


class TestConstructors:

    with_visual_check = True

    def test_pic_embedding_in_canvas(self):
        show_two_examples = 0
        for x in range(1, const.MAX_PAD_SIZE - 3):
            for y in range(1, const.MAX_PAD_SIZE - 3):
                image = np.ones((x, y))*2
                canvas = odg.embed_image_in_canvas(image)
                if self.with_visual_check:
                    if image.sum() > x*y*4:
                        plt.imshow(canvas)
                        plt.show()
                        show_two_examples += 1
                    if show_two_examples < 2 and x>10 and y>10:
                        plt.imshow(canvas)
                        plt.show()
                        show_two_examples += 1
                assert canvas.sum() == x * y + canvas.shape[0] * canvas.shape[1]
                assert len(canvas.shape) == 3
                assert canvas.shape[2] == 1

    def test_scale_pic(self):
        pic1 = np.array([[1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 2, 3, 4, 1, 1],
                        [1, 1, 2, 3, 4, 1, 1],
                        [1, 1, 2, 3, 4, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1]])

        pic2 = np.array([[1, 1, 1, 1, 1, 1, 1],
                         [1, 2, 3, 4, 1, 1, 1],
                         [1, 1, 2, 3, 4, 1, 1],
                         [1, 1, 1, 2, 3, 4, 1],
                         [1, 1, 1, 1, 2, 3, 1],
                         [1, 1, 1, 1, 1, 2, 1],
                         [1, 1, 1, 1, 1, 1, 1]])
        pics = [pic1, pic2]

        for i, pic in enumerate(pics):
            for factor in list(range(2, 5)):
                scaled_up = odg.scale(pic, factor)
                assert factor**2 * pic.sum() == scaled_up.sum()
                scaled_back_down = odg.scale(scaled_up, -factor)
                assert np.all(pic == scaled_back_down)

                if self.with_visual_check:
                    if i == 1 and factor == 4:
                        fig = plt.figure(1)
                        ax_pic = fig.add_subplot(1, 3, 1)
                        ax_pic.imshow(pic)
                        ax_su = fig.add_subplot(1, 3, 2)
                        ax_su.imshow(scaled_up)
                        ax_sd = fig.add_subplot(1, 3, 3)
                        ax_sd.imshow(scaled_back_down)
                        plt.show()

