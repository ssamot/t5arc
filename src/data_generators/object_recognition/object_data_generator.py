
import numpy as np
import constants as const

MAX_PAD_SIZE = const.MAX_PAD_SIZE


num_of_images = np.random.choice(np.arange(3, 2 * const.MAX_EXAMPLE_PAIRS + 2, 2))


def embed_pic_in_image(pic: np.ndarray):
    image = np.zeros((MAX_PAD_SIZE, MAX_PAD_SIZE, 1))
    [x, y] = pic.shape

    x_gap = MAX_PAD_SIZE - x
    y_gap = MAX_PAD_SIZE - y

    x0 = np.random.randint(0, x_gap)
    y0 = np.random.randint(0, y_gap)

    image[x0:x0+x, y0:y0+y, 0] = pic

    return image


