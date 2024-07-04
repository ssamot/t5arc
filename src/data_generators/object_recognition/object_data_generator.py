
import numpy as np
import constants as const
from typing import Union
import skimage
from data_generators.object_recognition.object_recognition_output_dict import *


np.random.seed(const.RANDOM_SEED_FOR_NUMPY)
MAX_PAD_SIZE = const.MAX_PAD_SIZE


def get_number_of_canvases_in_example():
    #TODO: Think about the possibility of using some distribution other than uniform
    return np.random.choice(np.arange(4, 2 * const.MAX_EXAMPLE_PAIRS + 3, 2))


def embed_image_in_canvas(image: np.ndarray) -> np.ndarray:
    """
    Canvas (32,32,1) is 0
    :param image: Image has colours between 1 and 10
    :return: The canvas with the image in it
    """
    canvas = np.ones((MAX_PAD_SIZE, MAX_PAD_SIZE, 1))

    return random_embed_pic_in_image(canvas, image)


def embed_pic_in_image(image: np.ndarray, pic: np.ndarray, x: int, y: int) -> np.ndarray:
    """
    Embeds a smaller pic in an image at x, y coordinates (from top left)
    :param image: The background image
    :param pic: The embedded pic
    :param x: left
    :param y: top
    :return: The final image
    """
    [y_pic, x_pic] = pic.shape

    image[y:y+y_pic, x:x+x_pic, 0] = pic

    return image


def random_embed_pic_in_image(image: np.ndarray, pic: np.ndarray) -> np.ndarray:
    """
    Embeds a pic in a larger image at a random location
    :param image: The background image
    :param pic: The embedded pic
    :return: The final image
    """
    [y, x] = pic.shape
    x_gap = MAX_PAD_SIZE - x
    y_gap = MAX_PAD_SIZE - y

    x0 = np.random.randint(0, x_gap)
    y0 = np.random.randint(0, y_gap)

    return embed_pic_in_image(image, pic, x0, y0)


def translate(image: np.ndarray, dx: int, dy: int):
    """
    Translate the whole image by dx x dy
    :param image: The original image
    :param dx: translation along the x axes
    :param dy: translation along the y axes
    :return: The final image
    """
    assert dx != 0 or dy != 0, print('At least one of x or y needs to be not zero')

    result = np.ones(image.shape)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if 0 <= i - dy < result.shape[0] and 0 <= j - dx < result.shape[1]:
                result[i, j] = image[i - dy, j - dx]

    return result


def generate_random_example(seed=0):
    num_of_canvases = get_number_of_canvases_in_example()

    number_of_objects = np.random.randint(1, 5)
    object_bbox_sizes = np.random.randint(1, 10, (number_of_objects, 2))
    input_size = np.random.randint(3, 32)

    
