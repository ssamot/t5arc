import keras
import numpy as np
#import os


def generate_random_arrays(n):
    array1 = np.random.rand(n, 32, 32, 11)
    array2 = np.random.rand(n, 32, 32, 11)
    return array1, array2

def merge_arrays(array1, array2):
    return np.concatenate((array1, array2), axis=-1)


class BatchedDataGenerator(keras.utils.PyDataset):
    def __init__(self, data_array, **kwargs):
        super().__init__(**kwargs)
        self.data_array = data_array

    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, idx):
        original_images, new_images = self.data_array[idx]
        s = original_images

        ssprime = merge_arrays(original_images, new_images)
        return (s, ssprime), s

    def on_epoch_end(self):
        # Shuffle the data at the end of each epoch if needed
        np.random.shuffle(self.data_array)


class DataGenerator(keras.utils.PyDataset):
    def __init__(self, len=10,
                 **kwargs):
        """
        :param train_data: List of image paths or pre-loaded images.
        :param batch_size: Size of each batch.
        :param augment_fn: Function to augment images.
        :param shuffle: Whether to shuffle the order of images after each epoch.
        """
        super().__init__(**kwargs)

        self.len = len


        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return self.len

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate data
        original_images, new_images = generate_random_arrays(5)
        s = original_images
        ssprime = merge_arrays(original_images, new_images)
        #print(ssprime.shape)
        #exit()



        return (s,ssprime), s

    def on_epoch_end(self):
        """Updates indices after each epoch."""
        pass

