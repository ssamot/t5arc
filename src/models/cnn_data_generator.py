import keras
import numpy as np
import os
import time
from data.generators.task_generator.random_transformations_task import RandomTransformationsTask


def generate_random_arrays(n):
    array1 = np.random.rand(n, 32, 32, 11)
    array2 = np.random.rand(n, 32, 32, 11)
    return array1, array2


def generate_samples(one_to_one):
    pid = os.getpid()

    # Check if this process has already executed the command
    if not hasattr(generate_samples, f'executed_{pid}'):
        seed = (os.getpid() * int(time.time())) % 123456789
        #print(f"Setting seed for process {pid}, {seed}")
        # Your command here
        np.random.seed(seed)

        # Mark this process as having executed the command
        setattr(generate_samples, f'executed_{pid}', True)
    else:
        pass


    n = np.random.randint(4, 11)  # Randomly choose n between 1 and 10

    #x = np.random.randn(n, 32, 32, 11)  # Generate random array for x
    #y = np.random.randn(n, 32, 32, 11)  # Generate random array for y

    t = RandomTransformationsTask(num_of_outputs=n, one_to_one = one_to_one)
    t.generate_samples()
    x, y = t.get_cnavasses_as_arrays()

    return (x, y)

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








