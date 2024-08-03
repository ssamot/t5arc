
import numpy as np
import keras
from models.tokenizer import CharacterTokenizer

class ExampleDataGenerator(keras.utils.PyDataset):

    def __init__(self, batch_size: int = 100, epoch_size: int = 100, repeats: int = 1):
        self.batch_size = batch_size
        self.repeats = repeats
        self.epoch_size = epoch_size

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(self.epoch_size / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indices of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X = self.__data_generation(indices)

        return X

    def on_epoch_end(self):
        """Updates indices after each epoch."""
        self.indices = [np.arange(len(self.train_data)) for i in range(self.repeats)]
        self.indices = np.concatenate(self.indices)

        if self.shuffle:
            np.random.shuffle(self.indices)