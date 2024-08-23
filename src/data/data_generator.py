import keras
import numpy as np


from data_generators.example_generator.auto_encoder_data_generator import AutoEncoderDataExample
from models.tokenizer import CharacterTokenizer
from models.tokens import token_list


class CanvasDataGenerator(keras.utils.PyDataset):
    def __init__(self, batch_size, len=10, repeats=10,
                 **kwargs):
        """
        :param train_data: List of image paths or pre-loaded images.
        :param batch_size: Size of each batch.
        :param augment_fn: Function to augment images.
        :param shuffle: Whether to shuffle the order of images after each epoch.
        """
        super().__init__(**kwargs)

        self.batch_size = batch_size
        self.len = len

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return self.len

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate data
        X = self.__data_generation()

        return X

    def on_epoch_end(self):
        """Updates indices after each epoch."""
        pass

    def __data_generation(self):
        """Generates data containing batch_size images."""

        e = AutoEncoderDataExample(self.batch_size)
        batch_targets = e.get_canvases_as_numpy_array()

        one_hot_encoded = np.eye(11)[np.array(batch_targets, dtype=np.int32)]



        return batch_targets[:,:,:,np.newaxis]/11.0,  one_hot_encoded
