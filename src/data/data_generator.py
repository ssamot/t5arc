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
        self.tokenizer = CharacterTokenizer(token_list, 2000000000)

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
        array_reps = e.get_canvases_as_numpy_array()
        str_reps = e.get_canvasses_as_string()

        # print(train.shape)
        # print(objects[-1])

        tokenized_inputs = self.tokenizer(
            str_reps,
            padding="longest",
            truncation=True,
            return_tensors="np",
        )

        batch_targets = tokenized_inputs.input_ids


        one_hot_encoded = np.zeros(
            (batch_targets.shape[0], batch_targets.shape[1], self.tokenizer.num_decoder_tokens + 1))
        rows = np.arange(batch_targets.shape[0])[:, None]
        cols = np.arange(batch_targets.shape[1])
        one_hot_encoded[rows, cols, batch_targets] = 1

        target_texts = np.zeros_like(batch_targets)
        target_texts[:, :-1] = batch_targets[:, 1:]

        targets_one_hot_encoded = one_hot_encoded[:, 1:, :]
        targets_inputs = batch_targets[:, :-1]

        return [array_reps,targets_inputs], targets_one_hot_encoded
