import keras
import numpy as np

from data.utils import load_data
from data_generators.object_recognition.random_objects_example import RandomObjectsExample
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

        json_data_list = []
        objects = []
        object_pixels = []
        for _ in (range(self.batch_size)):
            # Create an Example
            e = RandomObjectsExample()
            e.randomly_populate_canvases()
            arc_style_input = e.create_canvas_arrays_input()
            unique_objects, actual_pixels_array, positions_of_same_objects = e.create_output()
            json_data_list.append(arc_style_input)
            objects.append(str(unique_objects).replace(" ", ""))
            object_pixels.append(actual_pixels_array)

        batch_inputs, test = load_data(json_data_list)
        # print(objects)

        # print(train.shape)
        # print(objects[-1])

        tokenized_inputs = self.tokenizer(
            objects,
            padding="longest",
            truncation=True,
            return_tensors="np",
        )

        batch_targets = tokenized_inputs.input_ids
        batch_inputs = np.transpose(batch_inputs, (1, 0, 2, 3, 4))

        # print(batch_inputs.shape)
        # print(batch_targets.shape)

        batch_inputs = [np.array(b, dtype="int") for b in batch_inputs]
        # for b in batch_inputs:
        #     print(np.max(b), np.min(b))
        # exit()

        one_hot_encoded = np.zeros(
            (batch_targets.shape[0], batch_targets.shape[1], self.tokenizer.num_decoder_tokens + 1))
        rows = np.arange(batch_targets.shape[0])[:, None]
        cols = np.arange(batch_targets.shape[1])
        one_hot_encoded[rows, cols, batch_targets] = 1

        target_texts = np.zeros_like(batch_targets)
        target_texts[:, :-1] = batch_targets[:, 1:]

        targets_one_hot_encoded = one_hot_encoded[:, 1:, :]
        targets_inputs = batch_targets[:, :-1]

        return batch_inputs + [targets_inputs], targets_one_hot_encoded
