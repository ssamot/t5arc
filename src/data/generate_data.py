import numpy as np
import keras
from models.tokenizer import CharacterTokenizer


class CanvasDataGenerator(keras.utils.PyDataset):
    def __init__(self, train_data, target_data,  batch_size, augment_fn,
                 shuffle=True, max_token_length = 300, repeats = 1,
                 **kwargs):
        """
        :param train_data: List of image paths or pre-loaded images.
        :param batch_size: Size of each batch.
        :param augment_fn: Function to augment images.
        :param shuffle: Whether to shuffle the order of images after each epoch.
        """
        super().__init__(**kwargs)
        self.train_data = train_data
        self.target_data = target_data
        self.batch_size = batch_size
        self.augment_fn = augment_fn
        self.shuffle = shuffle
        self.repeats = repeats

        self.chars = set("".join(target_data))
        indices = [i for i, sublist in enumerate(self.target_data) if len(sublist) < max_token_length]
        self.train_data = [self.train_data[i] for i in indices]

        self.num_decoder_tokens = len(self.chars) + 10

        print("num_decoder_tokens", self.num_decoder_tokens)

        tokenizer = CharacterTokenizer(self.chars, 20000000)
        tokenized_inputs = tokenizer(
            [self.target_data[i] for i in indices],
            padding="longest",
            truncation=True,
            return_tensors="np",
        )
        self.tokenized_inputs = tokenized_inputs
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.indices) / self.batch_size))

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

    def __data_generation(self, indices):
        """Generates data containing batch_size images."""
        # Initialization
        batch_targets = self.tokenized_inputs.input_ids[indices]
        batch_inputs = []

        # Generate data
        for i in indices:
            padded = []
            for j in range(len(self.train_data[i])):
                p = self.augment_fn(np.array(self.train_data[i][j]))
                padded.append(p)
            batch_inputs.append(padded)


        #print(len(self.train_data[0][2]))
        #exit()
        #batch_inputs = [self.augment_fn(self.train_data[i]) for i in indices]

        batch_inputs = np.array(batch_inputs)
        batch_inputs = batch_inputs[:, :, np.newaxis, :, :]
        batch_inputs = np.transpose(batch_inputs, (1, 0, 3, 4, 2))
        #print(batch_inputs.shape)
        #exit()

        batch_inputs = [np.array(b, dtype="int") for b in batch_inputs]
        # for b in batch_inputs:
        #     print(np.max(b), np.min(b))
        # exit()


        one_hot_encoded = np.zeros(
            (batch_targets.shape[0], batch_targets.shape[1], self.num_decoder_tokens))
        rows = np.arange(batch_targets.shape[0])[:, None]
        cols = np.arange(batch_targets.shape[1])
        one_hot_encoded[rows, cols, batch_targets] = 1

        target_texts = np.zeros_like(batch_targets)
        target_texts[:, :-1] = batch_targets[:, 1:]

        targets_one_hot_encoded = one_hot_encoded[:, 1:, :]
        targets_inputs = batch_targets[:, :-1]


        return batch_inputs + [targets_inputs], targets_one_hot_encoded

