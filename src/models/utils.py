import numpy as np
import keras



import keras

class CustomModelCheckpoint(keras.callbacks.Callback):

    def __init__(self, models, path):
        self.models = models
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        print(f"epoch: {epoch}, train_acc: {logs['acc']}, : {logs['acc_seq']}")
        for name in self.models:
            model = self.models[name]
            model.save(f"{self.path}/{name}.keras", overwrite=True)


class ConcatenateRNN(keras.layers.Layer):
    def call(self, inputs):
        dec_embeddings, encoded = inputs

        # Step 2: Tile x to match the second dimension of y
        x_tiled = keras.ops.tile(encoded, [1, keras.ops.shape(dec_embeddings)[1], 1])  # Shape: (None, None, 640)

        # Step 3: Concatenate y and x_tiled along the last axis
        output = keras.ops.concatenate([dec_embeddings, x_tiled], axis=-1)  # Shape: (None, None, 672)

        return output


