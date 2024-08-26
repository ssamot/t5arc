
import keras
from keras import layers


activation = "relu"

def build_model(input_shape, num_decoder_tokens, encoder_units):
    total_features = input_shape[0] * input_shape[1]



    input_img = keras.Input(shape=input_shape)


    x = keras.layers.Reshape(input_shape, input_shape=total_features)(input_img)
    x = keras.layers.Embedding(input_dim=11, output_dim=4, input_length=total_features)(x)
    x = keras.layers.Flatten()(x)

    #x = keras.layers.Flatten()(input_img)

    n_neurons = 128
    x = keras.layers.Dense(n_neurons, activation=activation)(x)
    x = keras.layers.LayerNormalization()(x)
    xs = [x]
    for i in range(3):
        # skip connections
        x_new = keras.layers.Dense(n_neurons, activation=activation)(x)
        x_new = keras.layers.LayerNormalization()(x_new)
        xs.append(x_new)
        x = keras.layers.add(xs)

    #encoded = layers.Reshape([4,4,8])(x)


    encoded = keras.layers.LayerNormalization()(
        layers.Dense(encoder_units, activation="tanh")(x))

    ## decoder
    decoded_inputs = keras.Input(shape=(encoder_units,))
    # decoded = keras.layers.LayerNormalization()(
    #     layers.Dense(n_neurons,activation="relu")(decoded_inputs))

    decoded = layers.Dense(total_features*num_decoder_tokens)(decoded_inputs)
    decoded = layers.Reshape([input_shape[0],input_shape[1],num_decoder_tokens])(decoded)
    decoded = layers.Activation("softmax")(decoded)


    encoder = keras.Model(input_img, encoded)
    decoder = keras.Model(decoded_inputs,  decoded)



    autoencoder = keras.Model(input_img, decoder(encoded))
    #optimizer = keras.optimizers.SGD(momentum=0.3, weight_decay=0.01, nesterov=True, learning_rate=0.1)
    optimizer = keras.optimizers.AdamW()
    autoencoder.compile(optimizer=optimizer,

                        loss='categorical_crossentropy',
                        metrics=["acc",acc_seq])

    return autoencoder, encoder, decoder


class CustomModelCheckpoint(keras.callbacks.Callback):

    def __init__(self, models, path, saver_freq):
        self.models = models
        self.path = path
        self.save_freq = saver_freq

    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        if(epoch % self.save_freq == 0):
            print(f"Saving epoch: {epoch}, train_acc: {logs['acc']}, : {logs['acc_seq']}")
            for name in self.models:
                model = self.models[name]
                model.save(f"{self.path}/{name}.keras", overwrite=True)


def acc_seq(y_true, y_pred):
    return keras.ops.mean(keras.ops.min(keras.ops.equal(keras.ops.argmax(y_true, axis=-1),
                  keras.ops.argmax(y_pred, axis=-1)), axis=-1))

