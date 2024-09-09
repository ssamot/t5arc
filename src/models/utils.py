
import keras
from keras import layers
from lm import b_acc, cce

activation = "relu"

def build_NMF(n_image_embeddings, n_programme_embeddings, n_programmes,
              input_shape):
    total_features = input_shape[0] * input_shape[1] * input_shape[2]

    input_img = keras.Input(shape=input_shape, name="input_images")

    x = keras.layers.Flatten()(input_img)

    n_neurons = 256
    x = keras.layers.Dense(n_neurons, activation=activation, )(x)
    x = keras.layers.LayerNormalization()(x)

    xs = [x]
    for i in range(7):
        # skip connections
        x_new = keras.layers.Dense(n_neurons, activation=activation,
                                   )(x)
        x_new = keras.layers.LayerNormalization()(x_new)

        xs.append(x_new)
        x = keras.layers.add(xs)

    encoded = keras.layers.Dense(n_image_embeddings, activation="tanh",
                                 )(x)
    encoded = keras.layers.LayerNormalization()(encoded)


    input_programme = keras.Input((1,1), name = "input_programmes")
    input_decoder = keras.Input((n_image_embeddings,))

    encoder = keras.Model(input_img, encoded, name = "encoded")
    #####

    ttt_emb = layers.Flatten()(layers.Embedding(n_programmes,
                                                      n_programme_embeddings,
                                                      name = "embeddings_programmes",
                                                embeddings_regularizer="l2",
                                                embeddings_constraint=keras.constraints.NonNeg())(input_programme))

    programme_emb_input = keras.Input((n_programme_embeddings,))

    ttt = keras.Model(input_programme, ttt_emb, name = "ttt")
    x = keras.layers.concatenate([input_decoder, programme_emb_input])
    x = keras.layers.Dense(n_neurons, activation=activation, )(x)

    xs = [x]
    for i in range(7):
        # skip connections
        x_new = keras.layers.Dense(n_neurons, activation=activation,
                                   )(x)
        x_new = keras.layers.LayerNormalization()(x_new)

        xs.append(x_new)
        x = keras.layers.add(xs)

    decoded = layers.Dense(total_features, name = "dense_decoded")(x)
    decoded = layers.Reshape(input_shape)(decoded)
    decoded = layers.Activation("softmax")(decoded)
    decoder = keras.Model([input_decoder, programme_emb_input],
                          decoded,
                          name = "decoder")
    ######
    autoenc_input_img = keras.Input(shape=input_shape, name="input_images_auto")
    autoenc_input_programme = keras.Input((1, 1))
    autoencoder = keras.Model([autoenc_input_img, autoenc_input_programme],
                              decoder([encoder(autoenc_input_img),
                                       ttt(autoenc_input_programme)]) ,
                              name = "autoencoder")

    optimizer = keras.optimizers.AdamW(learning_rate=0.001)
    #optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.8, nesterov=True)

    autoencoder.compile(optimizer=optimizer,

                    loss='categorical_crossentropy',
                    metrics=["acc", b_acc, cce])

    return encoder, decoder, autoencoder, ttt


class CustomModelCheckpoint(keras.callbacks.Callback):

    def __init__(self, models, path, saver_freq):
        self.models = models
        self.path = path
        self.save_freq = saver_freq

    def on_epoch_end(self, epoch, logs=None):
        # logs is a dictionary
        if(epoch % self.save_freq == 0):
            #print(f"Saving epoch: {epoch}, train_acc: {logs['acc']}, : {logs['batch_acc']}")
            for name in self.models:
                model = self.models[name]
                model.save(f"{self.path}/{name}.keras", overwrite=True)





@keras.saving.register_keras_serializable(package="models")
class BinaryDense(keras.layers.Layer):
    def __init__(self, units, kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        super(BinaryDense, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(self.input_dim, self.units),
            initializer=self.kernel_initializer,
            trainable=True,
            name='kernel'
        )

        super(BinaryDense, self).build(input_shape)

    def call(self, inputs):
        # Binarize the kernel weights
        #kernel_binarized = keras.ops.sign(self.kernel)

        # Perform the matrix multiplication
        output = keras.ops.sign(keras.ops.matmul(inputs, self.kernel))



        return output

    def get_config(self):
        config = super(BinaryDense, self).get_config()
        config.update({
            'units': self.units,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
        })
        return config

# class OrthogonalConstraint(keras.constraints.Constraint):
#     def __call__(self, w):
#         #print(w.shape)
#         #exit()
#         # Perform Singular Value Decomposition
#         u, _, v = keras.ops.linalg.svd(w)
#         # Reconstruct the nearest orthogonal matrix
#         return keras.ops.matmul(u, keras.ops.transpose(v))

def average_maps(*maps):
    # Initialize an empty dictionary to store the averages
    averaged_map = {}

    # Iterate over the keys of the first map (assuming all maps have the same keys)
    for key in maps[0]:
        # Sum the values for the current key from all maps
        total = sum(map[key] for map in maps)
        # Calculate the average
        average = total / len(maps)
        # Store the average in the new map
        averaged_map[key] = average

    return averaged_map


import numpy as np


class NNWeightHelper:
    def __init__(self, model):
        self.model = model
        self.init_weights = self.model.get_weights()


    def _set_trainable_weight(self, model, weights):
        """Sets the weights of the model.

        # Arguments
            model: a keras neural network model
            weights: A list of Numpy arrays with shapes and types matching
                the output of `model.trainable_weights`.
        """

        # for sw, w in zip(layer.trainable_weights, weights):
        #      tuples.append((sw, w))


        model.set_weights(tuples)


    def set_weights(self, weights):
        tuples = []

        for w in self.init_weights:
            num_param = w.size

            layer_weights = weights[:num_param]
            new_w = np.array(layer_weights).reshape(w.shape)
            #print(new_w.shape)

            tuples.append(new_w)
            weights = weights[num_param:]

        self.model.set_weights(tuples)




    def get_weights(self):
        W_list = (self.model.trainable_weights)
        W_flattened_list = [np.array(k).flatten() for k in W_list]
        W = np.concatenate(W_flattened_list)

        return W