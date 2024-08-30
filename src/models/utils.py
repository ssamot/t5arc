
import keras
from keras import layers
from regulariser import CustomSplitRegularizer, HalfNeuronsLayer

activation = "relu"

def build_model(input_shape, num_decoder_tokens, encoder_units):
    total_features = input_shape[0] * input_shape[1]



    input_img = keras.Input(shape=input_shape)


    x = keras.layers.Reshape(input_shape, input_shape=total_features)(input_img)
    x = keras.layers.Embedding(input_dim=11, output_dim=16, input_length=total_features)(x)
    x = keras.layers.Flatten()(x)

    #x = keras.layers.Flatten()(input_img)

    n_neurons = 256
    x = keras.layers.Dense(n_neurons, activation=activation,)(x)
    x = keras.layers.LayerNormalization()(x)

    xs = [x]
    for i in range(7):
        # skip connections
        x_new = keras.layers.Dense(n_neurons, activation=activation,
                                   )(x)
        x_new = keras.layers.LayerNormalization()(x_new)

        xs.append(x_new)
        x = keras.layers.add(xs)

    #encoded = layers.Reshape([4,4,8])(x)
    #
    encoded = keras.layers.Dense(encoder_units, activation=activation,
                          )(x)
    encoded = keras.layers.LayerNormalization()(encoded)



    #encoded = keras.layers.Dropout(0.1)(encoded)



    ## decoder
    decoded_inputs = keras.Input(shape=(encoder_units,))
    # decoded = keras.layers.LayerNormalization()(
    #     layers.Dense(n_neurons,activation="relu")(decoded_inputs))

    ttt_input = keras.Input(shape=(encoder_units,))
    #ttt = layers.Activation("tanh", name = "ttt_input_activation")(ttt_input)

    ttt = layers.GaussianNoise(stddev=0.01)(ttt_input)

    ttt = (layers.Dense(encoder_units,name = "ttt_layer",
                               use_bias=False, activation = "tanh")
           (ttt))
    #ttt = BinaryDense(encoder_units)(ttt_input)
    #ttt = AddMultiplyLayer()(ttt_input)



    ttt_model = keras.models.Model(ttt_input, ttt, name = "ttt")

    x = keras.layers.Dense(n_neurons, activation=activation, )(decoded_inputs)
    x = keras.layers.LayerNormalization()(x)

    xs = [x]
    for i in range(7):
        # skip connections
        x_new = keras.layers.Dense(n_neurons, activation=activation,
                                   )(x)
        x_new = keras.layers.LayerNormalization()(x_new)

        xs.append(x_new)
        x = keras.layers.add(xs)


    decoded = layers.Dense(total_features*num_decoder_tokens)(x)
    decoded = layers.Reshape([input_shape[0],input_shape[1],num_decoder_tokens])(decoded)
    decoded = layers.Activation("softmax")(decoded)




    encoder = keras.Model(input_img, encoded, name = "encoder")
    decoder = keras.Model(decoded_inputs,  decoded, name = "decoder")

    #decoder.summary()
    #exit()

    input_left = keras.layers.Input(shape=input_shape)
    input_right = keras.layers.Input(shape=input_shape)

    encoded_left = encoder(input_left)
    encoded_right = encoder(input_right)

    merged = keras.layers.concatenate([encoded_left, encoded_right])
    unmerged = CustomSplitRegularizer("add", 1)(merged)




    autoencoder = keras.Model(input_img, decoder(ttt_model(encoded)),
                              name = "autoencoder")

    twin_autoencoder = keras.Model([input_left, input_right],
                                    decoder(ttt_model(unmerged)))
    #print(autoencoder.summary())
    #optimizer = keras.optimizers.SGD(momentum=0.3, weight_decay=0.01, nesterov=True, learning_rate=0.1)
    optimizer = keras.optimizers.AdamW(learning_rate=0.001)

    def cce(y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1):
        return keras.metrics.categorical_crossentropy(y_true, y_pred, from_logits, label_smoothing, axis)

    twin_autoencoder.compile(optimizer=optimizer,

                        loss='categorical_crossentropy',
                        metrics=["acc", b_acc, cce])

    return autoencoder, twin_autoencoder, encoder, decoder, ttt_model


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





def acc_seq(y_true, y_pred):
    return keras.ops.mean(keras.ops.min(keras.ops.equal(keras.ops.argmax(y_true, axis=-1),
                  keras.ops.argmax(y_pred, axis=-1)), axis=-1))


def b_acc(y_true, y_pred):
    # Get the predicted class by taking the argmax along the last dimension (the class dimension)
    pred_classes = keras.ops.argmax(y_pred, axis=-1)

    # Get the true class (already one-hot encoded, so take argmax)
    true_classes = keras.ops.argmax(y_true, axis=-1)

    # Compare predicted classes to true classes for each sample
    correct_predictions = keras.ops.equal(pred_classes, true_classes)

    # Sum the incorrect predictions for each sample
    incorrect_predictions = keras.ops.sum(keras.ops.cast(~correct_predictions,
                                                      "float32"),
                                                 axis=[1, 2])

    # A sample is correct only if the sum of incorrect predictions is 0
    all_correct = keras.ops.cast(keras.ops.equal(incorrect_predictions, 0),
                                 "float32")

    # Calculate the percentage of samples that are fully correct
    accuracy = keras.ops.mean(keras.ops.cast(all_correct, "float32"))

    return accuracy



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

