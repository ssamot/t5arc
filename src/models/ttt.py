import keras
from keras import layers

activation = "relu"

def build_model(input_shape, num_decoder_tokens, encoder, ttt,  decoder):
    total_features = input_shape[0] * input_shape[1]



    input_img = layers.Input(shape=input_shape)

    encoded = encoder(input_img)
    ttted = ttt(encoded)
    x = ttted

    n_neurons = 256
    x = keras.layers.Dense(n_neurons, activation=activation, )(x)
    x = keras.layers.LayerNormalization()(x)

    xs = [x]
    for i in range(3):
        # skip connections
        x_new = keras.layers.Dense(n_neurons, activation=activation,
                                   )(x)
        x_new = keras.layers.LayerNormalization()(x_new)

        xs.append(x_new)
        x = keras.layers.add(xs)

    x = (AddVector(constraint=
                  keras.constraints.MinMaxNorm(1.0,800.0))
         (x))

    decoded = decoder(x)









    ## decoder
    decoded_inputs = keras.Input(shape=(encoder_units,))
    # decoded = keras.layers.LayerNormalization()(
    #     layers.Dense(n_neurons,activation="relu")(decoded_inputs))

    ttt_input = keras.Input(shape=(encoder_units,))
    #ttt = layers.Activation("tanh", name = "ttt_input_activation")(ttt_input)

    ttt = (layers.Dense(encoder_units,name = "ttt_layer",
                               use_bias=False, activation = "linear")
           (ttt_input))

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



    autoencoder = keras.Model(input_img, decoder(ttt_model(encoded)),
                              name = "autoencoder")
    #print(autoencoder.summary())
    #optimizer = keras.optimizers.SGD(momentum=0.3, weight_decay=0.01, nesterov=True, learning_rate=0.1)
    optimizer = keras.optimizers.AdamW(learning_rate=0.001)
    autoencoder.compile(optimizer=optimizer,

                        loss='categorical_crossentropy',
                        metrics=["acc", b_acc])

    return autoencoder, encoder, decoder, ttt_model


@keras.saving.register_keras_serializable(package="models")
class AddVector(layers.Layer):
    def __init__(self, input_dim, constraint=None, **kwargs):
        super(AddVector, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.constraint = constraint  # Accept constraint as an argument

    def build(self, input_shape):
        # Create the trainable weight vector x with the provided constraint
        self.x = self.add_weight(shape=(self.input_dim,),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='x',
                                 constraint=self.constraint)
        super(AddVector, self).build(input_shape)

    def call(self, inputs):
        # Add x to each column of the input matrix A
        return inputs + keras.ops.expand_dims(self.x, -1)
