import math
from keras import layers, models, ops
import keras
#from cnn_neuron_binarizers import entropy_loss

normalisation = layers.LayerNormalization
activation = "relu"





def build_encoder(input_shape, n_neurons, encoded_neurons,  name="encoder"):

    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Flatten()(inputs)
    length = x.shape[-1]

    x = layers.Reshape(target_shape=(length,1))(x)
    print(x.shape)
    #exit()


    x = layers.Embedding(input_dim=11, output_dim=4, input_length=length)(x)
    x = layers.Flatten()(x)


    x = layers.Dense(n_neurons, activation=activation, )(x)
    x = normalisation()(x)

    xs = [x]
    for i in range(7):
        # skip connections
        x_new = layers.Dense(n_neurons, activation=activation,
                                   )(x)
        x_new = normalisation()(x_new)

        xs.append(x_new)
        x = layers.add(xs)

    encoded = layers.Dense(encoded_neurons)(x)
    encoded = normalisation()(encoded)


    encoder = models.Model(inputs, encoded, name=name)
    return encoder


def build_decoder(input_shape, output_channels, n_neurons):
    decoder_input = layers.Input(shape=input_shape)

    x = layers.Dense(n_neurons, activation=activation, )(decoder_input)
    xs = [x]
    for i in range(7):
        # skip connections
        x_new = layers.Dense(n_neurons, activation=activation,
                                   )(x)
        x_new = normalisation()(x_new)

        xs.append(x_new)
        x = layers.add(xs)

    # decoded = layers.Dense(total_features, name = "dense_decoded")(x)
    decoded = layers.Dense(math.prod(output_channels))(x)
    decoded = layers.Reshape(output_channels)(decoded)
    decoded = layers.Activation("softmax")(decoded)



    decoder = models.Model(decoder_input, decoded, name="decoder")
    return decoder


def build_parameters(input_shape, squeezed_neurons=8):
    ## inputs

    parameters_input = layers.Input(shape=input_shape[1:])
    n_neurons = math.prod(input_shape[1:])
    squeeze_input = layers.Input(shape=(squeezed_neurons,))

    ## attention
    squeezed = squeeze_input
    squeezed = layers.Dense(units=n_neurons, use_bias=False)(squeezed)
    squeezed = normalisation()(squeezed)
    squeezed = layers.Activation("sigmoid")(squeezed)
    squeezed = layers.Reshape(target_shape=input_shape[1:])(squeezed)
    squeeze_model = models.Model(squeeze_input, squeezed, name="attention_direct")

    ##bottlenectk
    ss_prime_parameters = layers.Flatten()(parameters_input)
    ss_prime_parameters = (layers.Dense(units=squeezed_neurons,
                                        use_bias=False)
                           (ss_prime_parameters))
    ss_prime_parameters = normalisation()(ss_prime_parameters)
    ss_prime_parameters = layers.Activation("tanh")(ss_prime_parameters)
    param_encoder = models.Model(parameters_input,
                                 squeeze_model(ss_prime_parameters),
                                 name="attention_through_ssprime")

    return param_encoder, squeeze_model




@keras.saving.register_keras_serializable()
class BatchAverageLayerMLP(layers.Layer):
    def __init__(self, **kwargs):
        super(BatchAverageLayerMLP, self).__init__(**kwargs)

    def build(self, input_shape):
        # No trainable weights in this layer, so we don't need to define anything in build()
        super(BatchAverageLayerMLP, self).build(input_shape)

    def call(self, inputs):
        # Compute the mean across the batch (axis 0)
        batch_mean = ops.mean(inputs, axis=0, keepdims=True)
        # Broadcast the mean back to the shape of the original input
        batch_mean_repeated = ops.tile(batch_mean, (ops.shape(inputs)[0], 1, ))
        return batch_mean_repeated

    def compute_output_shape(self, input_shape):
        # The output shape is the same as the input shape
        return input_shape


def get_components(squeeze_neurons, base_filters, encoder_filters):
    # Create the models
    input_shape = (32, 32, 11)
    ssprime_input_shape = (32, 32, 22)
    s_input = layers.Input(shape=input_shape)
    ssprime_input = layers.Input(shape=ssprime_input_shape)

    s_encoder = build_encoder(input_shape, base_filters, encoder_filters, "s_encoder")

    encoder_output_shape = s_encoder.output.shape[1:]
    sprime_decoder = build_decoder(encoder_output_shape, input_shape, base_filters)

    ssprime_encoder = build_encoder(ssprime_input_shape, base_filters,encoder_filters, "ssprime_encoder")

    ssprime_decoded = ssprime_encoder(ssprime_input)

    param_layer, squize_layer = build_parameters(ssprime_decoded.shape, squeeze_neurons)

    return s_input, ssprime_input, s_encoder, ssprime_encoder, sprime_decoder, param_layer, squize_layer, BatchAverageLayerMLP


