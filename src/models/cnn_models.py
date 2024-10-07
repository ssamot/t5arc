import math
from keras import layers, models, ops
import keras
#from cnn_neuron_binarizers import entropy_loss

normalisation = layers.LayerNormalization

def residual_block(x, filters, stride=1, transpose=False):
    conv = layers.Conv2DTranspose if transpose else layers.Conv2D

    y = conv(filters, kernel_size=3, strides=stride, padding='same')(x)
    y = normalisation()(y)
    y = layers.Activation('relu')(y)

    y = conv(filters, kernel_size=3, strides=1, padding='same')(y)
    y = normalisation()(y)

    if stride != 1 or x.shape[-1] != filters:
        x = conv(filters, kernel_size=1, strides=stride, padding='same')(x)
        x = normalisation()(x)

    y = layers.Add()([x, y])
    y = layers.Activation('relu')(y)

    return y


def build_dense_block(input_shape, filters, num_layers, stride=1, transpose=False, name=None):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    for i in range(num_layers):
        current_stride = stride if i == 0 else 1
        x = residual_block(x, filters, stride=current_stride, transpose=transpose)
        #print(x.shape, "x.shape", transpose)
    return models.Model(inputs, x, name=name)


def build_encoder(input_shape, base_filters, encoder_filters,  name="encoder"):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(base_filters, kernel_size=3, strides=1, padding='same')(inputs)
    x = normalisation()(x)
    x = layers.Activation('relu')(x)

    # Stage 1
    stage1 = build_dense_block(x.shape[1:], base_filters, num_layers=2, stride=1, name="encoder_stage1")
    x = stage1(x)

    # Stage 2
    stage2 = build_dense_block(x.shape[1:], base_filters * 2, num_layers=2, stride=2, name="encoder_stage2")
    x = stage2(x)

    # Stage 3
    stage3 = build_dense_block(x.shape[1:], base_filters * 4, num_layers=2, stride=2, name="encoder_stage3")
    x = stage3(x)

    encoded = layers.Conv2D(encoder_filters, kernel_size=3, strides=1,
                          padding='same')(x)
    encoded = normalisation()(encoded)
    encoded = layers.Activation('relu')(encoded)

    encoder = models.Model(inputs, encoded, name=name)
    return encoder


def build_decoder(input_shape, output_channels, base_filters):
    decoder_input = layers.Input(shape=input_shape)
    x = decoder_input

    # Stage 3
    stage3 = build_dense_block(x.shape[1:], base_filters * 4, num_layers=2, stride=2, transpose=True,
                               name="decoder_stage3")
    x = stage3(x)

    # Stage 2
    stage2 = build_dense_block(x.shape[1:], base_filters * 2, num_layers=2, stride=2, transpose=True,
                               name="decoder_stage2")
    x = stage2(x)

    # Stage 1
    stage1 = build_dense_block(x.shape[1:], base_filters, num_layers=2, stride=1, transpose=True, name="decoder_stage1")
    x = stage1(x)

    # Final convolution
    decoded = layers.Conv2DTranspose(output_channels, kernel_size=3, strides=1,
                                     padding='same')(x)
    decoded = layers.Activation("softmax")(decoded)

    decoder = models.Model(decoder_input, decoded, name="decoder")
    return decoder


def build_parameters(input_shape, squeezed_neurons):
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
class BatchAverageLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(BatchAverageLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # No trainable weights in this layer, so we don't need to define anything in build()
        super(BatchAverageLayer, self).build(input_shape)

    def call(self, inputs):
        # Compute the mean across the batch (axis 0)
        batch_mean = ops.mean(inputs, axis=0, keepdims=True)
        # Broadcast the mean back to the shape of the original input
        batch_mean_repeated = ops.tile(batch_mean, (ops.shape(inputs)[0], 1, 1, 1))
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
    sprime_decoder = build_decoder(encoder_output_shape, input_shape[-1], base_filters)

    ssprime_encoder = build_encoder(ssprime_input_shape, base_filters,encoder_filters, "ssprime_encoder")

    ssprime_decoded = ssprime_encoder(ssprime_input)

    param_layer, squize_layer = build_parameters(ssprime_decoded.shape, squeeze_neurons)

    return s_input, ssprime_input, s_encoder, ssprime_encoder, sprime_decoder, param_layer, squize_layer, BatchAverageLayer

def get_components_single(squeeze_neurons, base_filters, encoder_filters):
    # Create the models
    input_shape = (32, 32, 11)
    ssprime_input_shape = (32, 32, 22)
    s_input = layers.Input(shape=input_shape)
    ssprime_input = layers.Input(shape=ssprime_input_shape)

    s_encoder_masks = build_encoder(input_shape, base_filters, encoder_filters, "s_encoder")
    s_encoder_tranformer = build_encoder(input_shape, base_filters, encoder_filters, "s_encoder")

    encoder_output_shape = s_encoder_masks.output.shape[1:]
    sprime_decoder_masks = build_decoder(encoder_output_shape, 10, base_filters)

    ssprime_encoder_masks = build_encoder(ssprime_input_shape, base_filters,encoder_filters, "ssprime_encoder")
    ssprime_encoder_transformed = build_encoder(ssprime_input_shape, base_filters, encoder_filters, "ssprime_encoder")

    ssprime_decoded = ssprime_encoder(ssprime_input)

    param_layer, squize_layer = build_parameters(ssprime_decoded.shape, squeeze_neurons)

    return s_input, ssprime_input, s_encoder, ssprime_encoder, sprime_decoder, param_layer, squize_layer, BatchAverageLayer
