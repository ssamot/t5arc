import math
from keras import layers, models, ops
import keras
from cnn_neuron_binarizers import entropy_loss


def residual_block(x, filters, stride=1, transpose=False):
    conv = layers.Conv2DTranspose if transpose else layers.Conv2D

    y = conv(filters, kernel_size=3, strides=stride, padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    y = conv(filters, kernel_size=3, strides=1, padding='same')(y)
    y = layers.BatchNormalization()(y)

    if stride != 1 or x.shape[-1] != filters:
        x = conv(filters, kernel_size=1, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)

    y = layers.Add()([x, y])
    y = layers.Activation('relu')(y)

    return y


def build_dense_block(input_shape, filters, num_layers, stride=1, transpose=False, name=None):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    for i in range(num_layers):
        current_stride = stride if i == 0 else 1
        x = residual_block(x, filters, stride=current_stride, transpose=transpose)

    return models.Model(inputs, x, name=name)


def build_encoder(input_shape, base_filters=16, name="encoder"):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(base_filters, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
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

    block = layers.Conv2D(11, kernel_size=3, strides=1,
                          padding='same', activation='tanh')(x)

    encoder = models.Model(inputs, block, name=name)
    return encoder


def build_decoder(input_shape, output_channels, base_filters=16):
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
    decoded = layers.BatchNormalization()(decoded)
    decode = layers.Activation("relu")(decoded)

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
    squeezed = layers.BatchNormalization()(squeezed)
    squeezed = layers.Activation("sigmoid", activity_regularizer="l1")(squeezed)
    squeezed = layers.Reshape(target_shape=input_shape[1:])(squeezed)
    squeeze_model = models.Model(squeeze_input, squeezed, name="attention_direct")

    ##bottlenectk
    ss_prime_parameters = layers.Flatten()(parameters_input)
    ss_prime_parameters = (layers.Dense(units=squeezed_neurons,
                                        use_bias=False)
                           (ss_prime_parameters))
    ss_prime_parameters = layers.BatchNormalization()(ss_prime_parameters)
    ss_prime_parameters = layers.Activation("tanh")(ss_prime_parameters)
    param_encoder = models.Model(parameters_input,
                                 squeeze_model(ss_prime_parameters),
                                 name="attention_through_ssprime")

    return param_encoder, squeeze_model


def build_autoencoder(input_shape, base_filters=16):
    encoder = build_encoder(input_shape, base_filters)

    encoder_output_shape = encoder.output.shape[1:]
    decoder = build_decoder(encoder_output_shape, input_shape[-1], base_filters)

    inputs = layers.Input(shape=input_shape)
    encoded = encoder(inputs)
    decoded = decoder(encoded)

    autoencoder = models.Model(inputs, decoded, name="autoencoder")
    return encoder, decoder, autoencoder


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


def get_components(squeeze_neurons):
    # Create the models
    input_shape = (32, 32, 11)
    ssprime_input_shape = (32, 32, 22)
    base_filters = 32
    s_input = layers.Input(shape=input_shape)
    ssprime_input = layers.Input(shape=ssprime_input_shape)

    s_encoder = build_encoder(input_shape, base_filters, "s_encoder")

    encoder_output_shape = s_encoder.output.shape[1:]
    sprime_decoder = build_decoder(encoder_output_shape, input_shape[-1], base_filters)

    ssprime_encoder = build_encoder(ssprime_input_shape, base_filters, "ssprime_encoder")

    ssprime_decoded = ssprime_encoder(ssprime_input)

    param_layer, squize_layer = build_parameters(ssprime_decoded.shape, squeeze_neurons)

    return s_input, ssprime_input, s_encoder, ssprime_encoder, sprime_decoder, param_layer, squize_layer


def build_end_to_end(s_input, ssprime_input,
                     s_encoder, ssprime_encoder, sprime_decoder, param_layer):
    ssprime_decoded = ssprime_encoder(ssprime_input)
    attention = param_layer(ssprime_decoded)
    encoded = s_encoder(s_input)
    merged = layers.add([encoded,
                         BatchAverageLayer()(ssprime_decoded),
                         layers.multiply([encoded, attention])
                         ])

    autoencoder = models.Model([s_input, ssprime_input],
                               sprime_decoder(merged),
                               name="autoencoder")

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


def build_ttt(s_input, ssprime_input,
              s_encoder, ssprime_encoder, sprime_decoder, squize_layer):
    ssprime_decoded = ssprime_encoder(ssprime_input)

    ttt_ssprime_encoded = layers.Input(ssprime_decoded.shape[1:])
    ttt_ssprime_params = layers.Input(squize_layer.input.shape[1:])

    attention = squize_layer(ttt_ssprime_params)
    encoded = s_encoder(s_input)

    ttt_merged = layers.add([encoded,
                             ttt_ssprime_encoded,
                             layers.multiply([encoded, attention])
                             ])

    ttt_model = models.Model([s_input, ttt_ssprime_encoded, ttt_ssprime_params],
                             sprime_decoder(ttt_merged), name="ttt_model")

    return ttt_model


def build_ssprime_ave_model(ssprime_input, ssprime_encoder, param_layer):
    ssprime_decoded = ssprime_encoder(ssprime_input)
    ave = BatchAverageLayer()(ssprime_decoded)
    params = param_layer(ssprime_decoded)

    model = models.Model(ssprime_input, [ave, params], name="ssprime_ave_model")

    return model


def build_s_encoder(s_input, s_encoder):
    s_encoded = s_encoder(s_input)
    model = models.Model(s_input, s_encoded, name="ssprime_ave_model")

    return model


if __name__ == '__main__':
    (s_input, ssprime_input,
     s_encoder, ssprime_encoder, sprime_decoder,
     param_layer, squeeze_layer) = get_components()

    e2e = build_end_to_end(s_input, ssprime_input,
                           s_encoder, ssprime_encoder, sprime_decoder, param_layer)

    ttt = build_ttt(s_input, ssprime_input,
                    s_encoder, ssprime_encoder, sprime_decoder, squeeze_layer)

    ## get X_train, X_test for TTT
    s_encoder = build_s_encoder(s_input, s_encoder)

    ## get y_train for TTT, y_test for TTT
    ave_model = build_ssprime_ave_model(ssprime_input, ssprime_encoder, param_layer)

    e2e.summary()
    ttt.summary()
    ave_model.summary()
    s_encoder.summary()

#autoencoder.summary()

# Print model summaries

# print("\nEncoder Block Summaries:")
# for i, block in enumerate(encoder_blocks):
#     print(f"Encoder Block {i + 1}:")
#     block.summary()
#
# print("\nDecoder Block Summaries:")
# for i, block in enumerate(decoder_blocks):
#     print(f"Decoder Block {i + 1}:")
#     block.summary()
#
# # Test the autoencoder
# test_input = tf.random.normal((1,) + input_shape)
# test_output = autoencoder(test_input)
# print(f"\nTest input shape: {test_input.shape}")
# print(f"Test output shape: {test_output.shape}")
