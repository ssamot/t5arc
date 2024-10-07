from keras import layers
from keras import models
from keras import ops

def build_end_to_end(s_input, ssprime_input,
                     s_encoder, ssprime_encoder, sprime_decoder, param_layer, BatchAverageLayerMLP):
    ssprime_decoded = ssprime_encoder(ssprime_input)
    attention = param_layer(ssprime_decoded)
    encoded = s_encoder(s_input)


    merged = layers.add([encoded,
                         BatchAverageLayerMLP()(ssprime_decoded),
                         layers.multiply([encoded, attention])
                         ])


    autoencoder = models.Model([s_input, ssprime_input],
                               sprime_decoder(merged),
                               name="autoencoder")

    # Compile the autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder


# def build_end_to_end_copy(s_input, ssprime_input,
#                      s_encoder, ssprime_encoder, sprime_decoder, param_layer, BatchAverageLayerMLP):
#     masks = sprime_decoder(ssprime_encoder(ssprime_input))
#     transformations = ssprime_encoder(ssprime_input)
#
#
#
#     merged = layers.add([encoded,
#                          BatchAverageLayerMLP()(ssprime_decoded),
#                          layers.multiply([encoded, attention])
#                          ])
#
#
#     autoencoder = models.Model([s_input, ssprime_input],
#                                sprime_decoder(merged),
#                                name="autoencoder")
#
#     # Apply the transformation
#     invert_x = layers.Lambda(lambda x: 1 - x)
#
#
#     # Combine the original and transformed regions
#     #result = image * (1 - mask_3d) + transformed_region * mask_3d
#
#     result = (ops.multiply([s_input, invert_x(mask_3d)]) +
#               ops.multiply([transformed_region, mask_3d])
#
#     # Compile the autoencoder
#     autoencoder.compile(optimizer='adam', loss='mse')
#
#     return autoencoder


def build_end_to_end_single(s_input, ssprime_input,
                     s_encoder, ssprime_encoder, sprime_decoder, param_layer):

    image_pairs = ops.split(ssprime_input, 10, axis=-1)

    output_group = [ssprime_encoder(pair) for pair in image_pairs]
    ssprime_decoded = layers.maximum(output_group)
    #ssprime_decoded = ssprime_encoder(ssprime_input)
    attention = param_layer(ssprime_decoded)
    encoded = s_encoder(s_input)


    merged = layers.add([encoded,
                         ssprime_decoded,
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


def build_ssprime_ave_model(ssprime_input, ssprime_encoder, param_layer, BatchAverageLayerMLP):
    ssprime_decoded = ssprime_encoder(ssprime_input)
    ave = BatchAverageLayerMLP()(ssprime_decoded)
    params = param_layer(ssprime_decoded)

    model = models.Model(ssprime_input, [ave, params], name="ssprime_ave_model")

    return model


def build_s_encoder(s_input, s_encoder):
    s_encoded = s_encoder(s_input)
    model = models.Model(s_input, s_encoded, name="ssprime_ave_model")

    return model


def build_autoencoder(s_input,
                     s_encoder, sprime_decoder):
    encoded = s_encoder(s_input)

    autoencoder = models.Model(s_input,
                               sprime_decoder(encoded),
                               name="autoencoder")

    return autoencoder

