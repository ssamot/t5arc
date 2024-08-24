import keras
import click
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from data.data_generator import CanvasDataGenerator
from keras import layers
from utils import CustomModelCheckpoint, acc_seq





activation = "relu"

def build_model(input_shape, num_decoder_tokens):
    total_features = input_shape[0] * input_shape[1] * input_shape[2]



    input_img = keras.Input(shape=input_shape)

    x = keras.layers.Flatten()(input_img)

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

    encoder_units = 64

    encoded = layers.Dense(encoder_units, activation="tanh")(x)
    decoded = layers.Dense(total_features*num_decoder_tokens)(encoded)
    decoded = layers.Reshape([input_shape[0],input_shape[1],num_decoder_tokens])(decoded)
    decoded = layers.Activation("softmax")(decoded)


    encoder = keras.Model(input_img, encoded)
    decoder = keras.Model(keras.Input(shape=(encoder_units,)), decoded)



    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=["acc",acc_seq])

    return autoencoder, encoder, decoder


@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath):
    # build_model()
    # Initialize a list to store the contents of the JSON files

    max_pad_size = 32


    training_generator = CanvasDataGenerator(batch_size = 64,  len = 100,
                                             use_multiprocessing=True, workers=50, max_queue_size=1000)

    num_decoder_tokens = 11
    model, encoder, decoder = build_model((max_pad_size, max_pad_size, 1), int(num_decoder_tokens),)

    model.summary()
    models = {"encoder": encoder, "decoder": decoder}

    model.fit(x=training_generator, epochs=10000,
              callbacks=CustomModelCheckpoint(models,"./models", 100))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
