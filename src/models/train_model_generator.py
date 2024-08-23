import keras
import click
import os
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from data.data_generator import CanvasDataGenerator
from models.decoder import TransformerDecoder
from keras import layers


def acc_seq(y_true, y_pred):
    return keras.ops.mean(keras.ops.min(keras.ops.equal(keras.ops.argmax(y_true, axis=-1),
                  keras.ops.argmax(y_pred, axis=-1)), axis=-1))

activation = "relu"

def build_model(input_shape, num_decoder_tokens, latent_dim, max_num):
    total_features = input_shape[0] * input_shape[1] * input_shape[2]



    input_img = keras.Input(shape=input_shape)

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding = 'same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    #decode = layers.Dense()
    decoded = layers.Conv2D(num_decoder_tokens, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=["acc",acc_seq])

    return autoencoder


@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath):
    # build_model()
    # Initialize a list to store the contents of the JSON files

    max_pad_size = 32
    max_examples = 20


    training_generator = CanvasDataGenerator(batch_size = 24,  len = 100,
                                             use_multiprocessing=True, workers=50, max_queue_size=1000)

    num_decoder_tokens = 11
    model = build_model((max_pad_size, max_pad_size, 1), int(num_decoder_tokens), 256, max_examples)

    model.summary()

    model.fit(x=training_generator, epochs=10000)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
