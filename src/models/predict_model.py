import click

import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from tqdm import tqdm
import keras




@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
@click.argument('data_type', type=click.Path())
def main(data_filepath, model_filepath, output_filepath, data_type):
    logging.info("Loading models")
    encoder = keras.models.load_model(f"{model_filepath}/encoder.keras")
    decoder = keras.models.load_model(f"{model_filepath}/decoder.keras")
    # Freeze the weights
    encoder.trainable = False
    decoder.trainable = False
    input = keras.layers.Input([32,32,1])
    encoded = encoder(input)
    decoded = decoder(encoded)
    autoencoder = keras.models.Model(input, decoded)

    ttt_x = keras.layers.Dense(64)(encoded)

    ttt_x_decoded = decoder(ttt_x)

    ttt_autoencoder = keras.models.Model(input, ttt_x_decoded)
    ttt_autoencoder.summary()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
