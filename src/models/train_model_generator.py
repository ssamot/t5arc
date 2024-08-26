import keras
import click
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from data.data_generator import CanvasDataGenerator
from keras import layers
from utils import CustomModelCheckpoint, build_model






@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath):
    # build_model()
    # Initialize a list to store the contents of the JSON files

    max_pad_size = 32
    encoder_units = 16


    training_generator = CanvasDataGenerator(batch_size = 64,  len = 100,
                                             use_multiprocessing=True, workers=50,
                                             max_queue_size=1000)

    num_decoder_tokens = 11
    model, encoder, decoder = build_model((max_pad_size, max_pad_size, 1),
                                          int(num_decoder_tokens),
                                          encoder_units)

    model.summary()
    models = {f"encoder_{encoder_units}": encoder, f"decoder_{encoder_units}": decoder}

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
