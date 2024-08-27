import keras
import click
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from data.data_generator import CanvasDataGenerator
from keras import layers
from utils import CustomModelCheckpoint, build_model


from data_generators.example_generator.arc_data_generator import get_all_arc_data
import numpy as np



@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath):
    # build_model()
    # Initialize a list to store the contents of the JSON files

    max_pad_size = 32
    encoder_units = 128


    num_decoder_tokens = 11
    model, encoder, decoder, ttt = build_model((max_pad_size, max_pad_size),
                                          int(num_decoder_tokens),
                                          encoder_units)

    model.summary()
    models = {f"encoder_{encoder_units}": encoder,
              f"decoder_{encoder_units}": decoder,
              f"ttt_{encoder_units}": ttt
              }

    X_train  = get_all_arc_data(group='train')
    y_train  = np.eye(11)[np.array(X_train, dtype=np.int32)]
    X_train = np.array(X_train, dtype=np.int32)

    X_validation  = get_all_arc_data(group='eval')
    y_validation  = np.eye(11)[np.array(X_validation, dtype=np.int32)]
    X_validation = np.array(X_validation, dtype=np.int32)



    model.fit(x=X_train, y = y_train,
              validation_data=(X_validation, y_validation),
              batch_size=128,validation_batch_size=128,
              epochs=10000,
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
