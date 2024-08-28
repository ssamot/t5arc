import click
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from utils import CustomModelCheckpoint, build_model
import numpy as np
from tqdm.keras import TqdmCallback





@click.command()
@click.argument('train_data', type=click.Path())
@click.argument('eval_data', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(train_data, eval_data, output_filepath):
    # build_model()
    # Initialize a list to store the contents of the JSON files

    max_pad_size = 32
    encoder_units = 256
    num_decoder_tokens = 11



    model, encoder, decoder, ttt = build_model((max_pad_size, max_pad_size),
                                          int(num_decoder_tokens),
                                          encoder_units)


    model.summary()

    X_train = np.load(train_data)["X"]
    y_train = np.eye(num_decoder_tokens)[X_train]

    X_validation = np.load(eval_data)["X"]
    y_validation = np.eye(num_decoder_tokens)[X_validation]



    models = {f"encoder_{encoder_units}": encoder,
              f"decoder_{encoder_units}": decoder,
              f"ttt_{encoder_units}": ttt
              }




    model.fit(x=X_train, y = y_train,
              validation_data=(X_validation, y_validation),
              batch_size=128,validation_batch_size=128,
              epochs=100000,verbose = 0,
              callbacks=[CustomModelCheckpoint(models,"./models", 100),
              TqdmCallback(verbose=1)])


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
