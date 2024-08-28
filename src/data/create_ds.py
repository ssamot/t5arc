import click
import logging

import tqdm
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from data.generators.example_generator.auto_encoder_data_generator import AutoEncoderDataExample
from data.generators.example_generator.arc_data_generator import get_all_arc_data

import numpy as np
import secrets
from data.augment.colour import apply_colour_augmentation_whole_dataset


augmented = "augmented"
@click.command()
@click.argument('output_filepath', type=click.Path())
@click.argument('augment', type=click.Choice(['pure', augmented]))
@click.argument('max_perms', type=click.INT)
@click.argument('type', type=click.STRING)
def main(output_filepath, augment, max_perms, type):




    X = np.array(get_all_arc_data(group=type), dtype=np.int32)

    if(augment == augmented):
        X = apply_colour_augmentation_whole_dataset(X, max_perms)



    print(f"Dataset shape {X.shape} ")

    # X_validation = np.array(get_all_arc_data(group='eval'), dtype=np.int32)
    # y_validation = np.eye(11)[X_validation]
    # X_validation = np.array(X_validation, dtype=np.int32)


    output_filepath = f"{output_filepath}/{type}_{augment}.npz"
    np.savez(output_filepath, X = output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
