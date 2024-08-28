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
import numpy as np
from multiprocessing import Pool
from functools import partial


augmented = "augmented"


def process_batch(batch, max_perms):
    return apply_colour_augmentation_whole_dataset(batch, max_perms)


@click.command()
@click.argument('output_filepath', type=click.Path())
@click.argument('augment', type=click.Choice(['pure', augmented]))
@click.argument('max_perms', type=click.INT)
@click.argument('type', type=click.STRING)
def main(output_filepath, augment, max_perms, type):




    def parallel_process(arr, n_batches):
        splits = np.array_split(arr, n_batches)
        with Pool() as pool:
            results = pool.map(partial(process_batch, max_perms = max_perms), splits)
        return np.concatenate(results)



    X = np.array(get_all_arc_data(group=type), dtype=np.int8)

    if(augment == augmented):
        X  = parallel_process(X, 50)



    print(f"Dataset shape {X.shape} ")

    # X_validation = np.array(get_all_arc_data(group='eval'), dtype=np.int32)
    # y_validation = np.eye(11)[X_validation]
    # X_validation = np.array(X_validation, dtype=np.int32)


    output_filepath = f"{output_filepath}/{type}_{augment}.npz"
    np.savez_compressed(output_filepath, X = X)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
