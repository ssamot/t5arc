import click
import logging

from tqdm import tqdm
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from data.generators.example_generator.auto_encoder_data_generator import AutoEncoderDataExample
from data.generators.example_generator.arc_data_generator import get_all_arc_data
from data.generators.example_generator.ttt_data_generator import ArcExampleData


import numpy as np
import secrets
from data.augment.colour import apply_colour_augmentation_whole_dataset
import numpy as np
from multiprocessing import Pool
from functools import partial
from data.augment.colour import generate_consistent_combinations_2d

augmented = "augmented"





@click.command()
@click.argument('output_filepath', type=click.Path())
@click.argument('augment', type=click.Choice(['pure', augmented]))
@click.argument('max_perms', type=click.INT)
@click.argument('type', type=click.STRING)
def main(output_filepath, augment, max_perms, type):


    it = ArcExampleData(type)

    if(augment == augmented):
        it = ArcExampleData(type,
                            augment_with=['colour', 'rotation'],
                            max_samples=max_perms,
                            with_black=False)

    all_train_x = []
    all_test_x = []
    all_train_y = []
    all_test_y = []
    for r in tqdm(it):

        train_x = np.array(r["input"][:-1], dtype=np.int8)
        test_x = np.array(r["input"][-1:], dtype=np.int8)

        train_y = np.array(r["output"][:-1], dtype=np.int8)
        test_y = np.array(r["output"][-1:], dtype=np.int8)

        all_train_x.append(train_x)
        all_test_x.append(test_x)
        all_train_y.append(train_y)
        all_test_y.append(test_y)

    all_train_x = np.array(all_train_x, dtype=object)
    all_test_x = np.array(all_test_x, dtype=object)
    all_train_y = np.array(all_train_y, dtype=object)
    all_test_y = np.array(all_test_y, dtype=object)

    output_filepath = f"{output_filepath}/{type}_{augment}.npz"
    np.savez_compressed(output_filepath, train_x = all_train_x,
                        test_x = all_test_x, train_y = all_train_y, test_y = all_test_y)






if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
