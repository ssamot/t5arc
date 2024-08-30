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


def process_batch(batch, max_perms):
    return apply_colour_augmentation_whole_dataset(batch, max_perms)


def augment_batch_inout(batch, max_perms):


    for single_example in batch:
        train_x_a = []
        train_y_a = []
        for i in range(len(single_example)):
            merged_array = np.concatenate((single_example[i], single_example[i]), axis=1)
            # print(merged_array.shape)

            augmented = generate_consistent_combinations_2d(merged_array, max_perms)
            augmented = np.array(augmented)
            # print(augmented.shape)

            augmented_train_x, augmented_train_y = np.split(augmented, 2, axis=2)
            train_x_a.extend(augmented_train_x)
            train_y_a.extend(augmented_train_y)
            # print(np.array(augmented_train)[0])
            # exit()

        train_x_a = np.array(train_x_a)
        train_y_a = np.array(train_y_a)

    return np.array([train_x_a, train_y_a])


@click.command()
@click.argument('output_filepath', type=click.Path())
@click.argument('augment', type=click.Choice(['pure', augmented]))
@click.argument('max_perms', type=click.INT)
@click.argument('type', type=click.STRING)
@click.argument('inout', type=click.Choice(['inout', "single"]))
def main(output_filepath, augment, max_perms, type, inout):

    if(inout):
        it = ArcExampleData('train')
        all_train_x = []
        all_test_x = []
        all_train_y = []
        all_test_y = []
        for r in tqdm(it):

            # visualise_training_data(r, f"./plots/{r['name']}.pdf",)

            train_x = np.array(r["input"][:-1], dtype=np.int8)
            test_x = np.array(r["input"][-1:], dtype=np.int8)

            train_y = np.array(r["output"][:-1], dtype=np.int8)
            test_y = np.array(r["output"][-1:], dtype=np.int8)

            all_train_x.append(train_x)
            all_test_x.append(test_x)
            all_train_y.append(train_y)
            all_test_y.append(test_y)

        all_train_x = np.array(all_train_x,dtype=object)
        all_test_x = np.array(all_test_x,dtype=object)
        all_train_y = np.array(all_train_y,dtype=object)
        all_test_y = np.array(all_test_y,dtype=object)
        #print(all_train_x.shape)

        output_filepath = f"{output_filepath}/{type}_{augment}_{inout}.npz"
        np.savez_compressed(output_filepath, train_x = all_train_x,
                            test_x = all_test_x, train_y = all_train_y, test_y = all_test_y)

        return

        if(augment):
                pass



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
