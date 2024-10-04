import click
import logging


from dotenv import find_dotenv, load_dotenv
from pathlib import Path

import numpy as np
import concurrent.futures
from tqdm import tqdm
import os
import time


from data.generators.task_generator.random_transformations_task import RandomTransformationsTask


# Function that generates random arrays (x, y) using a random seed.
def generate_samples(seed):
    np.random.seed(os.getpid() + int(time.time()) + seed)
    n = np.random.randint(1, 11)  # Randomly choose n between 1 and 10

    #x = np.random.randn(n, 32, 32, 11)  # Generate random array for x
    #y = np.random.randn(n, 32, 32, 11)  # Generate random array for y

    t = RandomTransformationsTask(num_of_outputs=n, one_to_one=True)
    t.generate_samples()
    x, y = t.get_cnavasses_as_arrays()

    return (x, y)


@click.command()
@click.argument('output_filepath', type=click.Path())
@click.argument('num_samples', type=click.INT)

def main(output_filepath, num_samples):
    results = []
    num_workers = 50

    # Use ProcessPoolExecutor for true multiprocessing
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Creating a list of futures with the seed passed to each task
        futures = [executor.submit(generate_samples, i) for i in range(num_samples)]

        # Using tqdm to show progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_samples):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    # Convert results list into a numpy array and save it
    results_array = np.array(results, dtype=object)

    # Save results to .npz file
    np.savez(output_filepath, samples=results_array)
    print(f"Saved {num_samples} samples to {output_filepath}")





if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
