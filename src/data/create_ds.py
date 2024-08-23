import click
import logging

import tqdm
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from models.tokenizer import CharacterTokenizer
from data_generators.example_generator.auto_encoder_data_generator import AutoEncoderDataExample

import numpy as np
from data.utils import load_data
from models.tokens import token_list
import secrets

@click.command()
@click.argument('output_filepath', type=click.Path())
@click.argument('repetitions', type=click.INT)
def main(output_filepath, repetitions):
    # build_model()
    # Initialize a list to store the contents of the JSON files
    arrays = []
    strings = []


    model_max_length = 20000000
    for _ in tqdm.tqdm(range(repetitions)):
        # Create an Example
        e = AutoEncoderDataExample(100)
        array_reps = e.get_canvases_as_numpy_array()
        #str_reps = e.get_canvasses_as_string()


        #strings.extend([str(str_rep).replace(" ", "") for str_rep in str_reps])
        #strings.extend(str_reps)
        arrays.append(array_reps)
        #tokenizer = CharacterTokenizer(token_list, model_max_length)

        # try:
        #     tokenizer(
        #         [clean_object],
        #         padding="longest",
        #         truncation=True,
        #         return_tensors="np",
        #     )
        # except Exception:
        #     print("======")
        #     print(clean_object)
        #     print(tokenizer._tokenize(clean_object))
        #     print(e)


    arrays = np.concatenate(arrays)
    print(arrays.shape)


    tokenizer = CharacterTokenizer(token_list, model_max_length)
    tokenized_inputs = tokenizer(
        strings,
        padding="longest",
        truncation=True,
        return_tensors="np",
    )

    object_ids = tokenized_inputs.input_ids
    print(object_ids.shape)
    print(arrays.shape)
    #exit()

    hex = secrets.token_hex(nbytes=16)

    output_filepath = f"{output_filepath}/{hex}.npz"
    np.savez(output_filepath, inputs=train, outputs = object_ids,
             num_decoder_tokens = tokenizer.num_decoder_tokens,
             )


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
