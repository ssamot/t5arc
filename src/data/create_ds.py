import click
import logging

import tqdm
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from models.tokenizer import CharacterTokenizer
from data_generators.object_recognition.example import Example
import numpy as np
from data.utils import load_data
from models.tokens import token_list

@click.command()
@click.argument('json_files', type=click.Path(exists=True))
@click.argument('programme_files', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('repetitions', type=click.INT)
def main(json_files, programme_files, output_filepath, max_token_length, repetitions):
    # build_model()
    # Initialize a list to store the contents of the JSON files
    json_data_list = []
    objects = []
    object_pixels = []


    for _ in tqdm.tqdm(range(repetitions)):
        # Create an Example
        e = Example()
        e.populate_canvases()
        arc_style_input = e.create_canvas_arrays_input()
        unique_objects, actual_pixels_array, positions_of_same_objects = e.create_output()
        json_data_list.append(arc_style_input)
        objects.append(str(unique_objects).replace(" ", ""))
        object_pixels.append(actual_pixels_array)

    train, test = load_data(json_data_list)


    #print(train.shape)
    #print(objects[-1])
    model_max_length = 20000000

    tokenizer = CharacterTokenizer(token_list, model_max_length)
    tokenized_inputs = tokenizer(
        objects,
        padding="longest",
        truncation=True,
        return_tensors="np",
    )

    object_ids = tokenized_inputs.input_ids
    print(object_ids.shape)
    print(train.shape)



    np.savez(output_filepath, inputs=train, outputs = object_ids,
             max_token_length=max_token_length,
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
