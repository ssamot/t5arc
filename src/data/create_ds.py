import click
import logging

import tqdm
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from models.tokenizer import CharacterTokenizer
import numpy as np
from models.utils import load_data


@click.command()
@click.argument('json_files', type=click.Path(exists=True))
@click.argument('programme_files', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('max_token_length', type=click.INT)
@click.argument('repetitions', type=click.INT)
def main(json_files, programme_files, output_filepath, max_token_length, repetitions):
    # build_model()
    # Initialize a list to store the contents of the JSON files
    train_data, test_data, solvers = [], [], []
    for _ in tqdm.tqdm(range(repetitions)):
        tr_data, te_data, so = load_data(json_files, programme_files)
        train_data.extend(tr_data)
        test_data.extend(te_data)
        solvers.extend(so)

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    # print("train_data.shape", train_data.shape)
    chars = set("".join(solvers))  # This is character vocab

    indices = [i for i, sublist in enumerate(solvers) if len(sublist) < max_token_length]
    solvers = [sublist for sublist in solvers if len(sublist) < max_token_length]
    test_data = test_data[indices]

    num_decoder_tokens = len(chars) + 10

    print("num_decoder_tokens", num_decoder_tokens)

    model_max_length = 20000000

    tokenizer = CharacterTokenizer(chars, model_max_length)
    tokenized_inputs = tokenizer(
        solvers,
        padding="longest",
        truncation=True,
        return_tensors="np",
    )

    inputs = [c for c in np.transpose(train_data[indices], (1, 0, 2, 3, 4))]

    encoded_solutions = tokenized_inputs.input_ids
    #print(encoded_solutions[0])
    #print(tokenizer.decode(encoded_solutions[0], skip_special_tokens=True))
    #exit()

    # Create an empty array for the one-hot encoded data
    one_hot_encoded = np.zeros(
        (tokenized_inputs.input_ids.shape[0], tokenized_inputs.input_ids.shape[1], num_decoder_tokens))
    rows = np.arange(tokenized_inputs.input_ids.shape[0])[:, None]
    cols = np.arange(tokenized_inputs.input_ids.shape[1])
    one_hot_encoded[rows, cols, tokenized_inputs.input_ids] = 1

    target_texts = np.zeros_like(encoded_solutions)
    target_texts[:, :-1] = encoded_solutions[:, 1:]

    targets_one_hot_encoded = one_hot_encoded[:, 1:, :]
    targets_inputs = encoded_solutions[:, :-1]
    print("Target one-hot encoded shape", one_hot_encoded.shape)
    print("Inputs shape", len(inputs), inputs[0].shape)
    print("Test shape", len(test_data))
    print("Chars", len(chars))
    print("max_token_length", max_token_length)
    inputs, targets_inputs, targets_one_hot_encoded, test_data, chars, max_token_length
    np.savez(output_filepath, inputs=inputs,
             targets_inputs=targets_inputs,
             targets_one_hot_encoded=targets_one_hot_encoded,
             test_data=test_data,
             chars=chars,
             max_token_length=max_token_length,
             num_decoder_tokens = num_decoder_tokens,
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
