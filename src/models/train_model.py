import keras
import click
import os
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import json
import secrets
from tokenizer import CharacterTokenizer
import numpy as np
from utils import load_data
import string


def build_model(input_shape, num_decoder_tokens, latent_dim):

    conv_input = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(4, (1, 1), activation='relu', padding='same')(conv_input)
    x = keras.layers.Conv2D(32, input_shape[1:], activation='relu', padding='same')(x)
    x = keras.layers.Flatten()(x)
    conv_model = keras.models.Model(conv_input, x)

    # Inputs
    encoder_inputs = [keras.layers.Input(shape=input_shape) for _ in range(20)]
    # Apply the convolutional base to each input
    conv_outputs = [conv_model(img_input) for img_input in encoder_inputs]

    differences = []
    for i in range(1, len(conv_outputs), 2):
        differences.append(keras.layers.subtract([conv_outputs[i], conv_outputs[i - 1]]))

    encoder_states = keras.layers.add(conv_outputs)
    encoder_states = keras.layers.Dense(latent_dim, activation="relu")(encoder_states)
    #encoder_states = keras.ops.expand_dims(encoder_states, 1)


    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = keras.layers.Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    mask = keras.layers.Masking(mask_value=4.)

    decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(mask(decoder_inputs),
                                         initial_state=[encoder_states, encoder_states])
    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    encoder_model = keras.models.Model(encoder_inputs, encoder_states)

    model = keras.models.Model(encoder_inputs +  [decoder_inputs], decoder_outputs)

    decoder_state_input_h = keras.layers.Input(shape=(latent_dim,))
    decoder_state_input_c = keras.layers.Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    model.compile(optimizer='adamw', loss='categorical_crossentropy')

    #encoder_model.compile()


    return model, encoder_model, decoder_model


@click.command()
@click.argument('json_files', type=click.Path(exists=True))
@click.argument('programme_files', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('bootstrap', type=click.BOOL)
def main(json_files, programme_files, output_filepath, bootstrap):
    #build_model()
    # Initialize a list to store the contents of the JSON files
    train_data, test_data, solvers = load_data(json_files,programme_files)
    chars = set("".join(solvers))  # This is character vocab

    #print(chars)
    #exit()
    num_decoder_tokens = len(chars) + 10

    print("num_decoder_tokens", num_decoder_tokens)
    model, encoder_model, decoder_model = build_model((1, 32,32), num_decoder_tokens, 256)



    model_max_length = 20000

    tokenizer = CharacterTokenizer(chars, model_max_length)
    encoding = tokenizer(
        solvers,
        padding="longest",
        truncation=False,
        return_tensors="np",
    )

    inputs = [c for c in train_data]
    labels = encoding.input_ids
    print(labels.shape)

    # Create an empty array for the one-hot encoded data
    one_hot_encoded = np.zeros((labels.shape[0], labels.shape[1], num_decoder_tokens))

    # Perform one-hot encoding
    rows = np.arange(labels.shape[0])[:, None]
    cols = np.arange(labels.shape[1])
    one_hot_encoded[rows, cols, labels] = 1

    #print(one_hot_encoded.shape)
    #exit()


    decoder_input_data = np.zeros((labels.shape[0], 1, num_decoder_tokens))
    decoder_input_data[:, 0, 0] = 1.

    print(decoder_input_data.shape)
    (model.summary())


    model.fit(inputs + [decoder_input_data], one_hot_encoded[:, :1, :],
              batch_size=32,
              epochs=32,
              validation_split=0.2)


if __name__ == '__main__':
        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_fmt)

        # not used in this stub but often useful for finding various files
        project_dir = Path(__file__).resolve().parents[2]

        # find .env automagically by walking up directories until it's found, then
        # load up the .env entries as environment variables
        load_dotenv(find_dotenv())

        main()