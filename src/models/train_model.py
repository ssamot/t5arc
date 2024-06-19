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
from utils import load_data, masked_categorical_crossentropy
import keras_nlp

def build_model(input_shape, num_decoder_tokens, latent_dim):
    conv_input = keras.layers.Input(shape=input_shape)
    #print(input_shape[:-1])
    x = keras.layers.Conv2D(8, (1, 1), activation='relu', padding='valid')(conv_input)
    x = keras.layers.Conv2D(32, input_shape[:-1], activation='relu', padding='valid')(x)
    x = keras.layers.Flatten()(x)
    conv_model = keras.models.Model(conv_input, x)

    # Inputs
    encoder_inputs = [keras.layers.Input(shape=input_shape) for _ in range(20)]
    # Apply the convolutional base to each input
    conv_outputs = [conv_model(img_input) for img_input in encoder_inputs]

    differences = []
    for i in range(1, len(conv_outputs), 2):
        diff = keras.layers.subtract([conv_outputs[i], conv_outputs[i - 1]])
        #print(diff.shape)
        differences.append(diff)
    #exit()

    encoder_states = keras.layers.add(differences)

    encoder_states = keras.layers.Dense(latent_dim, activation="relu")(encoder_states)
    encoder_states = keras.ops.expand_dims(encoder_states, 1)

    decoder_inputs = keras.layers.Input(shape=(None,))  # (batch_size, sequence_length)
    dec_embedding = keras.layers.Embedding(input_dim=num_decoder_tokens, output_dim=latent_dim, mask_zero=True)(
        decoder_inputs)


    lstm_out = keras_nlp.layers.TransformerDecoder(
        intermediate_dim=latent_dim, num_heads=8)(dec_embedding,encoder_states)

    # lstm_out = keras.layers.LSTM(latent_dim, return_sequences=True)(dec_embedding,
    #                                                                 initial_state=[encoder_states, encoder_states])


    decoder_outputs = keras.layers.TimeDistributed(keras.layers.Dense(num_decoder_tokens, activation='softmax'))(
        lstm_out)


    model = keras.models.Model(encoder_inputs + [decoder_inputs], decoder_outputs)

    model.compile(optimizer='adamw', loss="categorical_crossentropy", metrics=["accuracy"])

    # encoder_model.compile()

    return model


@click.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('bootstrap', type=click.BOOL)
def main(data_file, output_filepath, bootstrap):
    x = np.load(data_file, allow_pickle=True)

    inputs = x["inputs"]
    print(inputs.shape)

    targets_inputs = x["targets_inputs"]
    targets_one_hot_encoded = x["targets_one_hot_encoded"]
    test_data = x["test_data"]
    chars = x["chars"]
    max_token_length = x["max_token_length"]
    num_decoder_tokens = x["num_decoder_tokens"]
    print(num_decoder_tokens)

    inputs = [c for c in inputs]
    model = build_model((32, 32,1), int(num_decoder_tokens), 128)

    model.summary()


    model.fit(inputs + [targets_inputs], targets_one_hot_encoded, epochs=10000, batch_size=128)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
