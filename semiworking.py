import keras
import click
import os
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from functools import partial
from data.utils import load_data_for_generator, pad_array_with_random_position
import keras_nlp
from data.data_generator import CanvasDataGenerator
import constants as consts
from models.utils import SequenceAccuracy

def categorical_accuracy_per_sequence(y_true, y_pred):
    return keras.ops.mean(keras.ops.min(keras.ops.equal(keras.ops.argmax(y_true, axis=-1),
                  keras.ops.argmax(y_pred, axis=-1)), axis=-1))


def build_model(input_shape, num_decoder_tokens, latent_dim, max_num):
    total_features = input_shape[0] * input_shape[1] * input_shape[2]

    conv_input = keras.layers.Input(shape=input_shape)
    x = conv_input
    x = keras.layers.Reshape(input_shape, input_shape=total_features)(x)
    x = keras.layers.Embedding(input_dim=20, output_dim=2, input_length=total_features)(x)
    x = keras.layers.Flatten()(x)

    n_neurons = 128
    x = keras.layers.Dense(n_neurons, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    xs = [x]
    for i in range(4):
        # skip connections
        x_new = keras.layers.Dense(n_neurons, activation='relu')(x)
        x_new = keras.layers.BatchNormalization()(x_new)
        xs.append(x_new)
        x = keras.layers.add(xs)

    x = keras.layers.Flatten()(x)
#    print(x.shape)
    conv_model = keras.models.Model(conv_input, x)

    # Inputs
    encoder_inputs = [keras.layers.Input(shape=input_shape) for _ in range(max_num)]
    # Apply the convolutional base to each input
    conv_outputs = [conv_model(img_input) for img_input in encoder_inputs]

    # differences = []
    # for i in range(1, len(conv_outputs), 2):
    #     diff = keras.layers.concatenate([conv_outputs[i], conv_outputs[i - 1]])
    #     #print(diff.shape)
    #     differences.append(diff)
    #exit()

    encoder_states = keras.layers.concatenate(conv_outputs)
    print(encoder_states.shape)

    encoder_states = keras.layers.Dense(latent_dim, activation="relu")(encoder_states)
    encoder_states = keras.ops.expand_dims(encoder_states, 1)

    decoder_inputs = keras.layers.Input(shape=(None,))  # (batch_size, sequence_length)
    dec_embedding = keras.layers.Embedding(input_dim=num_decoder_tokens, output_dim=latent_dim, mask_zero=True)(
        decoder_inputs)

    lstm_out = keras_nlp.layers.TransformerDecoder(
        intermediate_dim=latent_dim, num_heads=8)(dec_embedding, encoder_states)

    # lstm_out = keras.layers.LSTM(latent_dim, return_sequences=True)(dec_embedding,
    #                                                                 initial_state=[encoder_states, encoder_states])

    decoder_outputs = keras.layers.TimeDistributed(keras.layers.Dense(num_decoder_tokens, activation='softmax'))(
        lstm_out)

    model = keras.models.Model(encoder_inputs + [decoder_inputs], decoder_outputs)
    optimizer = keras.optimizers.AdamW(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy", categorical_accuracy_per_sequence ])

    # encoder_model.compile()

    return model


@click.command()
@click.argument('json_files', type=click.Path(exists=True))
@click.argument('programme_files', type=click.Path(exists=True))
@click.argument('max_token_length', type=click.INT)
@click.argument('output_filepath', type=click.Path())
def main(json_files, programme_files, max_token_length, output_filepath):
    # build_model()
    # Initialize a list to store the contents of the JSON files

    max_pad_size = 32
    max_examples = 20


    training_generator = CanvasDataGenerator(batch_size = 20,  len = 10,
                                             use_multiprocessing=True, workers=50, max_queue_size=1000)

    num_decoder_tokens = training_generator.tokenizer.num_decoder_tokens + 1
    model = build_model((max_pad_size, max_pad_size, 1), int(num_decoder_tokens), 256, max_examples)

    model.summary()

    model.fit(x=training_generator, epochs=10000)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
