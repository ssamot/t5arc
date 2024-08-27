import click

import logging

import numpy as np
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from tqdm import tqdm
import keras
from utils import acc_seq, batch_acc, AddMultiplyLayer

from data_generators.example_generator.arc_data_generator import get_all_arc_data
from data_generators.example_generator.ttt_data_generator import ArcExampleData
from visualization.visualse_training_data_sets import visualise_training_data


# def build_model(encoder, decoder, n_neurons):
#     #encoder.summary()
#     #decoder.summary()
#     input = keras.layers.Input([32,32,1])
#     encoded = encoder(input)
#     decoded = decoder(encoded)
#     #print(encoded)
#     #exit()
#     keras.ops.svd
#     ttx_input = keras.layers.Input((n_neurons,))
#     ttt_x = keras.layers.Dense(n_neurons, "relu")(ttx_input)
#     ttt_output = keras.layers.Dense(n_neurons)(ttt_x)
#
#
#     ttx_model = keras.models.Model(ttx_input, ttt_output, name = "ttx")
#
#     ttt_x_decoded = decoder(ttx_model(encoded))
#
#     ttt_autoencoder = keras.models.Model(input, ttt_x_decoded)
#     ttt_autoencoder.summary()
#
#     autoencoder = keras.models.Model(input, decoded)
#
#     ttt_autoencoder.compile(optimizer="AdamW",
#
#                         loss='categorical_crossentropy',
#                         metrics=["acc", batch_acc])
#     autoencoder.compile(optimizer="AdamW",
#
#                         loss='categorical_crossentropy',
#                         metrics=["acc", batch_acc])
#
#     ttx_model.compile(optimizer="AdamW", loss = "mse")
#
#
#     return ttt_autoencoder, autoencoder, ttx_model


@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
@click.argument('data_type', type=click.Path())
def main(data_filepath, model_filepath, output_filepath, data_type):

    n_neurons = 128

    logging.info("Loading models")
    encoder = keras.models.load_model(f"{model_filepath}/encoder_{n_neurons}.keras")
    decoder = keras.models.load_model(f"{model_filepath}/decoder_{n_neurons}.keras")
    ttt = keras.models.load_model(f"{model_filepath}/ttt_{n_neurons}.keras")

    #ttt.compile(optimizer="AdamW", loss="mse")
    # Freeze the weights
    encoder.trainable = False
    decoder.trainable = False
    ttt.trainable = False

    it = ArcExampleData('train')

    for r in tqdm(it):
        #print(r["name"])
        #print(r["input"].shape)
        #print(r["output"].shape)
        #exit()

        visualise_training_data(r, f"./plots/{r['name']}.pdf",)
        #exit()


        train_x = np.array(r["input"][:-1], dtype=np.int32)
        test_x = np.array(r["input"][-1:], dtype=np.int32)

        train_y = r["output"][:-1]
        test_y = r["output"][-1:]

        train_y_one_hot = np.eye(11)[np.array(train_x, dtype=np.int32)]
        test_y_one_hot = np.eye(11)[np.array(test_x, dtype=np.int32)]
        #print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        #ttt_autoencoder, autoencoder, ttx_model = build_model(encoder, decoder, n_neurons)

        #print(encoder(train_x))
        #exit()

        # ttt_x = encoder.predict(train_x)
        # ttt_y = encoder.predict(train_y)
        #
        # ttt_x_validation = encoder.predict(test_x)
        # ttt_y_validation = encoder.predict(test_y)

        # ttt.fit(ttt_x, ttt_y, validation_data=(ttt_x_validation, ttt_y_validation),
        # verbose=True, epochs=100000000)
        # exit()

        input = keras.layers.Input([32,32,1])
        ttt_autoencoder = keras.models.Model(input, decoder(ttt(encoder(input))))
        ttt_autoencoder.compile(optimizer="AdamW",
                                loss='categorical_crossentropy',
                                metrics=["acc", batch_acc])
        ttt_autoencoder.fit(x=train_x, y = train_y_one_hot,
                            validation_data=(test_x, test_y_one_hot), batch_size=256,
                            verbose=True, epochs=10000,
                            )

        #p = ttt_autoencoder.predict(r["input"]).argmax(axis=-1)
        #print(p)
        #exit()
        #print(r["output"].shape)
        r["output"] = ttt_autoencoder.predict(r["input"]).argmax(axis = -1)
        #print(r["output"].shape)
        visualise_training_data(r, f"./plots/{r['name']}_predicted.pdf", )

        exit()


        #print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)



        #exit()
        #print(r["name"])
        #exit()









if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
