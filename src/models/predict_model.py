import click

import logging

import numpy as np
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from tqdm import tqdm
import keras
from utils import acc_seq

from data_generators.example_generator.arc_data_generator import get_all_arc_data
from data_generators.example_generator.ttt_data_generator import ArcExampleData

def build_model(encoder, decoder):
    #encoder.summary()
    #decoder.summary()
    input = keras.layers.Input([32,32,1])
    encoded = encoder(input)
    decoded = decoder(encoded)
    #print(encoded)
    #exit()
    ttt_x = keras.layers.Dense(1024, activation = "relu")(encoded)
    ttt_x = keras.layers.Dense(1024, activation="relu")(ttt_x)
    ttt_x = keras.layers.Dense(1024, activation="relu")(ttt_x)
    ttt_x = keras.layers.Dense(64, activation="relu")(ttt_x)

    ttt_x_decoded = decoder(ttt_x)

    ttt_autoencoder = keras.models.Model(input, ttt_x_decoded)
    ttt_autoencoder.summary()

    autoencoder = keras.models.Model(input, decoded)

    ttt_autoencoder.compile(optimizer="AdamW",

                        loss='categorical_crossentropy',
                        metrics=["acc", acc_seq])

    return ttt_autoencoder, autoencoder


@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
@click.argument('data_type', type=click.Path())
def main(data_filepath, model_filepath, output_filepath, data_type):


    logging.info("Loading models")
    encoder = keras.models.load_model(f"{model_filepath}/encoder_64.keras")
    decoder = keras.models.load_model(f"{model_filepath}/decoder_64.keras")
    # Freeze the weights
    encoder.trainable = False
    decoder.trainable = False

    it = ArcExampleData('train')

    for r in tqdm(it):
        #print(r["name"])
        # print(r["input"].shape)
        # print(r["output"].shape)

        train_x = np.array(r["input"][:-1], dtype=np.int32)
        test_x = np.array(r["input"][-1:], dtype=np.int32)

        train_y = r["output"][:-1]
        test_y = r["output"][-1:]

        train_y = np.eye(11)[np.array(train_y, dtype=np.int32)]
        test_y = np.eye(11)[np.array(test_y, dtype=np.int32)]
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        ttt_autoencoder, autoencoder = build_model(encoder, decoder)
        ttt_autoencoder.fit(x=train_x, y = train_y,
                            validation_data=(test_x, test_y), batch_size=256,
                            verbose=True, epochs=100000,
                            )
        #exit()


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
