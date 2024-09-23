import logging
import warnings


# Suppress all logger messages
logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib.image").setLevel(logging.CRITICAL)

# Suppress all warnings
warnings.filterwarnings("ignore")






import click

import logging

import numpy as np
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from tqdm import tqdm
import keras
from utils import b_acc

from data.generators.task_generator.ttt_data_generator import ArcTaskData
from visualization.visualse_training_data_sets import visualise_training_data
from tqdm.keras import TqdmCallback
from data.augment.colour import generate_consistent_combinations_2d
from keras import layers

from sklearn.linear_model import MultiTaskLassoCV
from sklearn.linear_model import MultiTaskElasticNetCV, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge







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

    n_neurons = 1024
    keras.config.enable_unsafe_deserialization()


    # for c in combinations_2d:
    #     print(np.array(c))
    # exit()

    logging.info("Loading models")
    encoder = keras.models.load_model(f"{model_filepath}/encoder_{n_neurons}.keras")
    decoder = keras.models.load_model(f"{model_filepath}/decoder_{n_neurons}.keras")

    #ttt = keras.models.load_model(f"{model_filepath}/ttt_{n_neurons}.keras")

    #ttt.compile(optimizer="AdamW", loss="mse")
    # Freeze the weights
    encoder.trainable = False
    decoder.trainable = False
    #ttt.trainable = True

    it = ArcTaskData('train')

    for r in tqdm(it):

        # visualise_training_data(r, f"./plots/{r['name']}.pdf",)

        train_x = np.array(r["input"][:-1], dtype=np.int32)
        test_x = np.array(r["input"][-1:], dtype=np.int32)


        train_y = r["output"][:-1]
        test_y = r["output"][-1:]

        train_x_a = []
        train_y_a = []
        for i in range(len(train_x)):

            merged_array = np.concatenate((train_x[i], train_y[i]), axis=1)
            #print(merged_array.shape)

            augmented = generate_consistent_combinations_2d(merged_array)
            augmented = np.array(augmented)
            #print(augmented.shape)

            augmented_train_x, augmented_train_y = np.split(augmented, 2, axis=2)
            train_x_a.extend(augmented_train_x)
            train_y_a.extend(augmented_train_y)
            #print(np.array(augmented_train)[0])
            #exit()

        train_x_a = np.array(train_x_a)
        train_y_a = np.array(train_y_a)



        #print(train_x.shape, train_y_a.shape, "3424234")

        train_x_a = np.concatenate([train_x, train_x_a])
        train_y_a = np.concatenate([train_y, train_y_a])
        #print(train_x_a.shape, train_y_a.shape, "£23424234")
        #exit()


        train_y_one_hot_a = np.eye(11)[np.array(train_y_a, dtype=np.int32)]
        train_y_one_hot = np.eye(11)[np.array(train_y, dtype=np.int32)]
        test_y_one_hot = np.eye(11)[np.array(test_y, dtype=np.int32)]
        test_x_one_hot = np.eye(11)[np.array(test_x, dtype=np.int32)]


        input = keras.layers.Input([32,32,1])
        input_dec = keras.layers.Input([n_neurons,])


        ttt_input = keras.Input((1,1))

        ttt_emb = layers.Activation("sigmoid")(layers.Flatten()(layers.Embedding(2,
                                                                                 32,
                                                                                 name="embeddings_programmes",
                                                                                 # embeddings_regularizer="l2",
                                                                                 # embeddings_constraint=keras.constraints.NonNeg()
                                                                                 )(ttt_input)))

        decoded =  decoder([input_dec, ttt_emb])
        ttt_decoder = keras.models.Model([input_dec, ttt_input],decoded)

        ttt_decoder.compile(optimizer="AdamW",

                            loss='categorical_crossentropy',
                            metrics=["acc", b_acc, ])

        train_x_a_h = encoder.predict(train_y_one_hot_a)
        train_x_h = encoder.predict(train_y_one_hot)
        test_x_h = encoder.predict(test_x_one_hot)
        test_y_h = encoder.predict(test_y_one_hot)




        programme_train_a = np.ones(shape=(len(train_x_a_h), 1,1))
        programme_test = np.ones(shape=(len(test_y_h), 1, 1))

        print(train_x_a_h.shape,programme_train_a.shape )

        ttt_decoder.fit([train_x_a_h, programme_train_a], train_y_one_hot_a,
                        validation_data=([test_x_h, programme_test], test_y_one_hot),
                        verbose=False, epochs=10000, callbacks=[TqdmCallback(verbose=0)])

        train_output = ttt_decoder.predict([train_x_a_h,programme_train_a]).argmax(axis = -1)
        test_output = ttt_decoder.predict([test_x_h, programme_test]).argmax(axis = -1)

        print(train_output.shape, test_output.shape)
        #exit()




        r["output"] = np.concatenate([train_output, test_output])
        # #print(r["output"].shape)
        #visualise_training_data(r, f"./plots/{r['name']}_predicted.pdf", )
        #exit()
        #
        # #exit()


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
