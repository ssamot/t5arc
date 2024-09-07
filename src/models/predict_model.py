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

from data.generators.example_generator.ttt_data_generator import ArcExampleData
from visualization.visualse_training_data_sets import visualise_training_data
from tqdm.keras import TqdmCallback
from data.augment.colour import generate_consistent_combinations_2d

from sklearn.metrics import mean_squared_error, r2_score
from regulariser import SVDLinearRegression, AdditionRegression

from utils import build_model



@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
@click.argument('data_type', type=click.Path())
def main(data_filepath, model_filepath, output_filepath, data_type):

    n_neurons = 514
    #keras.config.enable_unsafe_deserialization()


    # for c in combinations_2d:
    #     print(np.array(c))
    # exit()

    logging.info("Loading models")
    encoder = keras.models.load_model(f"{model_filepath}/encoder_{n_neurons}.keras")
    decoder = keras.models.load_model(f"{model_filepath}/decoder_{n_neurons}.keras")

    #ttt.compile(optimizer="AdamW", loss="mse")
    # Freeze the weights

    it = ArcExampleData('train')

    decoder.trainable = False
    encoder.trainable = False

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

        #train_x_a = np.concatenate([train_x, train_x_a])
        #train_y_a = np.concatenate([train_y, train_y_a])
        #print(train_x_a.shape, train_y_a.shape, "£23424234")
        #exit()


        train_y_one_hot_a = np.eye(11)[np.array(train_y_a, dtype=np.int32)]
        train_y_one_hot = np.eye(11)[np.array(train_y, dtype=np.int32)]
        test_y_one_hot = np.eye(11)[np.array(test_y, dtype=np.int32)]


        input = keras.layers.Input([32,32,1])
        input_dec = keras.layers.Input([n_neurons,])

        ttt = keras.layers.Dense(514)



        #
        # _, _, encoder, _, ttt = build_model((32, 32),
        #                                                          int(11),
        #                                                          n_neurons)
        ev = keras.layers.Dense(n_neurons, activation = "linear")
       # bn = keras.layers.BatchNormalization()
        x = (ev(ttt(encoder(input))))
        #x = keras.layers.Dense(n_neurons, activation = "relu")(x)
        autoencoder = keras.Model(input, decoder(x))
        #optimizer = keras.optimizers.SGD(0.01)
        optimizer = keras.optimizers.AdamW(0.0001)
        autoencoder.compile(optimizer=optimizer,

                        loss='categorical_crossentropy',#run_eagerly=True,
                        metrics=["acc", b_acc])

        # from cma import CMAEvolutionStrategy
        # # Initialize CMA-ES with the initial mean and standard deviation
        #
        # from utils import NNWeightHelper
        #
        # nwe = NNWeightHelper(model=ev)
        #
        # w = nwe.get_weights()
        # #w = ev.get_weights()
        # #print(w.size)
        # #exit()
        #
        # initial_mean = w  # Starting point
        # initial_sigma = 0.5  # Initial standard deviation
        #
        # # Create an instance of the CMAEvolutionStrategy
        # opts = {}
        # opts['CMA_diagonal'] = True
        # es = CMAEvolutionStrategy(initial_mean, initial_sigma,inopts = opts)
        #
        # # Optimization loop using the ask-tell API
        # while not es.stop():
        #     # Generate a new population of candidate solutions
        #     solutions = es.ask(100)
        #
        #     # Evaluate the objective function for each candidate solution
        #     fitnesses = []
        #     print(len(solutions))
        #     for x in solutions:
        #         nwe.set_weights(x)
        #         score = autoencoder.evaluate(train_x_a, train_y_one_hot_a,
        #                                      return_dict=True, verbose=False)["acc"]
        #
        #         fitnesses.append(-score)
        #         #print(score)
        #         #exit()
        #
        #
        #
        #
        #     # Tell the optimizer the fitnesses of the candidate solutions
        #     es.tell(solutions, fitnesses)
        #
        #     # Optional: Print the current best solution and its fitness
        #     print(f"Best solution so far: {es.result.xbest}")
        #     print(f"Best fitness so far: {es.result.fbest}")

        #
        autoencoder.fit(train_x_a, train_y_one_hot_a, batch_size=16,
                        validation_data=(test_x, test_y_one_hot), verbose = False,
                        epochs = 2000, callbacks=[TqdmCallback(verbose=0)])
        score = autoencoder.evaluate(train_x_a, train_y_one_hot_a)
        print(score)
        #
        #
        #


        train_output = autoencoder.predict(train_x, verbose = False).argmax(axis = -1)
        test_output = autoencoder.predict(test_x, verbose = False).argmax(axis = -1)





        #r["output"] = np.concatenate([train_output, test_output])
        # #print(r["output"].shape)
        r["output"] = np.concatenate([train_output, test_output])
        visualise_training_data(r, f"./plots/{r['name']}_predicted.pdf", )
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
