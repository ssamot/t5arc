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
from build_models import build_ssprime_ave_model, build_ttt_emb, build_ttt, build_end_to_end
from cnn_models import BatchAverageLayer, get_components

from data.generators.task_generator.ttt_data_generator import ArcTaskData
from visualization.visualse_training_data_sets import visualise_training_data
from tqdm.keras import TqdmCallback
from data.augment.colour import generate_consistent_combinations_2d
from keras import layers
from cnn_data_generator import merge_arrays





@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
@click.argument('data_type', type=click.Path())
def main(data_filepath, model_filepath, output_filepath, data_type):

    n_neurons = 1024
    keras.config.enable_unsafe_deserialization()

    squeeze_neurons = 63
    base_filters = 64
    encoder_filters = 32

    logging.info("Loading models")

    s_encoder       = f"s_ee_encoder_{squeeze_neurons}_{base_filters}_{encoder_filters}"
    ssprime_encoder = f"ssprime_ee_encoder_{squeeze_neurons}_{base_filters}_{encoder_filters}"
    sprime_decoder  = f"sprime_ee_decoder_{squeeze_neurons}_{base_filters}_{encoder_filters}"
    param_layer     = f"param_ee_layer_{squeeze_neurons}_{base_filters}_{encoder_filters}"
    squeeze_layer   = f"squeeze_ee_layer_{squeeze_neurons}_{base_filters}_{encoder_filters}"



    logging.info(f"Loading model {s_encoder}")
    s_encoder = keras.models.load_model(f"{model_filepath}/{s_encoder}.keras")
    s_encoder.trainable = False

    logging.info(f"Loading model {ssprime_encoder}")
    ssprime_encoder = keras.models.load_model(f"{model_filepath}/{ssprime_encoder}.keras")
    ssprime_encoder.trainable = False

    logging.info(f"Loading model {sprime_decoder}")
    sprime_decoder = keras.models.load_model(f"{model_filepath}/{sprime_decoder}.keras")
    sprime_decoder.trainable = False

    logging.info(f"Loading model {param_layer}")
    param_layer = keras.models.load_model(f"{model_filepath}/{param_layer}.keras")
    param_layer.trainable = False

    logging.info(f"Loading model {squeeze_layer}")
    squeeze_layer = keras.models.load_model(f"{model_filepath}/{squeeze_layer}.keras")
    squeeze_layer.trainable = False



    ssprime_input = keras.layers.Input((32,32,22))
    s_input = keras.layers.Input((32, 32, 11))
    ave_model = build_ssprime_ave_model(ssprime_input,
                                        ssprime_encoder,
                                        param_layer, BatchAverageLayer)


    ttt_model = build_ttt(s_input, ssprime_input,
              s_encoder, ssprime_encoder, sprime_decoder, squeeze_layer)

    # (s_input, ssprime_input,
    #  s_encoder, ssprime_encoder, sprime_decoder,
    #  param_layer, squeeze_layer, BatchAverageLayer) = get_components(squeeze_neurons,
    #                                                                  base_filters,
    #                                                                  encoder_filters)

    end_to_end = build_end_to_end(s_input, ssprime_input,
                     s_encoder, ssprime_encoder, sprime_decoder, param_layer, BatchAverageLayer)


    optimizer = keras.optimizers.Adamax(gradient_accumulation_steps=20,
                                        learning_rate=0.0001,
                                        weight_decay=0.00001,
                                        epsilon=0.00001)

    end_to_end.compile(optimizer=optimizer,
                      loss=keras.losses.categorical_crossentropy,
                      metrics=["acc", b_acc],
                      )
    ## create the ttt and other models



    it = ArcTaskData('train')
    relevant = ["05f2a901"]

    for r in tqdm(it):

        if r["name"] not in relevant:
            continue
        print(r["name"])

        # visualise_training_data(r, f"./plots/{r['name']}.pdf",)

        train_x = np.array(r["input"][:-1], dtype=np.int32)
        test_x = np.array(r["input"][-1:], dtype=np.int32)


        train_y = r["output"][:-1]
        test_y = r["output"][-1:]



        train_s = np.eye(11)[train_x]
        #all_s = np.eye(11)[all_x]
        train_sprime = np.eye(11)[train_y]
        train_ssprime = np.concatenate([train_s, train_sprime], axis=-1)

        test_s = np.eye(11)[test_x]
        test_sprime = np.eye(11)[test_y]
        #test_ssprime = np.concatenate([test_s, train_sprime], axis=-1)

        ave_values, params = ave_model.predict(train_ssprime, verbose=False)
        #print(params.shape)
        #exit()
        #ave_values = np.vstack([ave_values, ave_values[0:1]])

        dummy = np.ones(shape=(ave_values.shape[0],1,1 ))
        dumm_test = np.ones(shape=(test_s.shape[0],1,1 ))

        optimizer = keras.optimizers.Adamax(learning_rate=0.001,
                                             weight_decay=0.00001)


        ttt_model.compile(optimizer=optimizer,
                      loss=keras.losses.categorical_crossentropy,
                      metrics=["acc", b_acc],
                      )
        #print(dummy.shape)



        #outcome = end_to_end.evaluate([train_s,train_ssprime],train_sprime,return_dict=True, batch_size=30000  )
        #print(outcome)
        #exit()

        #dummy[0] = 1
        #dummy[1] = 2
        #dummy[2] = 3
        #print(dummy)
        #exit()
        # ttt_model.fit([train_s, ave_values,dummy],train_sprime,
        #               validation_data=([test_s,ave_values[0:1],dumm_test],test_sprime),
        #               verbose=True, epochs=10000, batch_size=3000)

        #
        from cma import CMAEvolutionStrategy

        initial_mean = [0.0] * squeeze_neurons # Starting point
        initial_sigma = 0.3  # Initial standard deviation

        # Create an instance of the CMAEvolutionStrategy
        opts = {}
        #opts['CMA_diagonal'] = False
        opts["bounds"] = [-1,1]
        es = CMAEvolutionStrategy(initial_mean, initial_sigma,inopts = opts)

        # Optimization loop using the ask-tell API
        for i in range(1):
            # Generate a new population of candidate solutions
            solutions = es.ask(100)

            # Evaluate the objective function for each candidate solution
            fitnesses = []
            #print(len(solutions))
            for x in solutions:

                #score = autoencoder.evaluate(train_x_a, train_y_one_hot_a,
                #                             return_dict=True, verbose=False)["acc"]

                array = x.reshape(1, squeeze_neurons)

                # Repeat it to get the shape (5, 5, 64)
                output = np.repeat(array, repeats=train_s.shape[0], axis=0)  # Repeat along the first axis (5 times)
                #output = np.repeat(output, repeats=5, axis=1)  # Repeat along the second axis (5 times)
                #print(train_s.shape, ave_values.shape, output.shape)
                score = ttt_model.evaluate([train_s, ave_values,output],train_sprime, verbose = False)
                #print(score)

                fitnesses.append(-score[1])
                #print(score)
                #exit()




            # Tell the optimizer the fitnesses of the candidate solutions
            es.tell(solutions, fitnesses)

            # Optional: Print the current best solution and its fitness
            print(f"Best solution so far: {es.result.xbest}")
            print(f"Best fitness so far: {es.result.fbest}")

        best_att = es.result.xbest
        best_att = best_att.reshape(1, squeeze_neurons)
        best_att = np.repeat(best_att, repeats=train_s.shape[0], axis=0)

        train_output = ttt_model.predict([train_s, ave_values,best_att], batch_size=30000)
        test_output = ttt_model.predict([test_s, ave_values[0:1], best_att[0:1]], batch_size=30000)





        #r["output"] = np.concatenate([train_output, test_output])
        # #print(r["output"].shape)
        visualise_training_data(r, f"./plots/{r['name']}_predicted.pdf", )
        exit()
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
