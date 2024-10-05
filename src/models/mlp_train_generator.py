import keras
import click
import logging

import numpy as np
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from utils import CustomModelCheckpoint
from mlp_models import build_autoencoder, get_components, build_end_to_end
from cnn_data_generator import EndToEndDataGenerator, BatchedDataGenerator, merge_arrays
from lm import b_acc, cce
from tqdm.keras import TqdmCallback
from data.generators.task_generator.ttt_data_generator import ArcTaskData




@click.command()
@click.argument('input_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    # build_model()
    # Initialize a list to store the contents of the JSON files

    squeeze_neurons = 64
    base_filters = 512
    encoder_filters = 1024
    # samples = np.load(input_filepath, allow_pickle=True)["samples"]
    # training_generator = BatchedDataGenerator(samples)


    training_generator = EndToEndDataGenerator(len = 10000,
                                             use_multiprocessing=True, workers=100,
                                             max_queue_size=1000)

    (s_input, ssprime_input,
     s_encoder, ssprime_encoder, sprime_decoder,
     param_layer, squeeze_layer) = get_components(squeeze_neurons,
                            base_filters,
                            encoder_filters)

    model = build_end_to_end(s_input, ssprime_input,
                     s_encoder, ssprime_encoder, sprime_decoder, param_layer )


    ## find validation data;

    it = ArcTaskData('train')

    relevant = ["b775ac94"]


    for task in it:
        if(task["name"] in relevant):
            s = np.array(task["input"], dtype=np.int32)
            s = np.eye(11)[s]
            sprime = np.array(task["output"], dtype=np.int32)
            sprime = np.eye(11)[sprime]


            ssprime = merge_arrays(s, sprime)
            validation_data = (s,ssprime), s
            break




    model.summary()
    models = {f"s_encoder_{squeeze_neurons}_{base_filters}_{encoder_filters}": s_encoder,
              f"ssprime_encoder_{squeeze_neurons}_{base_filters}_{encoder_filters}": ssprime_encoder,
              f"sprime_decoder_{squeeze_neurons}_{base_filters}_{encoder_filters}": sprime_decoder,
              f"param_layer_{squeeze_neurons}_{base_filters}_{encoder_filters}": param_layer,
              f"squeeze_layer_{squeeze_neurons}_{base_filters}_{encoder_filters}": squeeze_layer,
              }

    optimizer = keras.optimizers.AdamW(gradient_accumulation_steps=50)
    model.compile(optimizer=optimizer,
                  loss=keras.losses.categorical_crossentropy,
                  metrics = ["acc", b_acc, cce],
                  )
    model.fit(x=training_generator,validation_batch_size=1000,
              validation_data=validation_data,
              epochs=10000,verbose=False,
              callbacks=[CustomModelCheckpoint(models, output_filepath, 100),
                         TqdmCallback(verbose=1)
                         ]
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
