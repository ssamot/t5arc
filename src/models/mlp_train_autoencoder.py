import keras
import click
import logging

import numpy as np
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from utils import CustomModelCheckpoint
from mlp_models import get_components
from build_models import build_autoencoder
from cnn_data_generator import generate_samples
from lm import b_acc, cce
from tqdm.keras import TqdmCallback
from data.generators.task_generator.ttt_data_generator import ArcTaskData


class AutoencoderDataGenerator(keras.utils.PyDataset):
    def __init__(self, len, **kwargs):
        super().__init__(**kwargs)
        self.len = len
        self.on_epoch_end()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        original_images, new_images = generate_samples()
        return new_images, new_images


@click.command()
@click.argument('input_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    # build_model()
    # Initialize a list to store the contents of the JSON files

    squeeze_neurons = 64
    base_filters = 1024
    encoder_filters = 1024
    # samples = np.load(input_filepath, allow_pickle=True)["samples"]
    # training_generator = BatchedDataGenerator(samples)

    training_generator = AutoencoderDataGenerator(len=10000,
                                                  use_multiprocessing=True, workers=100,
                                                  max_queue_size=1000)

    (s_input, _,
     s_encoder, _, sprime_decoder,
     _, _, _) = get_components(squeeze_neurons,
                            base_filters,
                            encoder_filters)

    model = build_autoencoder(s_input,
                              s_encoder,
                              sprime_decoder)

    ## find validation data;

    it = ArcTaskData('train')

    relevant = ["b775ac94"]

    for task in it:
        if (task["name"] in relevant):
            s = np.array(task["input"], dtype=np.int32)
            s = np.eye(11)[s]
            sprime = np.array(task["output"], dtype=np.int32)
            sprime = np.eye(11)[sprime]

            #ssprime = merge_arrays(s, sprime)
            validation_data = sprime,sprime
            break

    model.summary()
    models = {f"s_encoder_mlp_{squeeze_neurons}_{base_filters}_{encoder_filters}": s_encoder,
              f"s_decoder_mlp{squeeze_neurons}_{base_filters}_{encoder_filters}": sprime_decoder,
              }

    optimizer = keras.optimizers.Adamax(gradient_accumulation_steps=50,
                                     learning_rate=0.001, weight_decay=0.0001)
    model.compile(optimizer=optimizer,
                  loss=keras.losses.categorical_crossentropy,
                  metrics=["acc", b_acc, cce],
                  )
    model.fit(x=training_generator, validation_batch_size=1000,
              validation_data=validation_data,
              epochs=10000, verbose=False,
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
