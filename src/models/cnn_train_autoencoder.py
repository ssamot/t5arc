import keras
import click
import logging

import numpy as np
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from utils import CustomModelCheckpoint
from cnn_models import get_components
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
        original_images, new_images = generate_samples(True)
        images = np.concatenate([original_images, new_images])
        return images-0.5, images


@click.command()
@click.argument('input_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    # build_model()
    # Initialize a list to store the contents of the JSON files

    squeeze_neurons = 64
    base_filters = 128
    encoder_filters = 64
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
    validation_x = []
    validation_y = []
    for task in it:
        s = np.array(task["input"], dtype=np.int32)
        s = np.eye(11)[s]
        sprime = np.array(task["output"], dtype=np.int32)
        sprime = np.eye(11)[sprime]

        validation_x.append(s)
        validation_x.append(sprime)
        validation_y.append(s)
        validation_y.append(sprime)

    it = ArcTaskData('eval')
    validation_x = []
    validation_y = []
    for task in it:
        s = np.array(task["input"], dtype=np.int32)
        s = np.eye(11)[s]
        sprime = np.array(task["output"], dtype=np.int32)
        sprime = np.eye(11)[sprime]

        validation_x.append(s)
        validation_x.append(sprime)
        validation_y.append(s)
        validation_y.append(sprime)


    validation_x = np.concatenate(validation_x, axis = 0) - 0.5
    validation_y = np.concatenate(validation_y, axis = 0)
    print("Validation shape:",validation_x.shape)
    validation_data = (validation_x, validation_y)

    model.summary()
    models = {f"s_ae_encoder_{squeeze_neurons}_{base_filters}_{encoder_filters}": s_encoder,
              f"sprime_ae_decoder_{squeeze_neurons}_{base_filters}_{encoder_filters}": sprime_decoder,
              }


    # optimizer = keras.optimizers.SGD(gradient_accumulation_steps=50,
    #                                  learning_rate=0.001, momentum=0.9,
    #                                  weight_decay=0.001)

    optimizer = keras.optimizers.Adamax(gradient_accumulation_steps=50,
                                     learning_rate=0.0001, weight_decay=0.00001)
    model.compile(optimizer=optimizer,
                  loss=keras.losses.categorical_crossentropy,
                  metrics=["acc", b_acc, cce],
                  )
    model.fit(x=training_generator, validation_batch_size=32,
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
