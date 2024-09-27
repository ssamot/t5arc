import keras
import click
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from utils import CustomModelCheckpoint
from cnn_models import get_components, build_end_to_end
from cnn_data_generator import DataGenerator
from lm import b_acc



@click.command()
@click.argument('output_filepath', type=click.Path())
def main(output_filepath):
    # build_model()
    # Initialize a list to store the contents of the JSON files

    encoder_units = 16


    training_generator = DataGenerator(len = 1000,
                                             use_multiprocessing=True, workers=50,
                                             max_queue_size=1000)

    (s_input, ssprime_input,
     s_encoder, ssprime_encoder, sprime_decoder,
     param_layer, squeeze_layer) = get_components(encoder_units)

    model = build_end_to_end(s_input, ssprime_input,
                     s_encoder, ssprime_encoder, sprime_decoder, param_layer )


    model.summary()
    models = {f"s_encoder_{encoder_units}": s_encoder,
              f"ssprime_encoder{encoder_units}": ssprime_encoder,
              f"sprime_decoder{encoder_units}": sprime_decoder,
              f"param_layer{encoder_units}": param_layer,
              f"squeeze_layer{encoder_units}": squeeze_layer,
              }

    optimizer = keras.optimizers.AdamW(gradient_accumulation_steps=10)
    model.compile(optimizer=optimizer,
                  loss=keras.losses.categorical_crossentropy,
                  metrics = ["acc", b_acc],
                  )
    model.fit(x=training_generator,
              epochs=10000,
              callbacks=CustomModelCheckpoint(models,"./models", 100))



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
