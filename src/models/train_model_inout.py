import click
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from utils import CustomModelCheckpoint, build_NMF
import numpy as np
from tqdm.keras import TqdmCallback
from tqdm import tqdm



@click.command()
@click.argument('train_data', type=click.Path())
@click.argument('eval_data', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(train_data, eval_data, output_filepath):
    encoder_units = 514
    programme_units = 32

    print("Loading data...")
    train_data = np.load(train_data, allow_pickle=True)
    train_train_x = train_data["train_x"]
    train_test_x = train_data["test_x"]
    train_train_y = train_data["train_y"]
    train_test_y = train_data["test_y"]

    eval_data = np.load(eval_data, allow_pickle=True)
    eval_train_x = eval_data["train_x"]
    eval_test_x = eval_data["test_x"]
    eval_train_y = eval_data["train_y"]
    eval_test_y = eval_data["test_y"]

    #data_parallel = keras.distribution.DataParallel()
    #keras.distribution.set_distribution(data_parallel)

    X = []
    y = []
    programmes = []
    for batch in tqdm(range(len(train_train_x))):
        train_x = np.array(train_train_x[batch], dtype=np.int32)
        train_y = np.array(train_train_y[batch], dtype=np.int32)

        train_x = np.eye(11)[train_x]
        train_y = np.eye(11)[train_y]

        tr_x = [[0] for _ in range(len(train_x))]
        tr_y = [[batch] for _ in range(len(train_x))]
        #tr_x = np.array(tr)

        X.append(train_x)
        programmes.extend(tr_x)
        y.append(train_x)

        X.append(train_x)
        programmes.extend(tr_y)
        y.append(train_y)

    X = np.concatenate(X)
    y = np.concatenate(y)
    programmes = np.concatenate(programmes)
    programmes = programmes[:, np.newaxis]

    encoder, decoder, autoencoder, ttt = build_NMF(encoder_units, programme_units,
                                              len(train_train_x) + 1,
                                              X.shape[1:])

    autoencoder.summary()
    decoder.summary()

    print(X.shape, y.shape, programmes.shape)
    models = {
        f"decoder_{encoder_units}": decoder,
        f"encoder_{encoder_units}": encoder,
        f"ttt_{encoder_units}": ttt,

    }

    autoencoder.fit([X, programmes], y, epochs=10000, verbose=0, batch_size=128,
                    callbacks=[CustomModelCheckpoint(models, "./models", 100),
                               TqdmCallback()])


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
