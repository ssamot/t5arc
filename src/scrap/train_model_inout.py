import click
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from utils import CustomModelCheckpoint, build_model, build_NMF
import numpy as np
from tqdm.keras import TqdmCallback
from tqdm import tqdm
import jax
import keras
from sklearn.metrics import accuracy_score


from utils import average_maps


@click.command()
@click.argument('train_data', type=click.Path())
@click.argument('eval_data', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(train_data, eval_data, output_filepath):
    # build_model()
    # Initialize a list to store the contents of the JSON files

    max_pad_size = 32
    encoder_units = 514
    num_decoder_tokens = 11

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

    # data_parallel = keras.distribution.DataParallel()
    # keras.distribution.set_distribution(data_parallel)





    images = []
    programmes = []
    for batch in tqdm(range(len(train_train_x))):
        train_x = np.array(train_train_x[batch], dtype=np.int32)
        train_y = np.array(train_train_y[batch], dtype=np.int32)



        train_x = np.eye(11)[train_x]
        train_y = np.eye(11)[train_y]

        tr_x = [[0] for _ in range(len(train_x))]
        tr_y = [[batch] for _ in range(len(train_x))]
        #tr_x = np.array(tr)


        images.append(train_x)
        programmes.extend(tr_x)
        programmes.extend(tr_y)
        images.append(train_y)

    images = np.concatenate(images)
    print(images.shape)
    decoder = build_NMF(512, 10,
                        len(images)+1,
                        len(train_train_x)+1,
                        images.shape[1:])

    decoder.summary()
    models = {
              f"decoder_{encoder_units}": decoder,
              }

    X = [[i] for i in range(len(images))]
    #print(X)
    X = np.array(X, dtype="int32")
    programmes = np.array(programmes)
    X = X[:,:, np.newaxis]
    programmes = programmes[:,:, np.newaxis]
    print(X.shape, programmes.shape)

    from sklearn.decomposition import MiniBatchNMF

    n_components = 512

    clf = MiniBatchNMF(n_components=n_components,max_iter=1000,verbose=1000, batch_size=10000)
    news = images.shape[1]* images.shape[2]* images.shape[3]
    X_train_pca = clf.fit_transform(images.reshape(-1,news))

    X_train_reconstructed = clf.inverse_transform(X_train_pca)

    X_train_reconstructed = np.where(X_train_reconstructed < 0.5, 0, 1)

    # Calculate reconstruction error using Mean Squared Error (MSE)
    mse = accuracy_score(images.reshape(-1,news), X_train_reconstructed)
    print(mse, n_components)

    #
    # decoder.fit([X, programmes], images, epochs=10000, verbose=0, batch_size=128,
    #           callbacks=[CustomModelCheckpoint(models, "./models", 100),
    #                      TqdmCallback()])


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
