import click
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from utils import CustomModelCheckpoint, build_model
import numpy as np
from tqdm.keras import TqdmCallback
from tqdm import tqdm





@click.command()
@click.argument('train_data', type=click.Path())
@click.argument('eval_data', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(train_data, eval_data, output_filepath):
    # build_model()
    # Initialize a list to store the contents of the JSON files

    max_pad_size = 32
    encoder_units = 512
    num_decoder_tokens = 11



    _, twin_autoencoder, encoder, decoder, ttt = build_model((max_pad_size, max_pad_size),
                                          int(num_decoder_tokens),
                                          encoder_units)


    twin_autoencoder.summary()
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



    models = {f"encoder_{encoder_units}": encoder,
              f"decoder_{encoder_units}": decoder,
              f"ttt_{encoder_units}": ttt,
             # f"twin_autoencoder_{encoder_units}": twin_autoencoder
              }



    n_epochs = 100000
    order = list(range(len(train_train_x)))
    save_freq = 100
    with (tqdm(total=n_epochs, desc='Epochs') as outer_pbar):
        with tqdm(total=len(train_train_x), colour="green") as pbar:
            for epoch in range(n_epochs):
                np.random.shuffle(order)
                pbar.reset(total=len(train_train_x))
                twin_autoencoder.reset_metrics()
                for batch in order:
                    train_x = np.array(train_train_x[batch], dtype=np.int32)
                    train_y = np.array(train_train_y[batch], dtype=np.int32)

                    train_x,train_y = np.concatenate([train_x, train_y], dtype=np.int32
                                                     ), np.concatenate([train_y, train_y],  dtype=np.int32)

                    #print(train_train_x[i].shape)
                    #print(train_x.shape, train_y.shape)

                    losses = twin_autoencoder.train_on_batch([train_x, train_y],
                                                             np.eye(11)[train_x],
                                                             return_dict=True,
                                                             )


                    r2 = losses["cce"] - losses["loss"]
                    losses["r2"] = r2
                    pbar.set_postfix({key: f'{value:.3f}' for key, value in losses.items()}
)
                    pbar.update(1)


                    #print(losses)
                if (epoch % save_freq == 0):
                    # print(f"Saving epoch: {epoch}, train_acc: {logs['acc']}, : {logs['batch_acc']}")
                    for name in models:
                        model = models[name]
                        model.save(f"{output_filepath}/{name}.keras", overwrite=True)

                outer_pbar.update(1)
        #print(losses)
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
