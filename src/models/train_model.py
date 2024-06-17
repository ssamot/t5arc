from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig
from transformers import TrainingArguments, Trainer
import torch
import click
import pandas as pd
import glob
import os
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import secrets

class NLP2SQL(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, target_ids):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.target_ids = target_ids
        print(input_ids.shape, target_ids.shape)
        #exit()

    def to_cuda(self):
        self.input_ids.to("cuda")
        self.attention_masks.to("cuda")
        self.target_ids.to("cuda")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        target_ids = self.target_ids[idx]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': target_ids,
        }

def load_data(input_filepath, type,tokenizer, bootstrap ):

    pickle_files = glob.glob(os.path.join(os.path.join(input_filepath, type), "*.pkl"))



    dfs = []
    # Loop through the pickle files
    for file_path in pickle_files:
        # Load the DataFrame from the pickle file
        df = pd.read_pickle(file_path)
        logging.info(f"Read data file {file_path} with {len(df)} data points")
        dfs.append(df)
    # Merge the loaded DataFrame into the main merged DataFrame
    train_merged_df = pd.concat(dfs, ignore_index=True)

    # random sample
    if(bootstrap):
        logging.info("Bootstrapping..")
        train_merged_df = train_merged_df.sample(frac=1, replace=True)

    logging.info(f"Read {type} data")

    # the following 2 hyperparameters are task-specific

    encoding = tokenizer(
        train_merged_df['nl_inputs'].tolist(),
        padding="longest",
        truncation=False,
        return_tensors="pt",
    )

    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

    # encode the targets
    #print(train_merged_df['queries'].tolist())
    target_encoding = tokenizer(
        train_merged_df['queries'].tolist(),
        padding="longest",
        truncation=False,
        return_tensors="pt",
    )
    labels = target_encoding.input_ids
    #print(encoding.input_ids.shape)
    #print(target_encoding.input_ids.shape)

    # replace padding token id's of the labels by -100 so it's ignored by the loss
    labels[labels == tokenizer.pad_token_id] = -100


    #target_encoding.to('cuda')
    #encoding.to('cuda')


    return NLP2SQL(input_ids=input_ids, attention_masks=attention_mask, target_ids=labels),train_merged_df



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('pretrained', type=click.STRING)
@click.argument('bootstrap', type=click.BOOL)
def main(input_filepath, output_filepath,pretrained, bootstrap):

    bootstrap = bool(bootstrap)

    configuration = AutoConfig.from_pretrained(pretrained)
    #configuration.dropout_rate = 0.2

    training_args = TrainingArguments(output_dir="test_trainer",
                                      evaluation_strategy="epoch",
                                      optim="adamw_torch",
                                      per_device_train_batch_size=2,
                                      per_device_eval_batch_size=8,
                                      # gradient_checkpointing = True,
                                      gradient_accumulation_steps=3,
                                      torch_compile=True,

                                      logging_dir=f"./logs",
                                      logging_strategy="steps",
                                      logging_steps=200,
                                      report_to="tensorboard",

                                      # fp16 = True
                                      )

    training_args.set_lr_scheduler("constant", num_epochs=10)
    training_args.set_optimizer("adamw_torch", learning_rate=0.0001,
                                weight_decay=0.001)


    #training_args.set_optimizer("adamw_torch", weight_decay=0.01, learning_rate=0.0001)
    #training_args.set_lr_scheduler("constant")

    new_words = ['<', '<=']


    tokenizer = T5Tokenizer.from_pretrained(pretrained, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(pretrained, config = configuration)

    tokenizer.add_tokens(new_words)
    model.resize_token_embeddings(len(tokenizer))


    # Get the vocabulary and find tokens mapped to <unk>


    #model.to('cuda')



    train_data = load_data(input_filepath, "train",tokenizer, bootstrap)[0]
    dev_data = load_data(input_filepath, "dev",tokenizer, False)[0]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,

        # compute_metrics=compute_metrics,
    )

    trainer.train()

    model.to('cuda')
    train_data.to_cuda()
    dev_data.to_cuda()

    if(bootstrap):
        model_id = secrets.token_hex(nbytes=16)
    else:
        model_id = "full_data_model"
    model_id = model_id + ".hf"
    tokenizer.save_pretrained(os.path.join(output_filepath, model_id))
    model.save_pretrained(os.path.join(output_filepath, model_id))



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
