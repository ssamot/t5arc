import numpy as np
from transformers import ReformerModelWithLMHead, ReformerConfig
from transformers import TrainingArguments, Trainer
from transformers import DataCollator
import torch
import click
import pandas as pd
import glob
import os
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import secrets
from datasets import Dataset


# Load data from file


class ReformerCollator(DataCollator):
    def __init__(self, max_roll_length):
        self.max_roll_length = max_roll_length

    # From the official notebook: "Normally we would have a dataset with many examples, but for this demonstration we fit a language model on the single novel only. We don't want the model to just memorize the dataset by encoding the words in its position embeddings, so at each training iteration we will randomly select how much padding to put before the text vs. after it"
    def collate_batch(self, features):
        # get random shift int
        random_shift_length = torch.randint(self.max_roll_length, (1,)).item()

        # shift input and mask
        rolled_input_ids = torch.roll(
            features[0]["input_ids"], random_shift_length
        ).unsqueeze(0)
        rolled_attention_mask = torch.roll(
            features[0]["attention_mask"], random_shift_length
        ).unsqueeze(0)

        # return dict having the correct argument naming
        return {
            "input_ids": rolled_input_ids,  # BS x SEQ_LEN
            "labels": rolled_input_ids,  # BS x SEQ_LEN
            "attention_mask": rolled_attention_mask,  # BS x SEQ_LEN
        }

class TorchDataSet(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks):
        self.input_ids = input_ids
        self.attention_masks = attention_masks

        #exit()

    def to_cuda(self):
        self.input_ids.to("cuda")
        self.attention_masks.to("cuda")


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,

        }

def load_data(input_filepath,tokenizer ):
    def load_data_from_directory(directory_path):
        data = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".json"):
                with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    data.extend(lines)
        print("total lines:", len(data))
        return data


    data = load_data_from_directory(input_filepath)


    # the following 2 hyperparameters are task-specific

    encoding = tokenizer(
        data,
        padding="max_length",
        truncation=False,
        return_tensors="pt",
    )

    input_ids, attention_mask = encoding.input_ids, encoding.attention_mask



    return TorchDataSet(input_ids=input_ids, attention_masks=attention_mask)



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('bootstrap', type=click.BOOL)
def main(input_filepath, output_filepath, bootstrap):

    bootstrap = bool(bootstrap)

    from tokenizer import CharacterTokenizer
    import string
    chars = string.ascii_letters  # This is character vocab

    axial_shape =  [128, 256]
    model_max_length = np.multiply(*axial_shape)
    #print(model_max_length)
    #exit()
    tokenizer = CharacterTokenizer(chars, model_max_length)

    tokenized_dataset = load_data(input_filepath, tokenizer)


    tokenized_dataset = Dataset.from_list(tokenized_dataset)

    # Preprocess the dataset

    #exit()

    # Split the dataset into train and test sets
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    config = {
        "attention_head_size": 64,
        "attn_layers": ["local", "lsh", "local", "lsh", "local", "lsh"],
        "axial_pos_embds": True,
        "sinusoidal_pos_embds": False,
        "axial_pos_embds_dim": [64, 192],
        "axial_pos_shape": axial_shape,
        "lsh_attn_chunk_length": 64,
        "local_attn_chunk_length": 64,
        "feed_forward_size": 512,
        "hidden_act": "relu",
        "hidden_size": 256,
        "is_decoder": True,
        "max_position_embeddings": model_max_length,
        "num_attention_heads": 2,
        "num_buckets": [64, 128],
        "num_hashes": 1,
        "vocab_size": 320,
        "lsh_attention_probs_dropout_prob": 0.0,
        "lsh_num_chunks_before": 1,
        "lsh_num_chunks_after": 0,
        "local_num_chunks_before": 1,
        "local_num_chunks_after": 0,
        "local_attention_probs_dropout_prob": 0.025,
        "hidden_dropout_prob": 0.025,
    }

    # Create the model
    config = ReformerConfig(**config)
    model = ReformerModelWithLMHead(config)

    #configuration.dropout_rate = 0.2

    training_args = TrainingArguments(output_dir="test_trainer",
                                      evaluation_strategy="epoch",
                                      optim="adamw_torch",
                                      per_device_train_batch_size=2,
                                      per_device_eval_batch_size=8,
                                      # gradient_checkpointing = True,
                                      gradient_accumulation_steps=3,
                                      torch_compile=False,

                                      logging_dir=f"./logs",
                                      logging_strategy="steps",
                                      logging_steps=200,
                                      #report_to="tensorboard",

                                      # fp16 = True
                                      )

    training_args.set_lr_scheduler("constant", num_epochs=10)
    training_args.set_optimizer("adamw_torch", learning_rate=0.0001,
                                weight_decay=0.001)


    #training_args.set_optimizer("adamw_torch", weight_decay=0.01, learning_rate=0.0001)
    #training_args.set_lr_scheduler("constant")



    # Get the vocabulary and find tokens mapped to <unk>


    #model.to('cuda')

    def compute_metrics(pred):
        non_padded_indices = (pred.label_ids != -100)

        # correctly shift labels and pred as it's done in forward()
        labels = pred.label_ids[..., 1:][non_padded_indices[..., 1:]]
        pred = np.argmax(pred.predictions[:, :-1], axis=-1)[non_padded_indices[..., :-1]]

        acc = np.mean(np.asarray(pred == labels), dtype=np.float)
        return {"accuracy": acc}

    non_padded_sequence_length = model_max_length - sum(
        tokenized_dataset["attention_mask"][0]
    )

    data_collator = ReformerCollator(non_padded_sequence_length)

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
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
