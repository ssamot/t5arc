from transformers import T5Tokenizer, T5ForConditionalGeneration, PhrasalConstraint
from transformers import TrainingArguments, Trainer
import torch
import click
import pandas as pd
import glob
import os
import logging
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from models.train_model import load_data
from tqdm import tqdm
from test_against_database import test_sqls



def replace_tabs_and_spaces(input_string):
    # Replace tabs with spaces
    cleaned_string = input_string.replace('\t', ' ')

    # Replace double and triple spaces with a single space
    cleaned_string = ' '.join(cleaned_string.split())

    return cleaned_string


@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
@click.argument('data_type', type=click.Path())
def main(data_filepath, model_filepath, output_filepath, data_type):


    logging.info("Loading models")

    tokenizer = T5Tokenizer.from_pretrained(model_filepath, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_filepath)


    data = load_data(data_filepath, data_type, tokenizer, False)
    model.to("cuda")
    #data[0].to_cuda()

    val = data[0].input_ids
    att = data[0].attention_masks
    inputs = data[1]['nl_inputs'].tolist()

    queries = data[1]['queries'].tolist()
    db_ids = data[1]['db_ids'].tolist()
    table_names = data[1]['table_maps'].tolist()
    column_maps = data[1]['column_maps'].tolist()
    all_lookup_columns = data[1]['all_lookup_columns'].tolist()
    all_lookup_tables = data[1]['all_lookup_tables'].tolist()
    fks = data[1]['fks'].tolist()
    schemas = data[1]['schemas'].tolist()

    batch_size = 10
    data_len = len(val)
    #data_len = 10
    num_return_sequences = 10
    sqls = []
    for i in tqdm(range(0,data_len , batch_size)):
        batch_prompts = val[i:i+batch_size]
        batch_mask = att[i:i+batch_size]
        batch_db_ids = db_ids[i:i+batch_size]
        #batch_tables_used = tables_used[i:i+batch_size]
        batch_inputs = inputs[i:i+batch_size]

        batch_fks = fks[i:i + batch_size]
        batch_schemas = schemas[i:i + batch_size]



        generator_outputs = model.generate(batch_prompts.to("cuda"),
                                attention_mask=batch_mask.to("cuda"),
                                max_new_tokens=512,
                                num_beams=num_return_sequences,
                                num_return_sequences=num_return_sequences,

                                #repetition_penalty = 5.0
                                #force_word_ids = batch_tables_used
                                )
        #print(generator_outputs.shape)
        generator_outputs = generator_outputs.view(len(batch_prompts),
                                           num_return_sequences,
                                           generator_outputs.shape[1])
        #print(batch_inputs)
        #print(generator_outputs)

        batch_prompts.to("cpu")
        batch_mask.to("cpu")

        db_path = "./data/external/spider/database/"


        sql = test_sqls(db_path,
                        [{v: k for k, v in all_lookup_tables[l].items()} for l in
                         range(i, i + len(batch_prompts))],
                        [{v: k for k, v in all_lookup_columns[l].items()} for l
                         in
                         range(i, i + len(batch_prompts))],
                        generator_outputs,
                        batch_db_ids,
                        tokenizer,
                        [{v: k for k, v in table_names[l].items()} for l in range(i,i+len(batch_prompts))],
                        [{v: k for k, v in column_maps[l].items()}  for l in range(i,i+len(batch_prompts))],
        batch_fks,
                        batch_schemas,
                        batch_inputs
                        )
        sqls.extend(sql)
        #print(sql)
        #outputs.extend(output)



    resps = []
    for i in tqdm(range(len(sqls))):
        decoded_output = sqls[i].split(";")[0]
        resp = f"{decoded_output}\t{db_ids[i]}"
        #resp = replace_tabs_and_spaces(resp).strip()
        #print(resp)

        resps.append(resp)
    with open(os.path.join(output_filepath,f"{data_type}_pred.sql"), "w",encoding="utf-8") as myfile:
            myfile.write(u"\n".join(resps))


        #print(resp)
            #print()

        #print("=========")



    # Initialize an empty DataFrame to store the merged data


    # input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
    #
    #
    # model.to('cuda')
    # #target_encoding.to('cuda')
    # encoding.to('cuda')
    #
    #
    # outputs = model.generate(input_ids.to('cuda'), max_new_tokens = 10000)
    # print("after outputs")
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    # print(tokenizer.decode(outputs[1], skip_special_tokens=True))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
