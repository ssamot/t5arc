import numpy as np
import os
import json


def pad(size, array_list):
    padded_arrays = []
    target_shape = (size, size)

    for array in array_list:
        if array.shape[0] > size or array.shape[1] > size:
            raise ValueError("One of the arrays is larger than the target shape of 32x32.")

        # Create an array filled with -1
        padded_array = np.full(target_shape, -1)

        # Copy the original array into the top-left corner of the padded array
        padded_array[:array.shape[0], :array.shape[1]] = array

        padded_arrays.append(padded_array)

    return padded_arrays


def load_data(json_dir, solver_dir, max_examples = 20):
    json_data_list = []
    solver_list = []

    # Iterate over each file in the directory
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(json_dir, filename)
            solver_file = filename.split(".json")[0] + ".py"


            # Open and read the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
                json_data_list.append(data)

            with open(os.path.join(solver_dir, solver_file), 'r') as file:
                data = file.read()
                solver_list.append(data)

    train = [[] for _ in range(len(json_data_list))]
    test = [[] for _ in range(len(json_data_list))]
    for i, data in enumerate(json_data_list):
        #print(len(data["train"]), i)
        # train[i] =

        for j in range(len(data["train"])):
            example = data["train"][j]
            # print(data["train"][j])
            # print(train[i])
            train[i].append(example["input"])
            train[i].append(example["output"])

        train[i] = train[i] + [[[-1, -1], [-1, -1]] for _ in range(max_examples - len(train[i]))]
        train[i] = [np.array(e) for e in train[i]]
        # print(train[i])
        # exit([e.shape for e in train[i]])
        train[i] = pad(32, train[i])
        test[i] = dict(data["test"])
        # print(train[i])

    train = np.array(train)
    train = train[:,:,np.newaxis,:, :]
    train = np.transpose(train, (1,0,2,3,4))

    print(train.shape)
    return train, test, solver_list

def decode_sequence(images, encoder_model, decoder_model,
                    num_decoder_tokens,
                    target_token_index,
                    reverse_target_char_index,
                    max_decoder_seq_length):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(images)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence
