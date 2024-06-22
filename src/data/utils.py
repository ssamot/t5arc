import numpy as np
import os
import json
import keras





def pad_array_with_random_position(small_array, m):
    """
    Embeds a small array of any shape into an mxm array at a random position.

    Parameters:
    small_array (numpy array): The small array to be embedded.
    m (int): The size of the larger mxm array.

    Returns:
    numpy array: The mxm array with the small array embedded at a random position.
    """
    small_shape = small_array.shape
    if any(dim > m for dim in small_shape):
        raise ValueError("The larger array must be at least as large as the dimensions of the small array.")

    # Create the larger mxm array filled with zeros (or any other background value)
    large_array = np.zeros((m, m))

    # Determine the random position for the top-left corner of the small array
    max_row_position = m - small_shape[0]
    max_col_position = m - small_shape[1]
    random_row = np.random.randint(0, max_row_position + 1)
    random_col = np.random.randint(0, max_col_position + 1)

    # Place the small array into the random position in the mxm array
    large_array[random_row:random_row + small_shape[0], random_col:random_col + small_shape[1]] = small_array

    return large_array


def load_data_for_generator(json_dir, solver_dir, max_examples = 20):
    json_data_list = []
    solver_list = []

    # Iterate over each file in the directory
    for filename in os.listdir(json_dir):
        if filename.endswith('.json') and (not filename.startswith(".")):
            file_path = os.path.join(json_dir, filename)
            # print(filename)
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
        # print(len(data["train"]), i)
        # train[i] =

        for j in range(len(data["train"])):
            example = data["train"][j]
            # print(data["train"][j])
            # print(train[i])
            train[i].append(example["input"])
            train[i].append(example["output"])

        train[i] = [np.array(e) + 1 for e in train[i]]
        train[i] = train[i] + [[[0, 0], [0, 0]] for _ in range(max_examples - len(train[i]))]

        # print(train[i])
        # exit([e.shape for e in train[i]])
        # train[i] = pad(32, train[i])

        #train[i] = [pad_array_with_random_position(e, max_size) for e in train[i]]
        test[i] = dict(data["test"])
        # print(train[i])

    #train = np.array(train)
    #train = train[:, :, np.newaxis, :, :]
    #train = np.transpose(train, (0, 1, 3, 4, 2))

    # print(train.shape)
    return train, test, solver_list


def load_data(json_dir, solver_dir, max_examples = 20):
    json_data_list = []
    solver_list = []

    # Iterate over each file in the directory
    for filename in os.listdir(json_dir):
        if filename.endswith('.json') and (not filename.startswith(".")):
            file_path = os.path.join(json_dir, filename)
            #print(filename)
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
        #train[i] = pad(32, train[i])

        train[i] = [pad_array_with_random_position(e, 32) for e in train[i]]
        test[i] = dict(data["test"])
        # print(train[i])

    train = np.array(train)
    train = train[:,:,np.newaxis,:, :]
    train = np.transpose(train, (0,1, 3,4,2))

    #print(train.shape)
    return train, test, solver_list

