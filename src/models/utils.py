import numpy as np
import os
import json
import keras


def masked_categorical_crossentropy(y_true, y_pred):
    # find out which timesteps in `y_true` are not the padding character '#'
    mv = np.array([0] * 74)
    mv[4] = 1
    mask = keras.ops.equal(y_true, mv)
    mask = 1 - keras.ops.cast(mask, "float32")

    # multiply categorical_crossentropy with the mask
    loss = keras.losses.categorical_crossentropy(y_true, y_pred) * mask

    # take average w.r.t. the number of unmasked entries
    return keras.ops.sum(loss) / keras.ops.sum(mask)
def custom_loss(y_true, y_pred):

    #find which values in yTrue (target) are the mask value

    print(y_true)
    #isMask = keras.ops.equal(yTrue, mv) #true for all mask values
    indices = keras.ops.where(y_true == mv)
    mask = keras.ops.take_along_axis(y_true, indices)
    print(mask)
    return keras.losses.categorical_crossentropy(y_true, y_pred)
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


import numpy as np


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
        #train[i] = pad(32, train[i])

        train[i] = [pad_array_with_random_position(e, 32) for e in train[i]]
        test[i] = dict(data["test"])
        # print(train[i])

    train = np.array(train)
    train = train[:,:,np.newaxis,:, :]
    train = np.transpose(train, (0,1, 3,4,2))

    #print(train.shape)
    return train, test, solver_list


def generate_text(model, images, start_token, max_len=20):
    # Initialize the input sequence with the start token
    input_seq = np.array([[start_token]])

    # Loop to generate each token
    for _ in range(max_len - 1):
        # Predict the next token
        predictions = model.predict(images + [input_seq], verbose=0)

        # Get the token with the highest probability
        next_token = np.argmax(predictions[0, -1, :])

        # Append the predicted token to the input sequence
        input_seq = np.append(input_seq, [[next_token]], axis=1)

        # If the predicted token is the end token, break the loop
        if next_token == 0:  # Assuming '0' is the end token
            break

    return input_seq[0]


# # Example usage
# start_token = 1  # Assuming '1' is the start token
# generated_sequence = generate_text(model, [images[i][:1] for i in range(6)], start_token)
# print("Generated sequence:", generated_sequence)

