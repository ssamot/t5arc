import numpy as np
import os
import json
import keras



def generate_text(model, images, start_token = 4, max_len=20):
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
        if next_token == 1:  # Assuming '0' is the end token
            break

    return input_seq[0]


# # Example usage
# start_token = 1  # Assuming '1' is the start token
# generated_sequence = generate_text(model, [images[i][:1] for i in range(6)], start_token)
# print("Generated sequence:", generated_sequence)

