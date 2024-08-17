import numpy as np
import keras


import tensorflow as tf
import keras
class SequenceAccuracy(keras.metrics.Metric):
    def __init__(self, name="sequence_accuracy", **kwargs):
        super(SequenceAccuracy, self).__init__(name=name, **kwargs)
        self.total_sequences = self.add_weight(name="total_sequences", initializer="zeros")
        self.correct_sequences = self.add_weight(name="correct_sequences", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to class indices if they are softmax outputs
        y_pred = keras.ops.argmax(y_pred, axis=-1)

        # Ensure y_true is in the same format (integer indices)
        y_true = keras.ops.cast(y_true, dtype='int64')

        # Check if all elements in the sequence are equal for each sample
        is_correct = keras.ops.cast(keras.ops.all(keras.ops.equal(y_true, y_pred), axis=-1), dtype='float32')

        # Update total and correct sequence counts
        self.total_sequences.assign_add(keras.ops.sum(keras.ops.ones_like(is_correct)))
        self.correct_sequences.assign_add(keras.ops.sum(is_correct))

    def result(self):
        # Compute the ratio of correct sequences to total sequences
        return self.correct_sequences / self.total_sequences

    def reset_states(self):
        # Reset state variables at the start of each epoch
        self.total_sequences.assign(0.0)
        self.correct_sequences.assign(0.0)



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

