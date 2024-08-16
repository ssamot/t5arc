import numpy as np
import keras


class SequenceAccuracy(keras.metrics.Metric):
    def __init__(self, name="sequence_accuracy", **kwargs):
        super(SequenceAccuracy, self).__init__(name=name, **kwargs)
        self.total_sequences = self.add_weight(name="total_sequences", initializer="zeros")
        self.correct_sequences = self.add_weight(name="correct_sequences", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to binary or integer outputs (assume argmax for categorical predictions)
        y_pred = np.argmax(y_pred, axis=-1)

        # Ensure y_true and y_pred are numpy arrays
        y_true = np.array(y_true)

        # Check if the entire sequence is correct
        correct_predictions = np.all(np.equal(y_true, y_pred), axis=-1)  # Shape: (batch_size,)

        # Update the total and correct sequence counts
        self.total_sequences.assign_add(np.float32(y_true.shape[0]))  # Batch size
        self.correct_sequences.assign_add(np.sum(correct_predictions.astype(np.float32)))

    def result(self):
        # Calculate accuracy as the ratio of fully correct sequences
        return self.correct_sequences / self.total_sequences

    def reset_states(self):
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

