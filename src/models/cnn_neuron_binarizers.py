import keras
from keras import ops, random

epsilon = 0.000001

# 1. Passthrough (Step Function)

def passthrough_binarize(x):
    return ops.stop_gradient(
        ops.cast(ops.greater_equal(x, 0), "float32"))

# 2. Gumbel-Sigmoid
def gumbel_sigmoid(x, temperature=1.0):
    u = random.uniform(ops.shape(x))
    gumbel = -ops.log(-ops.log(u + epsilon) + epsilon)
    return ops.sigmoid((x + gumbel) / temperature)

# 3. Sigmoid with Entropy Regularizer
def sigmoid_with_entropy(x):
    y = ops.sigmoid(x)
    entropy = -y * ops.log(y + epsilon) - (1 - y) * ops.log(1 - y + epsilon)
    return y, entropy

def entropy_loss(x):
    entropy = -x * ops.log(x + epsilon) - (1 - x) * ops.log(1 - x + epsilon)
    entropy = 0.1* ops.mean(entropy)
    return entropy

# 4. Mean Squared Difference from 0.5
def mean_squared_diff_from_half(x):
    y = ops.sigmoid(x)
    msd = ops.mean(ops.square(y - 0.5))
    return y, msd

# Example usage in a custom layer
@keras.saving.register_keras_serializable()
class BinarizingNeuron(keras.layers.Layer):
    def __init__(self, method='passthrough', regularizer_weight=0.1, **kwargs):
        super(BinarizingNeuron, self).__init__(**kwargs)
        self.method = method
        self.regularizer_weight = regularizer_weight

    def call(self, inputs):
        if self.method == 'passthrough':
            return passthrough_binarize(inputs)
        elif self.method == 'gumbel_sigmoid':
            return gumbel_sigmoid(inputs)
        elif self.method == 'sigmoid_with_entropy':
            output, entropy = sigmoid_with_entropy(inputs)
            self.add_loss(self.regularizer_weight * ops.mean(entropy))
            return output
        elif self.method == 'mean_squared_diff':
            output, msd = mean_squared_diff_from_half(inputs)
            self.add_loss(self.regularizer_weight * msd)
            return output

    def get_config(self):
        return {
            "method": self.method,
            "regularizer_weight": self.regularizer_weight

        }

