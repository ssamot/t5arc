import keras
import numpy as np
from keras import ops
from sklearn.metrics import r2_score
from lm import nrmse, r2

class SVDLinearRegression:

    def __init__(self, alpha):
        self.alpha = alpha
        self.seed = keras.random.SeedGenerator(seed=1337)


    def fit(self, X, y):

        epsilon = 1e-4  # Small perturbation value

        #print(epsilon * keras.random.uniform(X.shape, seed = self.seed))


        X += epsilon * keras.random.uniform(X.shape, seed = self.seed)
        X_b = ops.concatenate([ops.ones((ops.shape(X)[0], 1)), X], axis=1)

        # Step 1: Perform SVD on X_b
        U, s, Vt = ops.linalg.svd(X_b, full_matrices=False)

        # Calculate theta (coefficients)

        if(self.alpha > 0.0000001):
            s_reg = s / (s ** 2 + self.alpha)
            s_inv = ops.diag(s_reg)
            #print("REGULAR")
        else:
            s_inv = ops.diag(1.0 / s)

        theta = ops.dot(ops.transpose(Vt),
                              ops.dot(s_inv, ops.dot(ops.transpose(U), y)))

        y_pred = ops.dot(X_b, theta)
        metric = r2(y, y_pred)
        average_metric = keras.ops.mean(metric)

        self.theta = theta
        return theta, metric, average_metric


    def predict(self, X):
        X_b = ops.concatenate([ops.ones((ops.shape(X)[0], 1)), X], axis=1)
        y_pred = ops.dot(X_b, self.theta)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        print(self.theta.shape, "theta")
        return r2_score(y, y_pred)




def addition(X,y):
    theta = ops.mean(y - X, axis=0)
    y_pred = X + theta
    mse = r2(y,y_pred)
    s_mse = ops.mean(mse)

    return theta, mse, s_mse

@keras.saving.register_keras_serializable(package="models")
class CustomSplitRegularizer(keras.layers.Layer):
    def __init__(self, regularization_type, coef, **kwargs):
        super(CustomSplitRegularizer, self).__init__(**kwargs)
        self.regularization_type = regularization_type
        self.coef = coef
        self.constant_loss_value = keras.ops.ones([1])*1000.0


    def call(self, x):
        # Get the number of features
        x_left, x_right = x

        # Use dummy inputs to influence the regularization (example)
        # Here, just a demonstration of using these parameters
        if self.regularization_type == 'lr':
            clf = SVDLinearRegression(0.001)
            #print("LR")
            _, _, error = clf.fit(x_right, x_left )
            x_left = clf.predict(x_right)
        else:
           _,_, error = addition(x_right, x_left )


        # Return a dummy regularization term based on parameters
        #loss = self.coef * error
        #print(loss)
        #self.add_loss(loss)

        return x_left

    # def compute_output_shape(self, input_shape):
    #     # The output shape will be the same as the input shape except for the last dimension
    #     return (input_shape[0], input_shape[-1] // 2)


    def get_config(self):
        return {
            "regularization_type": self.regularization_type,
            "coef": self.coef
        }

    def compute_output_shape(self, input_shape):
        # The output shape will be the same as the input shape except for the last dimension
        return input_shape[0]

@keras.saving.register_keras_serializable(package="models")
class HalfNeuronsLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(HalfNeuronsLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Ensure the input tensor has more than one dimension

        # Get the number of features (last dimension)
        num_features = keras.ops.shape(inputs)[-1]

        # Ensure the number of features is even
        if num_features % 2 != 0:
            raise ValueError("Number of features must be even to split into two equal halves")

        # Calculate the split point
        split_point = num_features // 2

        # Split the tensor into two halves
        x_left = inputs[:, :split_point]
        # Optionally, you could return x_right instead if you want the other half
        return x_left

    def compute_output_shape(self, input_shape):
        # The output shape will be the same as the input shape except for the last dimension
        return (input_shape[0], input_shape[-1] // 2)


    def get_config(self):
        return {}


if __name__ == '__main__':

    # X = np.array([[1, 2, 3],
    #               [4, 5, 6],
    #               [7, 8, 9],
    #               [7, 8, 9]])
    #
    # y = np.array([[2, 3, 4],
    #               [5, 6, 7],
    #               [8, 9, 10],
    #               [8, 9, 10]])

    # X = np.random.random(size = (10,3))
    # y = np.random.random(size=(10, 3))
    #
    # theta_val, mse_val, s_mse = addition(X, y)
    #
    # print("Coefficients (theta):", theta_val)
    # print("Mean Squared Error (MSE):", mse_val)
    # print("S Mean Squared Error (MSE):", s_mse)

    X = np.random.rand(100, 3).astype(np.float32)  # Input features
    theta = np.array([[3,5,6.0], [1,2,3]]).T
    print("theta", theta.shape)
    y = np.dot(X, theta)


    if len(np.shape(y)) == 1:
        y = np.reshape(y, (-1, 1))

    print(X.shape, y.shape)
    # # Convert numpy arrays to TensorFlow tensors
    # X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
    # y_tf = tf.convert_to_tensor(y, dtype=tf.float32)

    # Call the function
    clf = SVDLinearRegression(0.01)
    theta_val, mse_val, s_mse = clf.fit(X, y)
    pr = clf.score(X,y)
    # print("predict", pr)


    print("Coefficients (theta):", theta_val)
    print("Mean Squared Error (MSE):", pr)
    print("S Mean Squared Error (MSE):", s_mse)
