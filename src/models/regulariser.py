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

        epsilon = 1e-8  # Small perturbation value
        reg = keras.random.normal(X.shape,0,  epsilon, seed = self.seed)


        X += reg
        #print(X.shape)
       # exit()

        X_b = ops.concatenate([ops.ones((ops.shape(X)[0], 1)), X], axis=1)

        # Step 1: Perform SVD on X_b
        U, s, Vt = ops.linalg.svd(X_b, full_matrices=False)

    # Calculate theta (coefficients)

        if(self.alpha > 0.0000001):
            s_reg = s / (s ** 2 + self.alpha)
            s_inv = ops.diag(s_reg)
        else:
            s_inv = ops.diag(1.0 / s)


        theta = ops.dot(ops.transpose(Vt),
                              ops.dot(s_inv, ops.dot(ops.transpose(U), y)))

        y_pred = ops.dot(X_b, theta)
        #print(y_pred.shape, theta.shape, X_b.shape)
        #exit()
        metric = r2(y, y_pred)
        average_metric = keras.ops.mean(metric)

        self.theta = theta
        return theta, metric, average_metric


    def predict(self, X):
        epsilon = 1e-2  # Small perturbation value
        #reg = keras.random.normal(X.shape, 0, epsilon, seed=self.seed)

        #X += reg
        X_b = ops.concatenate([ops.ones((ops.shape(X)[0], 1)), X], axis=1)
        y_pred = ops.dot(X_b, self.theta)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        print(self.theta.shape, "theta")
        return r2_score(y, y_pred)


class AdditionRegression:

    def __init__(self):
        pass


    def fit(self, X, y):
        theta = ops.mean(y - X, axis=0)
        y_pred = X + theta
        mse = r2(y, y_pred)
        s_mse = ops.mean(mse)

        self.theta = theta
        #print(theta)


        return theta, mse, s_mse




    def predict(self, X):
        y_pred = X + self.theta
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        print(self.theta.shape, "theta")
        return r2_score(y, y_pred)


@keras.saving.register_keras_serializable(package="models")
class CustomSplitRegularizer(keras.layers.Layer):
    def __init__(self, regularization_type, **kwargs):
        super(CustomSplitRegularizer, self).__init__(**kwargs)
        self.regularization_type = regularization_type
        self.constant_loss_value = keras.ops.ones([1])*1000.0


    def call(self, x):
        # Get the number of features
        x_left, x_right = x

        # Use dummy inputs to influence the regularization (example)
        # Here, just a demonstration of using these parameters
        if self.regularization_type == 'lr':
            clf = SVDLinearRegression(0.001)
        else:
            clf = AdditionRegression()

        theta,_, error = clf.fit(x_right, x_left )

        #print(error, theta.shape)

        x_left = clf.predict(x_right)

        #self.add_loss(error * 0.1)

        return x_left

    # def compute_output_shape(self, input_shape):
    #     # The output shape will be the same as the input shape except for the last dimension
    #     return (input_shape[0], input_shape[-1] // 2)


    def get_config(self):
        return {
            "regularization_type": self.regularization_type,

        }

    def compute_output_shape(self, input_shape):
        # The output shape will be the same as the input shape except for the last dimension
        return input_shape[0]



if __name__ == '__main__':

    X = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [7, 8, 9]])

    y = np.array([[2, 3, 4],
                  [5, 6, 7],
                  [8, 9, 10],
                  [8, 9, 10.5]])

    # X = np.random.random(size = (10,3))
    # y = np.random.random(size=(10, 3))

    clf = AdditionRegression()
    theta_val, mse_val, s_mse = clf.fit(X, y)

    print("Coefficients (theta):", theta_val)
    print("Mean Squared Error (MSE):", mse_val)
    print("S Mean Squared Error (MSE):", s_mse)

    #exit()

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
    clf = SVDLinearRegression(0.001)
    theta_val, mse_val, s_mse = clf.fit(X, y)
    pr = clf.score(X,y)
    # print("predict", pr)


    print("Coefficients (theta):", theta_val.shape)
    print("Mean Squared Error (MSE):", pr)
    print("S Mean Squared Error (MSE):", s_mse)
