import keras
import numpy as np
from keras import ops
from sklearn.metrics import r2_score
from lm import nrmse, r2
import jax.numpy as jnp
#import torch
import numpy as np
import jax


def callback(to_save):
    np.save("./models/broken.np", to_save)

class SVDLinearRegression:

    def __init__(self, alpha):
        self.alpha = alpha
        self.seed = keras.random.SeedGenerator(seed=1337)

    # def rmcorr_torch(self, X, threshold=0.95):
    #     corr_matrix = torch.abs(torch.corrcoef(X.T))
    #
    #     upper_tri = torch.triu(corr_matrix, diagonal=1)
    #     # print(upper_tri)
    #
    #     # Create a mask for correlated features
    #     correlated = torch.any(upper_tri > threshold, axis=0)
    #
    #     keep = torch.logical_not(correlated)
    #
    #
    #     # keep = keep.at[jnp.argmax(correlated)].set(True)
    #
    #     return X[:, keep]  # , jnp.where(keep)[0]
    #
    def rmcorr(self,X, threshold):
        corr_matrix = jnp.abs(jnp.corrcoef(keras.ops.transpose(X)))


        upper_tri = jnp.triu(corr_matrix, k=1)
        # print(upper_tri)

        # Create a mask for correlated features
        correlated = jnp.any(upper_tri > threshold, axis=0)



        keep = jnp.logical_not(correlated)

        self.kept = jnp.sum(keep)
        #print(f"-={self.kept}=-")

        mask = jnp.broadcast_to(keep, X.shape)

        return jnp.where(
            mask, X, ops.zeros_like(X)
        )




    def fit(self, X, y):



        epsilon = 1e-12  # Small perturbation value
        reg = keras.random.normal(X.shape,0,  epsilon, seed = self.seed)


        X += reg
        X_b = ops.concatenate([ops.ones((ops.shape(X)[0], 1)), X], axis=1)


        #jax.debug.callback(callback=callback, to_save = X_b)


    #     # Step 1: Perform SVD on X_b
      #  U, s, Vt = ops.linalg.svd(X_b, full_matrices=False)

        theta  = ops.linalg.lstsq(X_b,y)


        #print(theta.shape)
        #exit()

        #P, L, U = jax.scipy.linalg.lu(X)

        #Q, R = jnp.linalg.qr(X_normalized)

        # Step 1: Solve Q^T z = y for z (since Q is orthogonal, Q^T is its inverse)
        #z = jnp.dot(Q.T, y_normalized)

        # Step 2: Solve the triangular system Rw = z for w
        #theta = jax.scipy.linalg.solve_triangular(R, z, lower=False)

        # First, solve LY = B using forward substitution
        #Y = jax.scipy.linalg.solve_triangular(L, y, lower=True)

        # Then solve UX = Y using backward substitution
        #theta = jax.scipy.linalg.solve_triangular(U, Y, lower=False)

        #
    # Calculate theta (coefficients)

        # if(self.alpha > 0.0000001):
        #     s_reg = s / (s ** 2 + self.alpha)
        #     s_inv = ops.diag(s_reg)
        # else:
        #     s_inv = ops.diag(1.0 / s)
    #
    #
        # theta = ops.dot(ops.transpose(Vt),
        #                       ops.dot(s_inv, ops.dot(ops.transpose(U), y)))

        #theta = torch.linalg.lstsq(X_b,y)[0]
        #print(X_b)

        #theta = jnp.linalg.lstsq(X_b,y, rcond=0)[0]

        #theta = jnp.where(jnp.isnan(theta), jnp.zeros_like(theta), theta)

        #print(jnp.any(jnp.isnan(theta)))

        # if(jnp.any(jnp.isnan(theta))):
        #     print("Nans")
            #theta = jnp.ones_like(theta)

        y_pred = ops.dot(X_b, theta)

        #print(y_pred.shape, theta.shape, X_b.shape)
        #exit()
        metric = r2(y, y_pred)
        average_metric = keras.ops.mean(metric)

        self.theta = theta
        return theta, metric, average_metric


    def predict(self, X):
        #X = self.rmcorr_torch(X, 0.95)

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

    y = np.array([[2, 4],
                  [5, 7],
                  [10,  10],
                  [8, 10.5]])


    # # X = np.random.random(size = (10,3))
    # # y = np.random.random(size=(10, 3))
    #
    # clf = AdditionRegression()
    # theta_val, mse_val, s_mse = clf.fit(X, y)
    #
    # print("Coefficients (theta):", theta_val)
    # print("Mean Squared Error (MSE):", mse_val)
    # print("S Mean Squared Error (MSE):", s_mse)
    #
    # #exit()

    # X = np.random.rand(100, 3).astype(np.float32)  # Input features
    # theta = np.array([[3,5,6.0], [1,2,3]]).T
    # print("theta", theta.shape)
    # y = np.dot(X, theta)
    #
    #
    # if len(np.shape(y)) == 1:
    #     y = np.reshape(y, (-1, 1))

    #X = np.load("./models/broken.np.npy")

    #y = X

    #print(X.shape, y.shape)

    #print(X.shape, y.shape)
    #exit()
    # # Convert numpy arrays to TensorFlow tensors
    # X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
    # y_tf = tf.convert_to_tensor(y, dtype=tf.float32)

    # Call the function

    from jax import jit
    clf = SVDLinearRegression(0.001)


    theta_val, mse_val, s_mse = clf.fit(X, y)
    #pr = clf.score(X,y)
    # print("predict", pr)


    print("Coefficients (theta):", theta_val.shape)
    #print("Mean Squared Error (MSE):", pr)
    print("S Mean Squared Error (MSE):", s_mse)
