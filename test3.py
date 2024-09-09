import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import fetch_openml

# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X = mnist.data.astype('float32') / 255.0

# Define the autoencoder model
input_dim = 784
encoding_dim = 49

inputs = keras.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='linear', use_bias=False,
                       kernel_constraint=keras.constraints.NonNeg())(inputs)
decoded = layers.Dense(input_dim, activation='sigmoid', use_bias=False,
                       kernel_constraint=keras.constraints.NonNeg())(encoded)

autoencoder = keras.Model(inputs, decoded)
encoder = keras.Model(inputs, encoded)


autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X, X, epochs=500, batch_size=256, shuffle=True, validation_split=0.2)

# Get the encoder weights (components)
components = encoder.layers[1].get_weights()[0].T

# Visualize the components
fig, axes = plt.subplots(7, 7, figsize=(15, 15))
fig.suptitle("Autoencoder Components", fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < encoding_dim:
        ax.imshow(components[i].reshape(28, 28), cmap='gray')
        ax.axis('off')
    else:
        ax.remove()

plt.tight_layout()
plt.show()

# Visualize reconstructions of some digits
n_samples = 5
fig, axes = plt.subplots(2, n_samples, figsize=(15, 6))
fig.suptitle("Original vs Reconstructed Digits", fontsize=16)

for i in range(n_samples):
    original = X[i].reshape(28, 28)
    reconstructed = autoencoder.predict(X[i:i + 1]).reshape(28, 28)

    axes[0, i].imshow(original, cmap='gray')
    axes[0, i].set_title(f"Original {i + 1}")
    axes[0, i].axis('off')

    axes[1, i].imshow(reconstructed, cmap='gray')
    axes[1, i].set_title(f"Reconstructed {i + 1}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.show()