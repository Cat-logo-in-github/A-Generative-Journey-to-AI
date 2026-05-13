import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# Download fashion mnist dataset and normalize the pixel values to be between 0 and 1
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define the encoder-decoder architectures
encoder = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(30, activation='relu')
])

decoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(28 * 28, activation='sigmoid'),
    tf.keras.layers.Reshape([28, 28])
])

# Combine encoder and decoder into an autoencoder model
autoencoder = tf.keras.models.Sequential([encoder, decoder])

# Compile and train the autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_train, X_train, epochs=10)

# Visualize the original and reconstructed images
plt.imshow(X_test[100], cmap='binary')
plt.imshow(autoencoder.predict(X_test[100].reshape((1, 28, 28))).reshape((28, 28)), cmap='binary')
plt.show()

#Add random noise to the input images and visualize the noisy and denoised images
noise = np.random.random((28, 28)) /4
X_train_noisy = X_train + noise
X_test_noisy = X_test + noise
X_train_noisy = np.clip(X_train_noisy, 0.0, 1.0)
X_test_noisy = np.clip(X_test_noisy, 0.0, 1.0)

autoencoder.fit(X_train_noisy, X_train, epochs=10)
autoencoder.save("cnn_autoencoderDense.keras")

for i in range(8):
    plt.subplot(2, 8, i + 1)
    plt.imshow(X_test_noisy[i], cmap='binary')

    plt.subplot(2, 8, i + 9)
    plt.imshow(autoencoder.predict(X_test_noisy[i].reshape((1, 28, 28))).reshape((28, 28)), cmap='binary')

plt.tight_layout()
plt.show()

plt.imshow(X_test_noisy[700], cmap='binary')
pred = autoencoder.predict(X_test_noisy[700].reshape((1, 28, 28))).reshape((28, 28))
plt.imshow(pred, cmap='binary')

plt.show()

# Define a convolutional encoder-decoder architecture
encoder = tf.keras.models.Sequential([
    tf.keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=2),
])

decoder = tf.keras.models.Sequential([
    tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=2, padding='valid', activation='relu', input_shape=(3, 3, 64)),
    tf.keras.layers.Conv2DTranspose(16, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
    tf.keras.layers.Conv2DTranspose(1, kernel_size=(3, 3), strides=2, padding='same', activation='sigmoid'),
    tf.keras.layers.Reshape((28, 28))
])

autoencoder = tf.keras.models.Sequential([encoder, decoder])

# Combine and train the convolutional autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_train_noisy, X_train, epochs=10)

autoencoder.save("cnn_autoencoderConv.keras")