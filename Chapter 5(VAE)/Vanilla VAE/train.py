import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.datasets import mnist
from keras.layers import Input, Dense, Layer
from keras.models import Model
import keras.ops as ops
from keras.losses import binary_crossentropy

# -----------------------
# Load data
# -----------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape((len(x_train), -1))
x_test = x_test.reshape((len(x_test), -1))

# -----------------------
# Params
# -----------------------
original_dim = 784
intermediate_dim = 512
latent_dim = 50
batch_size = 100
epochs = 50
beta=0.2  # KL weight (beta-VAE)

# -----------------------
# Sampling layer
# -----------------------
class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

# -----------------------
# KL loss layer (IMPORTANT)
# -----------------------
class KLLossLayer(Layer):
    def call(self, inputs):
        z, z_mean, z_log_var = inputs

        kl = -0.5 * ops.sum(
            1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var),
            axis=-1
        )

        kl *= beta # apply beta weight
        self.add_loss(ops.mean(kl))

        return z

# -----------------------
# Encoder
# -----------------------
inputs = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation="relu")(inputs)

z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

z = Sampling()([z_mean, z_log_var])
z = KLLossLayer()([z, z_mean, z_log_var])

# -----------------------
# Decoder
# -----------------------
decoder_h = Dense(intermediate_dim, activation="relu")
decoder_out = Dense(original_dim, activation="sigmoid")

h_decoded = decoder_h(z)
outputs = decoder_out(h_decoded)

# -----------------------
# VAE model
# -----------------------
vae = Model(inputs, outputs)

# reconstruction loss only (KL already inside layer)
def reconstruction_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred)
    return ops.sum(loss, axis=-1)

vae.compile(optimizer="rmsprop", loss=reconstruction_loss)

# -----------------------
# Train
# -----------------------
vae.fit(
    x_train,
    x_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, x_test),
)

# -----------------------
# Generator
# -----------------------
decoder_input = Input(shape=(latent_dim,))
h = decoder_h(decoder_input)
out = decoder_out(h)
generator = Model(decoder_input, out)

# -----------------------
# Generate images
# -----------------------
n = 10
figure = np.zeros((28 * n, 28 * n))

z_sample = np.random.normal(size=(n * n, latent_dim))
x_dec = generator.predict(z_sample, verbose=0)

for i in range(n):
    for j in range(n):
        figure[
            i * 28:(i + 1) * 28,
            j * 28:(j + 1) * 28
        ] = x_dec[i * n + j].reshape(28, 28)

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap="Greys_r")
plt.axis("off")
plt.savefig("generated_images.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------
# Reconstructions
# -----------------------
plt.figure(figsize=(20, 6))

for i in range(10):
    # original
    plt.subplot(3, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.axis("off")

    # recon
    recon = vae.predict(x_test[i:i+1], verbose=0)
    plt.subplot(3, 10, i + 1 + 10)
    plt.imshow(recon.reshape(28, 28), cmap="gray")
    plt.axis("off")

    # recon 2
    recon2 = vae.predict(x_test[i:i+1], verbose=0)
    plt.subplot(3, 10, i + 1 + 20)
    plt.imshow(recon2.reshape(28, 28), cmap="gray")
    plt.axis("off")

plt.savefig("reconstructions.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------
# Latent space
# -----------------------
encoder = Model(inputs, z_mean)
z_test = encoder.predict(x_test, verbose=0)

plt.figure(figsize=(8, 6))
plt.scatter(z_test[:, 0], z_test[:, 1], c=y_test, cmap="viridis", s=5)
plt.colorbar()
plt.title("Latent Space")
plt.savefig("Latent_Space.png", dpi=300, bbox_inches="tight")
plt.show()