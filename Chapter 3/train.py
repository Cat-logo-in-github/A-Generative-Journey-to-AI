import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Making a Basic Generative Model

def build_generator():
    input_layer = Input(shape=(10,))
    x = Dense(20, activation='relu')(input_layer)
    x = Dense(10, activation='relu')(x)
    generated_number = Dense(28 * 28, activation='sigmoid')(x)
    return Model(input_layer, generated_number)

generator = build_generator()
optimizer = Adam(learning_rate=0.001)
generator.compile(optimizer=optimizer, loss='mse')

# Training the Generator with Random Noise
target_value = 0.5
epochs = 5000
batch_size = 32
losses = []

for epoch in range(epochs):
    noise = np.random.normal(0, 1, (batch_size, 10))
    target = np.full((batch_size, 28 * 28), target_value)
    loss = generator.train_on_batch(noise, target)
    losses.append(loss)

    if (epoch + 1) % 500 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')

# Visualize the Loss
plt.plot(losses)
plt.title('Generator Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Visualize Real vs Generated data

test_noise = np.random.normal(0, 1, (1000, 10))

generated_numbers = generator.predict(test_noise)

# Use only one generated value per sample
generated_numbers = generated_numbers[:, 0]

# Real target values
real_numbers = np.full((1000,), target_value)

plt.figure(figsize=(10, 6))

plt.hist(
    real_numbers,
    bins=20,
    color='red',
    edgecolor='black',
    alpha=0.5,
    label='Target Numbers'
)
plt.hist(
    generated_numbers,
    bins=20,
    color='blue',
    edgecolor='black',
    alpha=0.5,
    label='Generated Numbers'
)


plt.title("Real vs Generated Data")
plt.xlabel("Number Value")
plt.ylabel("Frequency")
plt.legend()

plt.grid(True)

# Visualize Generated vs Target Values for the first 100 samples (more clarity)
plt.figure(figsize=(12, 5))

plt.plot(generated_numbers[:100],
         'bo',
         label='Generated')

plt.plot(real_numbers[:100],
         'r_',
         markersize=12,
         label='Target')

plt.title("Generated vs Target Values")

plt.xlabel("Sample Index")
plt.ylabel("Value")

plt.legend()
plt.grid(True)

plt.show()

plt.show()