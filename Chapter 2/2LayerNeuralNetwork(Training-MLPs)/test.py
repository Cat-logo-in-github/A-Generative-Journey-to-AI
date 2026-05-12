import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation

# 1. Download and load the MNIST dataset
# This contains 60,000 training images and 10,000 test images
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Preprocess the data
# Convert pixel values to float32 and normalize them to the [0.0, 1.0] range
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

gray_scale = 255
x_train /= gray_scale
x_test /= gray_scale

# 3. Create the model architecture
model = Sequential()
model.add(Flatten(input_shape=(28, 28))) # Flatten 28x28 images to a 1D vector
model.add(Dense(128))                    # Hidden layer with 128 neurons
model.add(Activation('relu'))            # Non-linear activation
model.add(Dense(10))                     # Output layer for 10 digits (0-9)
model.add(Activation('softmax'))         # Probability distribution output

# 4. Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train the model
print("Starting training...")
history = model.fit(x_train, y_train, epochs=5)

# 6. Get weights from first Dense layer
weights = model.layers[1].get_weights()[0]
print(weights.shape)

# 7. Get second layer weights
second_weights = model.layers[3].get_weights()[0]

print(second_weights.shape)


# 8. Visualize first layer neurons

fig, axes = plt.subplots(10, 3, figsize=(8, 20))

for digit in range(10):

    # weights from hidden neurons → this digit
    digit_weights = second_weights[:, digit]

    # indices of most positive neurons
    top_two = np.argsort(digit_weights)[-2:]

    # least important / most negative
    least_one = np.argsort(digit_weights)[0]

    selected = [
        (top_two[1], "Top 1"),
        (top_two[0], "Top 2"),
        (least_one, "Least")
    ]

    for col, (neuron_idx, label) in enumerate(selected):

        # get corresponding first-layer feature
        neuron_image = weights[:, neuron_idx].reshape(28, 28)

        ax = axes[digit, col]

        ax.imshow(neuron_image, cmap='seismic')
        ax.set_title(f'D{digit} | {label}\tN{neuron_idx}', fontsize=6)

        ax.axis('off')

plt.tight_layout()

# 8. Visualize second layer weights

plt.figure(figsize=(12, 8))

plt.imshow(second_weights, cmap='seismic', aspect='auto')

plt.colorbar(label='Weight Value')

plt.xlabel('Digit Class (0-9)')
plt.ylabel('Hidden Neuron Index')

plt.title('Weights from Hidden Layer → Output Layer')

# 9. Evaluate the model
print("\nEvaluating on test data...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nFinal Test Accuracy: {test_acc*100:.2f}%')

print("\nGenerating predictions for the first 5 test images...")
predictions = model.predict(x_test[:5])

# 10. Visualize the predictions
plt.figure(figsize=(10, 3))
for i in range(5):
    # Get the index of the highest probability
    predicted_label = np.argmax(predictions[i])
    actual_label = y_test[i]
    
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Pred: {predicted_label}\nActual: {actual_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()