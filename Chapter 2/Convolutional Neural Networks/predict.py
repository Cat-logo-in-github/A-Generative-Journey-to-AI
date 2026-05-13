import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load saved model
model = tf.keras.models.load_model("cifar10_cnn_model.keras")

# Load CIFAR-10 test data
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize
x_test = x_test.astype('float32') / 255.0

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Take random samples
sample_indices = np.random.choice(x_test.shape[0], 10, replace=False)
sample_images = x_test[sample_indices]

# Predict
predictions = model.predict(sample_images)
predicted_classes = np.argmax(predictions, axis=1)

# Show images
plt.figure(figsize=(15, 3))

for i in range(len(sample_images)):
    plt.subplot(1, len(sample_images), i + 1)
    plt.imshow(sample_images[i])
    plt.title(class_names[predicted_classes[i]])
    plt.axis('off')

plt.tight_layout()
plt.show()