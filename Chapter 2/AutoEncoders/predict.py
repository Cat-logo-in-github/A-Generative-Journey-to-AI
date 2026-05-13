import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1. LOAD FASHION-MNIST DATA
# ============================================================

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

X_test = X_test / 255.0

# ============================================================
# 2. LOAD TRAINED CNN AUTOENCODER
# ============================================================
# Make sure this file exists:
# cnn_autoencoderConv.keras

model = tf.keras.models.load_model("cnn_autoencoderConv.keras")

# ============================================================
# 3. ADD NOISE FUNCTION
# ============================================================

noise = np.random.random(X_test.shape) / 4
X_test_noisy = X_test + noise
X_test_noisy = np.clip(X_test_noisy, 0.0, 1.0)

# ============================================================
# 4. PREDICTIONS ON FIRST 8 IMAGES
# ============================================================

predictions = model.predict(X_test_noisy[:8])

# ============================================================
# 5. VISUALIZE (2 x 8 GRID EXACTLY LIKE YOUR CODE)
# ============================================================

plt.figure(figsize=(12, 3))

for i in range(8):

    # Noisy images (top row)
    plt.subplot(2, 8, i + 1)
    plt.imshow(X_test_noisy[i], cmap='binary')
    plt.axis("off")

    # Reconstructed images (bottom row)
    plt.subplot(2, 8, i + 9)
    plt.imshow(predictions[i], cmap='binary')
    plt.axis("off")

plt.tight_layout()
plt.show()

# ============================================================
# 6. SINGLE IMAGE CHECK (optional but useful)
# ============================================================

idx = 700

single_pred = model.predict(X_test_noisy[idx].reshape(1, 28, 28))[0]

plt.figure()
plt.imshow(X_test_noisy[idx], cmap='binary')
plt.title("Noisy Image")
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(single_pred, cmap='binary')
plt.title("CNN Reconstruction")
plt.axis("off")
plt.show()